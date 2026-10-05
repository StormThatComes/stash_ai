"""Only durable publication proofs survive replacement, without voting in active fusion."""
import sqlite3

import pytest

from stash_ai import db
from stash_ai.evidence.frozen_replay import PublishedRecords
from stash_ai.evidence.identity import marker_segment_signature
from stash_ai.storage.archive import records as archived_records
from stash_ai.storage.codec import encode
from stash_ai.storage.store import StateStore
from tests.test_result_thresholds import outputs


def proof(segment, *, nested=False):
    signature = marker_segment_signature(segment)
    item = {'source': 'wd', 'evidence_signature': signature}
    if nested:
        item = {'scene_evidence': [{'frame_context': {'wd': signature},
                                    'scene_context': {'wd': signature}}]}
    return {'contributions': [item]}


@pytest.mark.parametrize('location', ['published', 'pending', 'recovery', 'nested'])
@pytest.mark.parametrize('preview', [False, True])
def test_replacement_preserves_exact_pins_and_replay_while_active_results_are_latest(tmp_path, location, preview):
    path = tmp_path / 'state.sqlite3'
    old, new = outputs(signature='old'), outputs(signature='new')
    with StateStore(path, 'scope') as store:
        store.put('1', 'wd', old)
        saved = proof(old['markers'], nested=location == 'nested')
        if location in ('published', 'nested'):
            store.publish('1', {'markers': saved})
        else:
            store.prepare('1', {'markers': saved})
            if location == 'recovery':
                from stash_ai.storage.recovery import abandon
                abandon(store, lambda _: None)
    before = path.read_bytes()
    with StateStore(path, 'scope', dry_run=preview) as store:
        store.put('1', 'wd', new)
        assert store.model_record('1', 'markers', 'wd') == new['markers']
        archived = archived_records(store, '1', 'markers')['wd']
        signature = marker_segment_signature(old['markers'])
        assert marker_segment_signature(archived[0]) == signature
        assert not archived_records(store, '1', 'scene')  # no scene proof pinned
        exact = PublishedRecords(store, '1').matching('markers', {'wd': signature}, error='lost proof')
        assert marker_segment_signature(exact['wd']) == signature
        # Repeated replacements do not retain unreferenced intermediate inference.
        store.put('1', 'wd', outputs(signature='third'))
        assert len(archived_records(store, '1', 'markers')['wd']) == 1
        # A pending publication in the OTHER output can still pin this marker proof.
        store.prepare('1', {'scene': saved})
        store.publish('1', {'markers': {'contributions': []}})
        assert archived_records(store, '1', 'markers')
        store.publish('1', {'scene': {'contributions': []}})
        assert not archived_records(store, '1', 'markers')
    if preview:
        assert path.read_bytes() == before
    else:
        assert db.check_database(path, 'scope')['ok']


def test_archive_and_replacement_roll_back_together(tmp_path):
    with StateStore(tmp_path/'state.sqlite3', 'scope') as store:
        old = outputs(signature='old')
        store.put('1', 'wd', old)
        store.publish('1', {'markers': proof(old['markers'])})
        with pytest.raises(RuntimeError, match='rollback'):
            with store.connection:
                store.results.put_in_transaction('1', 'wd', outputs(signature='new'))
                raise RuntimeError('rollback')
        assert store.model_record('1', 'markers', 'wd') == old['markers']
        assert not archived_records(store, '1', 'markers')


def test_v13_migration_keeps_latest_and_only_pinned_history_across_scopes(tmp_path):
    path = tmp_path/'state.sqlite3'
    history = [outputs(signature=signature)['markers'] for signature in ('unpinned', 'pinned', 'latest')]
    for scope in ('first', 'second'):
        with StateStore(path, scope) as store:
            store.put('1', 'wd', {'markers': history[-1]})
            store.prepare('1', {'markers': proof(history[1])})
    with sqlite3.connect(path) as connection:
        connection.execute('UPDATE results SET data=?', (encode(history),))
        connection.execute('PRAGMA user_version=13')
    assert db.update_database(path) == (13, 20)
    for scope in ('first', 'second'):
        with StateStore(path, scope) as store:
            assert store.model_record('1', 'markers', 'wd') == history[-1]
            archived = archived_records(store, '1', 'markers')['wd']
            assert len(archived) == 1
            assert marker_segment_signature(archived[0]) == marker_segment_signature(history[1])
        assert db.check_database(path, scope)['ok']
    with sqlite3.connect(path) as connection:
        before = [connection.execute('SELECT * FROM ' + table).fetchall()
                  for table in ('results', 'inference_archive', 'snapshots')]
    assert db.update_database(path) == (20, 20)
    with sqlite3.connect(path) as connection:
        assert [connection.execute('SELECT * FROM ' + table).fetchall()
                for table in ('results', 'inference_archive', 'snapshots')] == before


def test_scene_frames_can_be_replayed_from_pinned_scene_archive(tmp_path):
    old = outputs(signature='old')
    from stash_ai.evidence.cache_views import as_frames
    signature = marker_segment_signature(as_frames(old['scene']))
    with StateStore(tmp_path/'state.sqlite3', 'scope') as store:
        store.put('1', 'wd', old)
        store.publish('1', {'scene': {'contributions': [{'frame_context': {'wd': signature}}]}})
        store.put('1', 'wd', outputs(signature='new'))
        matching = PublishedRecords(store, '1').matching('scene', {'wd': signature}, error='missing scene frames')
        frames = as_frames(matching['wd'])
        assert frames['mode'] == 'frames'
        assert marker_segment_signature(frames) == signature



def test_deep_check_rejects_archived_proof_signature_corruption(tmp_path):
    path = tmp_path/'state.sqlite3'
    old = outputs(signature='old')
    with StateStore(path, 'scope') as store:
        store.put('1', 'wd', old)
        store.publish('1', {'markers': proof(old['markers'])})
        store.put('1', 'wd', outputs(signature='new'))
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE inference_archive SET signature='tampered'")
    report = db.check_database(path, 'scope')
    assert [issue['code'] for issue in report['errors']] == ['state_payload_invalid']
