"""Only durable publication proofs survive replacement, without voting in active fusion."""
from stash_ai.storage.schema import LATEST_SCHEMA_VERSION
import sqlite3

import pytest

from stash_ai.db.checks import check_database
from stash_ai.db.update import update_database
from stash_ai.evidence.frozen_replay import PublishedRecords
from stash_ai.evidence.identity import marker_segment_signature
from stash_ai.storage.archive import records as archived_records
from stash_ai.storage.codec import encode
from stash_ai.storage.store import StateStore
from tests.test_result_thresholds import outputs


def proof(segment, *, nested=False):
    signature = marker_segment_signature(segment)
    item = dict(source='wd', tag='core', score=.9, start=0., end=1.,
                extra_tags=[], meta={'origin': 'test'}, evidence_signature=signature)
    if nested:
        item['extra_support'] = {'core': {'frame_context': {'wd': signature}}}
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
        assert check_database(path, 'scope')['ok']


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


def test_v18_upgrade_preserves_latest_record_and_existing_pinned_archive(tmp_path):
    path = tmp_path/'state.sqlite3'
    history = [outputs(signature=signature)['markers'] for signature in ('unpinned', 'pinned', 'latest')]
    for scope in ('first', 'second'):
        with StateStore(path, scope) as store:
            store.put('1', 'wd', {'markers': history[-1]})
            store.prepare('1', {'markers': proof(history[1])})
    with sqlite3.connect(path) as connection:

        for scope in ('first', 'second'):
            connection.execute('INSERT INTO inference_archive VALUES (?,?,?,?,?,?)', (scope, '1', 'wd', 'markers', marker_segment_signature(history[1]), encode(history[1])))
        connection.execute('PRAGMA user_version=18')
    assert update_database(path) == (18, LATEST_SCHEMA_VERSION)
    for scope in ('first', 'second'):
        with StateStore(path, scope) as store:
            assert store.model_record('1', 'markers', 'wd') == history[-1]
            archived = archived_records(store, '1', 'markers')['wd']
            assert len(archived) == 1
            assert marker_segment_signature(archived[0]) == marker_segment_signature(history[1])
        assert check_database(path, scope)['ok']
    with sqlite3.connect(path) as connection:
        before = [connection.execute('SELECT * FROM ' + table).fetchall()
                  for table in ('results', 'inference_archive', 'snapshots')]
    assert update_database(path) == (LATEST_SCHEMA_VERSION, LATEST_SCHEMA_VERSION)
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
    report = check_database(path, 'scope')
    assert {issue['code'] for issue in report['errors']} == {'state_payload_invalid', 'snapshot_proof_missing'}


@pytest.mark.parametrize('version', [18, 19, 20, 21])
def test_upgrade_removes_only_obsolete_archives_and_rebases_their_publication(tmp_path, version):
    from copy import deepcopy
    from stash_ai.storage.snapshots import FORMAT_KEY
    path = tmp_path/'state.sqlite3'
    current = outputs(signature='current')['markers']
    old = deepcopy(current)
    old['meta']['analysis_format'] = 3
    for scope in ('first', 'second'):
        with StateStore(path, scope) as store:
            store.connection.execute('INSERT INTO inference_archive VALUES (?,?,?,?,?,?)',
                (scope, '1', 'wd', 'markers', marker_segment_signature(old), encode(old)))
            store.connection.execute('INSERT INTO inference_archive VALUES (?,?,?,?,?,?)',
                (scope, '1', 'wd', 'markers', marker_segment_signature(current), encode(current)))
            snapshot = {**proof(old), 'owned_ids': ['marker']}
            snapshot['contributions'][0].update(tag='core', start=1., end=2., extra_ids=['old-tag-id'])
            blob = encode({FORMAT_KEY: 5, 'snapshot': snapshot, 'proofs': []})
            store.connection.execute('INSERT INTO snapshots VALUES (?,?,?,?,?)', (scope, '1', 'markers', blob, blob))
            store.connection.commit()
    with sqlite3.connect(path) as db:
        db.execute(f'PRAGMA user_version={version}')
    update_database(path)
    for scope in ('first', 'second'):
        with StateStore(path, scope) as store:
            assert archived_records(store, '1', 'markers')['wd'] == [current]
            for snapshot in store.snapshot('1', 'markers'):
                assert snapshot['owned_ids'] == ['marker']
                item = snapshot['contributions'][0]
                assert item['requires_rebase'] and 'extra_ids' not in item
                assert (item['tag'], item['start'], item['end']) == ('core', 1., 2.)
        assert check_database(path, scope)['ok']


def test_unknown_archive_format_rolls_back_retirement(tmp_path):
    from tests.helpers.evidence import inject_records
    path = tmp_path/'state.sqlite3'
    old = outputs(signature='old')
    for record in old.values():
        record['meta']['analysis_format'] = 3
    future = outputs(signature='future')['markers']
    future['meta']['analysis_format'] = 99
    with StateStore(path, 'scope') as store:
        inject_records(store, '1', 'wd', old)
        store.connection.execute('INSERT INTO inference_archive VALUES (?,?,?,?,?,?)',
            ('scope', '1', 'wd', 'markers', marker_segment_signature(future), encode(future)))
        store.connection.execute('PRAGMA user_version=21')
        store.connection.commit()
    with sqlite3.connect(path) as db:
        before = db.execute('SELECT * FROM results').fetchall()
    with pytest.raises(ValueError, match='Unsupported inference'):
        update_database(path)
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT * FROM results').fetchall() == before
        assert not db.execute('SELECT * FROM refresh_requests').fetchall()
        assert db.execute('PRAGMA user_version').fetchone() == (21,)
