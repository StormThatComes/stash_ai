"""Corrupt provenance fails early; unsorted timelines are retired without rewriting."""
from tests.helpers.evidence import empty_policies as empty_tag_policies
from copy import deepcopy
import json
import sqlite3

import pytest

from stash_ai.db.checks import check_database
from stash_ai.db.update import update_database
from stash_ai.evidence.formats import UnorderedFrames
from stash_ai.evidence.frames import FrameIndex
from stash_ai.evidence.identity import marker_segment_signature
from stash_ai.evidence.occurrences import occurrence_contributions
from stash_ai.storage import metadata
from stash_ai.storage.archive import records as archived_records
from stash_ai.storage.codec import encode
from stash_ai.storage.schema import LATEST_SCHEMA_VERSION
from stash_ai.storage.store import StateStore
from tests.helpers.evidence import inject_records, paired_records, policies, span
from tests.helpers.runtime import runtime as app

pytestmark = pytest.mark.usefixtures('reset_temporal')


def records():
    scenes, markers = paired_records({'wd': [.9, .95, .85]})
    return json.loads(json.dumps({'scene': scenes['wd'], 'markers': markers['wd']}))


@pytest.mark.parametrize('field', ['frame_input_signature', 'inference_signature', 'marker_threshold',
                                  'scene_threshold', 'marker_score_floor', 'scene_support_floor'])
@pytest.mark.parametrize('kind', ['scene', 'markers'])
def test_required_metadata_is_checked_on_write_read_and_derived_metadata(tmp_path, field, kind):
    current = records()[kind]
    broken = deepcopy(current)
    broken['meta'].pop(field)
    with StateStore(tmp_path/'state.sqlite3', 'scope') as store:
        store.put('1', 'wd', {kind: current})
        with pytest.raises(ValueError, match='requires'):
            store.put('1', 'wd', {kind: broken})
        assert store.model_record('1', kind, 'wd') == current
        summary = metadata.summarize(current)
        summary['meta'].pop(field)
        with pytest.raises(ValueError, match='requires'):
            metadata.decode(json.dumps(summary))
        # Emulate corruption outside the writer and its cache invalidation.
        inject_records(store, '1', 'wd', {kind: broken})
        with pytest.raises(ValueError, match='requires'):
            store.records('1', kind)


@pytest.mark.parametrize('field', ['frame_input_signature', 'marker_threshold', 'scene_threshold'])
def test_grouping_and_occurrences_fail_instead_of_silently_losing_model_votes(field):
    value = records()['markers']
    value['meta'].pop(field)
    before = deepcopy(value)
    with pytest.raises(ValueError, match='requires'):
        FrameIndex(app, {'wd': value}, policies(), tag_policies=empty_tag_policies())
    with pytest.raises(ValueError, match='requires'):
        occurrence_contributions(app, [], {'wd': value}, [], policies(), policies())
    assert value == before


@pytest.mark.parametrize('kind,field', [('markers', 'items'), ('scene', 'frames')])
def test_unsorted_current_records_are_rejected_without_replacing_valid_evidence(tmp_path, kind, field):
    current = records()[kind]
    broken = deepcopy(current)
    broken[field].reverse()
    with StateStore(tmp_path/'state.sqlite3', 'scope') as store:
        store.put('1', 'wd', {kind: current})
        with pytest.raises(UnorderedFrames, match='chronological'):
            store.put('1', 'wd', {kind: broken})
        assert store.model_record('1', kind, 'wd') == current
        assert not archived_records(store, '1', kind)
        inject_records(store, '1', 'wd', {kind: broken})
        with pytest.raises(UnorderedFrames):
            store.model_record('1', kind, 'wd')


@pytest.mark.parametrize('kind,field', [('markers', 'items'), ('scene', 'frames')])
def test_v22_refreshes_only_unsorted_records_and_keeps_all_publication_retry_state(tmp_path, kind, field):
    path = tmp_path/'state.sqlite3'
    current = records()
    broken = deepcopy(current[kind])
    broken[field].reverse()
    signature = marker_segment_signature(broken)
    saved = {'owned_ids': ['native-marker'], 'contributions': [
        {**span('wd', 1., 2., tag='core', extras=['extra']), 'evidence_signature': signature,
         'meta': broken['meta']}]}
    for scope in ('first', 'second'):
        with StateStore(path, scope) as store:
            store.put('1', 'wd', current)
            inject_records(store, '1', 'wd', {kind: broken})
            store.put('2', 'wd', current)
            store.publish('1', {'markers': saved})
            store.prepare('1', {'markers': saved})
            with store.connection:
                store.connection.execute('INSERT INTO recovery_publications VALUES (?,?,?,?,?)',
                    (scope, '1', 'markers', 'lost-response', store.pack_snapshot(saved)))
                for value in (broken, current[kind]):
                    store.connection.execute('INSERT INTO inference_archive VALUES (?,?,?,?,?,?)',
                        (scope, '1', 'wd', kind, marker_segment_signature(value), encode(value)))
    with sqlite3.connect(path) as db:
        db.execute('PRAGMA user_version=22')
        before = db.execute("SELECT * FROM results WHERE scene='2'").fetchall()
    raw = path.read_bytes()
    assert update_database(path, dry_run=True) == (22, LATEST_SCHEMA_VERSION)
    assert path.read_bytes() == raw
    assert update_database(path) == (22, LATEST_SCHEMA_VERSION)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT * FROM results WHERE scene='2'").fetchall() == before
    for scope in ('first', 'second'):
        with StateStore(path, scope) as store:
            assert store.model_record('1', kind, 'wd') is None
            other = 'scene' if kind == 'markers' else 'markers'
            assert store.model_record('1', other, 'wd') == current[other]
            assert list(store.connection.execute('SELECT scene,model,kind FROM refresh_requests WHERE scope=?',
                                                (scope,))) == [('1', 'wd', kind)]
            assert archived_records(store, '1', kind)['wd'] == [current[kind]]
            from stash_ai.storage.recovery import publications
            for snapshot in (*store.snapshot('1', 'markers'), *publications(store, '1', 'markers')):
                assert snapshot['owned_ids'] == ['native-marker']
                item = snapshot['contributions'][0]
                assert item['requires_rebase'] and 'build_options' not in item
                assert item['evidence_signature'] == signature
                assert (item['start'], item['end'], item['extra_tags']) == (1., 2., ['extra'])
        assert check_database(path, scope)['ok']


def test_missing_authoritative_metadata_rolls_back_upgrade_instead_of_guessing(tmp_path):
    path = tmp_path/'state.sqlite3'
    broken = records()['markers']
    broken['meta'].pop('frame_input_signature')
    with StateStore(path, 'scope') as store:
        inject_records(store, '1', 'wd', {'markers': broken})
        store.connection.execute('PRAGMA user_version=22')
        store.connection.commit()
    with sqlite3.connect(path) as db:
        before = db.execute('SELECT * FROM results').fetchall()
    with pytest.raises(ValueError, match='requires both input'):
        update_database(path)
    with sqlite3.connect(path) as db:
        assert db.execute('PRAGMA user_version').fetchone() == (22,)
        assert db.execute('SELECT * FROM results').fetchall() == before
        assert not db.execute('SELECT * FROM refresh_requests').fetchall()


def test_chronological_tiny_overlaps_remain_valid_and_occurrence_is_a_point(tmp_path):
    scenes, markers = paired_records({'wd': [0, .95, 0]}, threshold=.35, marker_threshold=.65)
    scenes['wd']['frames'][1]['start'] -= .001
    markers['wd']['items'][1]['start'] -= .001
    with StateStore(tmp_path/'state.sqlite3', 'scope') as store:
        store.put('1', 'wd', {'scene': scenes['wd'], 'markers': markers['wd']})
        from stash_ai.evidence.pipeline import evidence
        _, spans = evidence(app, store, '1', policies(), policies())
        assert len(spans) == 1 and spans[0]['end'] is None
        assert spans[0]['start'] == .999


def test_upgrade_then_reinference_reuses_marker_ids_and_preserves_manual_tags(pipeline, monkeypatch):
    from stash_ai.evidence import formats
    from stash_ai.inference import results
    from stash_ai.storage.store import scope_for
    p = pipeline
    capture = results.capture_outputs
    def old_capture(*args, **kwargs):
        value = capture(*args, **kwargs)
        value['markers']['items'].reverse()
        return value
    # Emulate the previous writer: raw timeline unsorted, marker build sorted.
    with monkeypatch.context() as previous:
        previous.setattr(formats, 'validate_current', lambda _: None)
        previous.setattr(results, 'capture_outputs', old_capture)
        assert app.main(p.args) == 0
    marker = p.stash.scene['scene_markers'][0]
    human = p.stash.tag('human')
    marker['tags'].append(human)
    ids = {item['id'] for item in p.stash.scene['scene_markers']}
    with sqlite3.connect(p.path) as db:
        db.execute('PRAGMA user_version=22')
    update_database(p.path)
    assert app.main(p.args) == 0
    assert len(p.loaded) == 2
    assert {item['id'] for item in p.stash.scene['scene_markers']} == ids
    assert human in next(item for item in p.stash.scene['scene_markers'] if item['id'] == marker['id'])['tags']
    with StateStore(p.path, scope_for(app.STASH_URL)) as store:
        assert not store.connection.execute('SELECT * FROM refresh_requests').fetchall()
        assert all(not item.get('requires_rebase') for item in store.snapshot('1', 'markers')[0]['contributions'])
    assert check_database(p.path, scope_for(app.STASH_URL))['ok']


@pytest.mark.parametrize('incomplete', [None, {}, {'wd': frozenset()},
                                       {'wd': [], 'dtq': [], 'joytag': [], 'oppai': []}])
def test_internal_policy_boundaries_reject_implicit_empty_bans(incomplete):
    from stash_ai.exclusions import selected_policies, excluded_names
    for operation in (lambda: FrameIndex(app, {}, policies(), tag_policies=incomplete),
                      lambda: selected_policies(incomplete, {'wd'}),
                      lambda: excluded_names(incomplete)):
        with pytest.raises(ValueError, match='every model'):
            operation()


def test_frame_grouping_does_not_silently_ignore_scene_records():
    from stash_ai.evidence.frames import make_groups
    with pytest.raises(ValueError, match='frame evidence'):
        make_groups({'wd': records()['scene']})


def test_marker_builder_rejects_policies_different_from_its_frame_index():
    from stash_ai.evidence.source_markers import marker_contributions
    value = {'wd': records()['markers']}
    bans = policies()
    with FrameIndex(app, value, bans, tag_policies=bans) as index:
        with pytest.raises(ValueError, match='same exclusion policies'):
            marker_contributions(app, value, bans, index=index, tag_policies=policies({'core'}))
