"""SQL preflight preserves retention guards, history and legacy preview isolation."""
import json
import sqlite3

import pytest

from stash_ai.rethreshold import execution as rethreshold_execution, scenes as scene_updates
from stash_ai import cli as app, db
from stash_ai.publication import fingerprints as performance
import stash_ai.storage.merge_checks as performance_merge_checks
from stash_ai.reports.developer import report as developer_report
import stash_ai.evidence.identity as scene_cache_identity
import stash_ai.evidence.identity as thresholds_identity
from stash_ai.runtime.report import RunReport
from stash_ai.storage.store import StateStore
from tests.test_result_thresholds import blobs, config, outputs, rethreshold, threshold_rows

pytestmark = pytest.mark.usefixtures('reset_temporal')


def metadata_rows(store):
    return store.connection.execute('SELECT * FROM ' + store.table('result_metadata') +
                                    ' ORDER BY scope,scene,model,kind').fetchall()


def test_combined_check_and_apply_never_read_payloads(tmp_path, monkeypatch):
    with StateStore(tmp_path/'state.sqlite3', 'scope') as store:
        for scene in ('1', '2'):
            store.put(scene, 'wd', outputs())
        report = RunReport(write=lambda _: None)
        store.report = report
        before = blobs(store), metadata_rows(store)
        plan, unpack = scene_updates.plan, store.unpack
        checking, checks = False, 0

        def guarded_unpack(value):
            assert not checking, 'Check decoded frame evidence'
            return unpack(value)

        def guarded_plan(*args, **kwargs):
            nonlocal checking, checks
            checking = kwargs.get('metadata_only', False)
            checks += int(checking)
            try:
                return plan(*args, **kwargs)
            finally:
                checking = False

        monkeypatch.setattr(store, 'unpack', guarded_unpack)
        monkeypatch.setattr(scene_updates, 'plan', guarded_plan)
        def authorize(action, table, column, database, trigger):
            if checking and action == sqlite3.SQLITE_READ and table == 'results' and column == 'data':
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        store.connection.set_authorizer(authorize)
        result = rethreshold_execution.run(app, store, ['1', '2'], [config(.8)], report, markers=True)
        assert result == {'scene_thresholds_updated': 2}
        assert checks == 4  # Check and Apply both use the small metadata view.
        assert report.timing_calls['state_json_decode'] == 0
        assert (blobs(store), metadata_rows(store)) == before


def test_marker_check_uses_metadata_for_scene_threshold_guard(tmp_path, monkeypatch):
    with StateStore(tmp_path/'state.sqlite3', 'scope') as store:
        store.put('1', 'wd', outputs())
        report = RunReport(write=lambda _: None)
        store.report = report
        monkeypatch.setattr(store, 'unpack', lambda *a: pytest.fail('Check decoded evidence'))
        result = rethreshold_execution.run(app, store, ['1'], [config(.8)], report, scenes=False, markers=True)
        assert result == {'errors': 1}
        assert not threshold_rows(store)
        assert report.timing_calls['state_json_decode'] == 0


@pytest.mark.parametrize('problem', [
    'scene_floor', 'marker_floor', 'frames', 'format', 'origin',
    'blocklist_missing', 'blocklist_invalid', 'blocklist_unblocked', 'imported',
])
def test_metadata_check_matches_payload_validation(tmp_path, problem):
    saved = outputs()
    if problem == 'scene_floor':
        saved['scene']['meta']['scene_score_floor'] = .9
    elif problem == 'marker_floor':
        saved['markers']['meta']['marker_score_floor'] = .9
    elif problem == 'frames':
        saved['scene'].pop('frames')
    elif problem == 'format':
        saved['scene']['meta']['scene_frames_format'] = 2
    elif problem == 'origin':
        saved['scene']['meta']['origin'] = 'legacy'
    elif problem == 'blocklist_missing':
        saved['markers']['meta'].pop('marker_tag_blocklist')
    elif problem == 'blocklist_invalid':
        saved['markers']['meta']['marker_tag_blocklist'] = None
    elif problem == 'blocklist_unblocked':
        saved['markers']['meta']['marker_tag_blocklist'] = ['core']
    elif problem == 'imported':
        saved['markers']['mode'] = 'spans'
    with StateStore(tmp_path/'state.sqlite3', 'scope') as store:
        store.put('1', 'wd', saved)
        messages = []
        for metadata_only in (False, True):
            with pytest.raises(ValueError) as error:
                scene_updates.plan(app, store, '1', config(.8), markers=True,
                                 direct=True, metadata_only=metadata_only)
            messages.append(str(error.value))
        assert messages[0] == messages[1]


def test_late_historical_failure_applies_no_thresholds(tmp_path, monkeypatch):
    with StateStore(tmp_path/'state.sqlite3', 'scope') as store:
        store.put('1', 'wd', outputs())
        old = outputs()
        old['markers']['meta']['marker_score_floor'] = .95
        store.put('2', 'wd', old)
        store.put('2', 'wd', outputs(.6, 'new'), replace=dict(scene=False, markers=False))
        raw = blobs(store)
        monkeypatch.setattr(store, 'unpack', lambda *a: pytest.fail('Check decoded evidence'))
        assert rethreshold(store, .8, scenes=('1', '2')) == {'errors': 1}
        assert not threshold_rows(store)
        assert blobs(store) == raw


@pytest.mark.parametrize('same_threshold', [False, True])
def test_combined_pending_guard_keeps_noop_semantics(tmp_path, monkeypatch, same_threshold):
    with StateStore(tmp_path/'state.sqlite3', 'scope') as store:
        store.put('1', 'wd', outputs())
        store.prepare('1', {'markers': {'contributions': []}})
        target = config(.4 if same_threshold else .8)
        planned = scene_updates.plan(app, store, '1', target, markers=True,
                                   direct=True, metadata_only=True, pending_checked=True)
        assert bool(planned) == (not same_threshold)
        # A pending combined no-op remains valid; an actual edit fails before Apply.
        if not same_threshold:
            monkeypatch.setattr(store, 'unpack', lambda *a: pytest.fail('Check decoded evidence'))
        assert rethreshold(store, target.scene_threshold) == (
            {'scene_thresholds_updated': 0} if same_threshold else {'errors': 1})
        assert not threshold_rows(store)


@pytest.mark.parametrize('dry_run', [False, True])
@pytest.mark.parametrize('version', [5, 7, 8])
def test_old_database_backfills_once_and_keeps_inline_overrides(tmp_path, monkeypatch, dry_run, version):
    path = tmp_path/'legacy.sqlite3'
    saved = outputs()
    saved['scene'][scene_cache_identity.SCENE_OVERRIDE] = .9
    saved['markers'][thresholds_identity.OVERRIDE] = .9
    with StateStore(path, 'scope') as store:
        store.put('1', 'wd', saved)
        raw = blobs(store)
    with sqlite3.connect(path) as connection:
        for event in ('insert', 'update', 'delete'):
            connection.execute('DROP TRIGGER results_metadata_' + event)
        connection.execute('DROP TABLE result_metadata')
        connection.execute(f'PRAGMA user_version={version}')
    source = path.read_bytes()
    with StateStore(path, 'scope', dry_run=dry_run) as store:
        report = RunReport(write=lambda _: None)
        store.report = report
        statements = []
        store.connection.set_trace_callback(statements.append)
        assert scene_updates.plan(app, store, '1', config(.4), markers=True,
                                direct=True, metadata_only=True)
        assert report.timing_calls['state_json_decode'] == 2
        assert store.connection.execute('PRAGMA user_version').fetchone() == (version,)
        first = metadata_rows(store)
        assert len(first) == 2
        monkeypatch.setattr(store, 'unpack', lambda *a: pytest.fail('Repeated legacy decode'))
        assert scene_updates.plan(app, store, '1', config(.4), markers=True,
                                direct=True, metadata_only=True)
        assert metadata_rows(store) == first
        assert not any(sql.startswith(('INSERT INTO results ', 'UPDATE results ')) for sql in statements)
    if dry_run:
        assert path.read_bytes() == source
    else:
        with StateStore(path, 'scope') as store:
            assert blobs(store) == raw
        assert db.check_database(path, 'scope')['ok']


@pytest.mark.parametrize('dry_run', [False, True])
@pytest.mark.parametrize('replace', [False, True])
def test_preview_new_inference_metadata_hides_source_and_tracks_history(tmp_path, dry_run, replace):
    path = tmp_path/'state.sqlite3'
    with StateStore(path, 'scope') as store:
        store.put('1', 'wd', outputs())
        rethreshold(store, .8)
    source = path.read_bytes()
    with StateStore(path, 'scope', dry_run=dry_run) as store:
        new = outputs(.6, 'new')
        store.put('1', 'wd', new, replace=dict(scene=replace, markers=replace))
        for kind in ('scene', 'markers'):
            metadata = store.result_metadata('1', kind, 'wd')
            assert len(metadata) == (1 if replace else 2)
            assert metadata[-1]['meta']['inference_signature'] == 'new'
            assert metadata[-1]['meta']['scene_threshold'] == .6
            if not replace:
                key = scene_cache_identity.SCENE_OVERRIDE if kind == 'scene' else thresholds_identity.OVERRIDE
                assert metadata[0][key] == .8
    if dry_run:
        assert path.read_bytes() == source
    else:
        assert db.check_database(path, 'scope')['ok']


def test_legacy_backfill_only_selected_pairs_and_commits_once_per_check(tmp_path):
    with StateStore(tmp_path/'state.sqlite3', 'scope') as store:
        for scene in ('1', '2', '3'):
            for model in ('wd', 'dtq'):
                store.put(scene, model, outputs())
        with store.connection:
            store.connection.execute('DELETE FROM result_metadata')
        report = RunReport(write=lambda _: None)
        store.report = report
        statements = []
        store.connection.set_trace_callback(statements.append)
        assert rethreshold_execution.run(app, store, ['1', '2'], [config(.8)], report,
                                        markers=True) == {'scene_thresholds_updated': 2}
        assert {(row[1], row[2], row[3]) for row in metadata_rows(store)} == {
            (scene, 'wd', kind) for scene in ('1', '2') for kind in ('scene', 'markers')}
        assert statements.count('COMMIT') == 3  # One Check backfill plus two Apply pairs.
        assert report.timing_calls['state_json_decode'] == 4  # Check backfills each scene/kind once.


@pytest.mark.parametrize('operation', ['update', 'replace', 'delete', 'move'])
def test_raw_sql_writes_invalidate_metadata_even_without_new_writer(tmp_path, operation):
    with StateStore(tmp_path/'state.sqlite3', 'scope') as store:
        store.put('1', 'wd', outputs())
        with store.connection:
            if operation == 'update':
                new = outputs()['markers']
                new['meta']['marker_score_floor'] = .95
                store.connection.execute("UPDATE results SET data=? WHERE kind='markers'", (store.pack([new]),))
            elif operation == 'replace':
                store.connection.execute("INSERT OR REPLACE INTO results SELECT * FROM results WHERE kind='markers'")
            elif operation == 'delete':
                store.connection.execute("DELETE FROM results WHERE kind='markers'")
            else:
                store.connection.execute("UPDATE results SET scene='2' WHERE kind='markers'")
        assert [row[3] for row in metadata_rows(store)] == ['scene']
        if operation == 'update':
            assert rethreshold(store, .8) == {'errors': 1}
        elif operation == 'delete':
            assert store.result_metadata('1', 'markers', 'wd') == []
        elif operation == 'move':
            assert store.result_metadata('1', 'markers', 'wd') == []
            assert store.result_metadata('2', 'markers', 'wd')
        else:
            assert store.result_metadata('1', 'markers', 'wd')


@pytest.mark.parametrize('damage', ['json', 'shape', 'format'])
def test_damaged_derived_row_rebuilds_from_payload(tmp_path, damage):
    with StateStore(tmp_path/'state.sqlite3', 'scope') as store:
        store.put('1', 'wd', outputs())
        expected = store.result_metadata('1', 'markers', 'wd')
        raw = blobs(store)
        with store.connection:
            if damage == 'format':
                store.connection.execute("UPDATE result_metadata SET format=99 WHERE kind='markers'")
            else:
                store.connection.execute("UPDATE result_metadata SET data=? WHERE kind='markers'",
                                         ('{' if damage == 'json' else '{}',))
        assert store.result_metadata('1', 'markers', 'wd') == expected
        assert blobs(store) == raw


def test_metadata_and_results_roll_back_together(tmp_path, monkeypatch):
    with StateStore(tmp_path/'state.sqlite3', 'scope') as store:
        store.put('1', 'wd', outputs())
        before = blobs(store), metadata_rows(store)
        save = store.results._save_metadata

        def fail_second(scene, model, kind, segments):
            if kind == 'markers':
                raise RuntimeError('metadata write failed')
            return save(scene, model, kind, segments)

        monkeypatch.setattr(store.results, '_save_metadata', fail_second)
        with pytest.raises(RuntimeError, match='metadata write failed'):
            store.put('1', 'wd', outputs(.6, 'new'))
        assert (blobs(store), metadata_rows(store)) == before


def test_metadata_is_small_scoped_and_does_not_invalidate_merge(tmp_path):
    path = tmp_path/'state.sqlite3'
    for scope in ('scope', 'other'):
        with StateStore(path, scope) as store:
            saved = outputs(signature=scope)
            saved['markers']['items'] *= 1000
            store.put('1', 'wd', saved)
            with store.connection:
                performance_merge_checks.record_check(store, '1', 'policy', 'fingerprint')
                store.connection.execute('DELETE FROM result_metadata WHERE scope=?', (scope,))
            value = store.result_metadata('1', 'markers', 'wd')[0]
            assert value['meta']['inference_signature'] == scope
            assert value['populated'] is True
            assert 'items' not in value and 'frames' not in value
            assert len(metadata_rows(store)[-1][-1]) < 1000
            assert performance_merge_checks.checked(store, '1', 'policy', 'fingerprint')
    with StateStore(path, 'scope') as store:
        assert all(row[0] in ('scope', 'other') for row in metadata_rows(store))


def test_db_check_reports_corrupt_mismatched_orphan_metadata_without_repair(tmp_path):
    path = tmp_path/'state.sqlite3'
    with StateStore(path, 'scope') as store:
        store.put('1', 'wd', outputs())
        with store.connection:
            metadata = store.result_metadata('1', 'markers', 'wd')
            metadata[0]['meta']['marker_score_floor'] = .9
            store.connection.execute("UPDATE result_metadata SET data=? WHERE kind='markers'",
                                     (json.dumps(metadata),))
            store.connection.execute("UPDATE result_metadata SET data='{' WHERE kind='scene'")
            store.connection.execute("INSERT INTO result_metadata VALUES ('scope','orphan','wd','markers',1,'[]')")
    before = path.read_bytes()
    report = db.check_database(path, 'scope')
    errors = {item['code']:item['count'] for item in report['errors']}
    assert errors['result_metadata_invalid'] == 1
    assert errors['result_metadata_mismatch'] == 1
    assert errors['result_metadata_orphan'] == 1
    assert path.read_bytes() == before


def test_compaction_and_anonymous_report_preserve_only_metadata_counts(tmp_path):
    path = tmp_path/'state.sqlite3'
    with StateStore(path, 'scope') as store:
        store.put('1', 'wd', outputs())
        original = metadata_rows(store)
    before, after = db.compact_database(path, 'scope')
    assert before['result_metadata'] == after['result_metadata'] == 2
    with StateStore(path, 'scope') as store:
        assert metadata_rows(store) == original
    report = developer_report.build_report(path)
    assert report['database']['table_rows']['result_metadata'] == 2
    assert 'result_metadata' not in report['database']['blob_tables']
    assert 'marker_tag_blocklist' not in json.dumps(report)
