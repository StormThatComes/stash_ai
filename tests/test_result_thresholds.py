"""Threshold policy writes preserve evidence, preview isolation and completion."""
from copy import deepcopy
import sqlite3
from types import SimpleNamespace

import pytest

from stash_ai import cli as app
from stash_ai import cleanup, db, developer_report, performance, progress, recovery
from stash_ai import scene_cache, thresholds
from stash_ai.runtime import RunReport
from stash_ai.state import StateStore
from stash_ai.taggers import SOURCES

pytestmark = pytest.mark.usefixtures('reset_temporal')


def outputs(threshold=.4, signature='original'):
    meta = dict(origin='inference', inference_signature=signature,
                scene_threshold=threshold, marker_threshold=threshold,
                scene_frames_format=1, scene_score_floor=.1, marker_score_floor=.1,
                marker_tag_blocklist=[])
    frames = [dict(start=0., end=1., tags=[['core', .8]])]
    return {
        'scene': dict(mode='scene', meta=deepcopy(meta), frames=deepcopy(frames),
                      items=[dict(tag='core', score=.8)]),
        'markers': dict(mode='frames', meta=deepcopy(meta), items=frames),
    }


def config(threshold):
    return SimpleNamespace(source=SOURCES['wd'], scene_threshold=threshold, marker_threshold=threshold)


def rethreshold(store, threshold, *, markers=True, scenes=('1',)):
    return scene_cache.rethreshold(app, store, scenes, [config(threshold)],
                                  RunReport(write=lambda _: None), markers=markers)


def blobs(store):
    return store.connection.execute('SELECT * FROM results ORDER BY scope,scene,model,kind').fetchall()


def threshold_rows(store):
    return store.connection.execute(
        'SELECT * FROM result_thresholds ORDER BY scope,scene,model,kind').fetchall()


@pytest.mark.parametrize('cache_bytes', [0, 2**20])
def test_raise_lower_restore_never_encode_or_write_evidence(tmp_path, monkeypatch, cache_bytes):
    path = tmp_path / 'state.sqlite3'
    with StateStore(path, 'scope', cache_bytes=cache_bytes) as store:
        store.put('1', 'wd', outputs())
        before = blobs(store)
        original = {kind: store.records('1', kind) for kind in ('scene', 'markers')}
        assert progress.completed(store, '1', 'wd', ['scene', 'markers'], 'original')
        monkeypatch.setattr(store, 'pack', lambda *a: pytest.fail('encoded inference evidence'))
        statements = []
        store.connection.set_trace_callback(statements.append)
        for value in (.8, .3, .4):
            assert rethreshold(store, value) == {'scene_thresholds_updated': 1}
            assert blobs(store) == before
            assert len(threshold_rows(store)) == 2
            for kind in ('scene', 'markers'):
                segment = store.model_records('1', kind, 'wd')[0]
                key = scene_cache.SCENE_OVERRIDE if kind == 'scene' else thresholds.OVERRIDE
                if value == .4:
                    assert key not in segment
                else:
                    assert segment.pop(key) == value
                assert segment == original[kind]['wd'][0]
            signature = store.connection.execute(
                "SELECT signature FROM result_index WHERE kind='markers'").fetchone()[0]
            assert progress.completed(store, '1', 'wd', ['markers'], signature)
        assert not any(sql.startswith(('INSERT INTO results ', 'UPDATE results ')) for sql in statements)
        assert rethreshold(store, .4) == {'scene_thresholds_updated': 0}
    with StateStore(path, 'scope', dry_run=True) as store:
        assert store.preview_source
        assert {kind: store.records('1', kind) for kind in original} == original
        assert progress.completed(store, '1', 'wd', ['scene', 'markers'], 'original')
    assert db.check_database(path, 'scope')['ok']


def test_marker_only_updates_policy_without_writing_blob(tmp_path, monkeypatch):
    with StateStore(tmp_path / 'state.sqlite3', 'scope') as store:
        store.put('1', 'wd', {'markers': outputs()['markers']})
        before = blobs(store)
        monkeypatch.setattr(store, 'pack', lambda *a: pytest.fail('encoded inference evidence'))
        target = config(.8)
        target.scene_threshold = .4
        result = thresholds.rethreshold(app, store, ['1'], [target], RunReport(write=lambda _: None))
        assert result == {'marker_thresholds_updated': 1}
        assert blobs(store) == before
        assert store.records('1', 'markers')['wd'][0][thresholds.OVERRIDE] == .8


@pytest.mark.parametrize('dry_run', [False, True])
def test_v7_inline_overrides_are_read_lazily_and_can_be_restored(tmp_path, dry_run):
    path = tmp_path / 'old.sqlite3'
    old = outputs()
    old['scene'][scene_cache.SCENE_OVERRIDE] = .9
    old['markers'][thresholds.OVERRIDE] = .9
    with StateStore(path, 'scope') as store:
        store.put('1', 'wd', old)
        before = blobs(store)
    with sqlite3.connect(path) as connection:
        connection.execute('DROP TABLE result_thresholds')
        connection.execute('PRAGMA user_version=7')
    source_before = path.read_bytes()
    with StateStore(path, 'scope', dry_run=dry_run) as store:
        assert store.connection.execute('PRAGMA user_version').fetchone() == (7,)
        assert store.records('1', 'scene')['wd'] == [old['scene']]
        assert store.model_records('1', 'markers', 'wd') == [old['markers']]
        assert rethreshold(store, .4) == {'scene_thresholds_updated': 1}
        assert store.connection.execute('PRAGMA user_version').fetchone() == (8,)
        assert blobs(store) == ([] if dry_run else before)
        assert store.records('1', 'scene')['wd'] == [outputs()['scene']]
        assert store.model_records('1', 'markers', 'wd') == [outputs()['markers']]
    if dry_run:
        assert path.read_bytes() == source_before
    else:
        assert db.check_database(path, 'scope')['ok']


@pytest.mark.parametrize('dry_run', [False, True])
@pytest.mark.parametrize('replace', [False, True])
def test_new_inference_resets_override_and_append_keeps_effective_history(tmp_path, dry_run, replace):
    path = tmp_path / 'state.sqlite3'
    with StateStore(path, 'scope') as store:
        store.put('1', 'wd', outputs())
        rethreshold(store, .8)
    source_before = path.read_bytes()
    with StateStore(path, 'scope', dry_run=dry_run) as store:
        if dry_run:
            assert not threshold_rows(store)  # the source stays in the read-only attachment
        new = outputs(.6, 'new')
        store.put('1', 'wd', new, replace=dict(scene=replace, markers=replace))
        assert not threshold_rows(store)
        assert not store.connection.execute('SELECT * FROM ' + store.table('result_thresholds')).fetchall()
        for kind, segment in new.items():
            effective = store.records('1', kind)['wd']
            assert effective[-1] == segment
            assert len(effective) == (1 if replace else 2)
            if not replace:
                key = scene_cache.SCENE_OVERRIDE if kind == 'scene' else thresholds.OVERRIDE
                assert effective[0][key] == .8
        assert progress.completed(store, '1', 'wd', ['scene', 'markers'], 'new')
    if dry_run:
        assert path.read_bytes() == source_before
    else:
        assert db.check_database(path, 'scope')['ok']


def test_override_applies_to_every_historical_segment_without_mutating_borrowed_records(tmp_path):
    with StateStore(tmp_path / 'state.sqlite3', 'scope', cache_bytes=2**20) as store:
        store.put('1', 'wd', outputs())
        store.put('1', 'wd', outputs(.6, 'second'), replace=dict(scene=False, markers=False))
        borrowed = store.records('1', 'scene', borrow=True)['wd']
        before = deepcopy(borrowed)
        raw = blobs(store)
        rethreshold(store, .6)
        assert borrowed == before
        effective = store.records('1', 'scene')['wd']
        assert effective[0][scene_cache.SCENE_OVERRIDE] == .6
        assert scene_cache.SCENE_OVERRIDE not in effective[1]
        assert blobs(store) == raw
        rethreshold(store, .4)
        effective = store.records('1', 'scene')['wd']
        assert scene_cache.SCENE_OVERRIDE not in effective[0]
        assert effective[1][scene_cache.SCENE_OVERRIDE] == .4


def test_override_invalidates_only_affected_scene_and_scope(tmp_path):
    path = tmp_path / 'state.sqlite3'
    for scope in ('scope', 'other'):
        with StateStore(path, scope) as store:
            for scene in ('1', '2'):
                store.put(scene, 'wd', outputs())
                with store.connection:
                    performance.record_check(store, scene, 'policy', 'fingerprint')
    with StateStore(path, 'scope') as store:
        rethreshold(store, .8)
        assert not performance.checked(store, '1', 'policy', 'fingerprint')
        assert performance.checked(store, '2', 'policy', 'fingerprint')
    with StateStore(path, 'other', dry_run=True) as store:
        assert performance.checked(store, '1', 'policy', 'fingerprint')
        assert store.records('1', 'scene')['wd'] == [outputs()['scene']]


@pytest.mark.parametrize('initial_threshold', [None, .7])
@pytest.mark.parametrize('metadata_only', [False, True])
def test_both_thresholds_and_version_roll_back_if_index_write_fails(tmp_path, monkeypatch, initial_threshold, metadata_only):
    path = tmp_path / 'state.sqlite3'
    with StateStore(path, 'scope', cache_bytes=2**20) as store:
        store.put('1', 'wd', outputs())
        if initial_threshold is not None:
            rethreshold(store, initial_threshold)
        raw = blobs(store)
        old_thresholds = threshold_rows(store)
        old_records = store.records('1', 'scene')
        version = store.connection.execute('PRAGMA user_version').fetchone()
        index = store.connection.execute('SELECT * FROM result_index').fetchall()
        planned = scene_cache.plan(app, store, '1', config(.8), markers=True, metadata_only=metadata_only)
        original = progress.index_result

        def fail_second(store, scene, model, kind, segments, **options):
            if kind == 'markers':
                raise RuntimeError('index write failed')
            original(store, scene, model, kind, segments, **options)

        monkeypatch.setattr(progress, 'index_result', fail_second)
        with pytest.raises(RuntimeError, match='index write failed'):
            scene_cache.apply(store, '1', 'wd', planned, metadata_only=metadata_only)
        assert blobs(store) == raw
        assert threshold_rows(store) == old_thresholds
        assert store.connection.execute('PRAGMA user_version').fetchone() == version
        assert store.connection.execute('SELECT * FROM result_index').fetchall() == index
        assert store.records('1', 'scene') == old_records


@pytest.mark.parametrize('blocked', ['missing_frames', 'pending'])
def test_preflight_failure_applies_no_table_changes(tmp_path, blocked):
    with StateStore(tmp_path / 'state.sqlite3', 'scope') as store:
        for scene in ('1', '2'):
            store.put(scene, 'wd', outputs())
        if blocked == 'pending':
            store.prepare('2', {'markers': {'contributions': []}})
        else:
            bad = outputs()
            bad['scene'].pop('frames')
            store.put('2', 'wd', bad)
        raw = blobs(store)
        version = store.connection.execute('PRAGMA user_version').fetchone()
        assert rethreshold(store, .8, scenes=('1', '2')) == {'errors': 1}
        assert not threshold_rows(store)
        assert blobs(store) == raw
        assert store.connection.execute('PRAGMA user_version').fetchone() == version


def test_sparse_v8_preview_can_change_and_restore_threshold_without_copying_results(tmp_path, monkeypatch):
    path = tmp_path / 'state.sqlite3'
    with StateStore(path, 'scope') as store:
        store.put('1', 'wd', outputs())
        rethreshold(store, .8)
    before = path.read_bytes(), path.stat().st_mtime_ns
    with StateStore(path, 'scope', dry_run=True) as store:
        assert store.preview_source
        assert store.records('1', 'scene')['wd'][0][scene_cache.SCENE_OVERRIDE] == .8
        monkeypatch.setattr(store, 'pack', lambda *a: pytest.fail('encoded inference evidence'))
        rethreshold(store, .9)
        rethreshold(store, .4)
        assert not blobs(store)
        assert len(threshold_rows(store)) == 2
        assert store.records('1', 'scene')['wd'] == [outputs()['scene']]
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_abandon_and_cleanup_preserve_v8_and_effective_thresholds(tmp_path):
    path = tmp_path / 'state.sqlite3'
    with StateStore(path, 'scope') as store:
        store.put('1', 'wd', outputs())
        rethreshold(store, .8)
        store.prepare('1', {'scene': {'owned_ids': [], 'contributions': []}})
        recovery.abandon(store, lambda _: None)
        cleanup.begin(store, 'cleanup', {'excluded'}, set(), [], kinds=('scene',))
        cleanup.finish(store)
        assert store.connection.execute('PRAGMA user_version').fetchone() == (8,)
        assert store.records('1', 'scene')['wd'][0][scene_cache.SCENE_OVERRIDE] == .8
    assert db.check_database(path, 'scope')['ok']


def test_diagnostics_and_compaction_preserve_threshold_policy(tmp_path):
    path = tmp_path / 'state.sqlite3'
    with StateStore(path, 'scope') as store:
        store.put('1', 'wd', outputs())
        rethreshold(store, .8)
    report = developer_report.build_report(path)
    assert report['database']['table_rows']['result_thresholds'] == 2
    assert 'threshold' not in report['database']['blob_tables'].get('result_thresholds', {})
    before, after = db.compact_database(path, 'scope')
    assert before['result_thresholds'] == after['result_thresholds'] == 2
    assert db.check_database(path, 'scope')['ok']
    with StateStore(path, 'scope', dry_run=True) as store:
        assert store.records('1', 'scene')['wd'][0][scene_cache.SCENE_OVERRIDE] == .8


def test_deleting_result_removes_its_threshold(tmp_path):
    with StateStore(tmp_path / 'state.sqlite3', 'scope') as store:
        store.put('1', 'wd', outputs())
        rethreshold(store, .8)
        with store.connection:
            store.connection.execute("DELETE FROM results WHERE kind='markers'")
        assert [row[3] for row in threshold_rows(store)] == ['scene']


def test_db_check_reports_orphan_invalid_threshold_and_signature_mismatch(tmp_path):
    path = tmp_path / 'state.sqlite3'
    with StateStore(path, 'scope') as store:
        store.put('1', 'wd', outputs())
        rethreshold(store, .8)
        with store.connection:
            store.connection.execute('PRAGMA ignore_check_constraints=ON')
            store.connection.execute("UPDATE result_thresholds SET threshold=1.5 WHERE kind='markers'")
            store.connection.execute("INSERT INTO result_thresholds VALUES ('scope','orphan','wd','scene',.8)")
            store.connection.execute("UPDATE result_index SET signature='wrong' WHERE kind='scene'")
    report = db.check_database(path, 'scope')
    errors = {issue['code']: issue['count'] for issue in report['errors']}
    assert errors['result_threshold_invalid'] == 1
    assert errors['result_threshold_orphan'] == 1
    assert errors['result_index_mismatch'] >= 1


def test_v8_missing_threshold_table_is_a_db_check_error(tmp_path):
    path = tmp_path / 'state.sqlite3'
    with StateStore(path, 'scope') as store:
        store.put('1', 'wd', outputs())
        rethreshold(store, .8)
    with sqlite3.connect(path) as connection:
        connection.execute('DROP TABLE result_thresholds')
    report = db.check_database(path, 'scope')
    assert not report['ok']
    assert any(issue['code'] == 'missing_base_tables' for issue in report['errors'])
