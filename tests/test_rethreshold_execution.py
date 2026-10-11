"""Threshold execution preserves all-target preflight and pending/no-op behavior."""
from dataclasses import replace

import pytest

from tests.helpers.runtime import runtime as app
from stash_ai.rethreshold.execution import run
from stash_ai.runtime.report import RunReport
from stash_ai.storage.store import StateStore
from tests.test_result_thresholds import blobs, config, outputs, threshold_rows

pytestmark = pytest.mark.usefixtures('reset_temporal')
MODES = [(True, False), (False, True), (True, True)]


@pytest.mark.parametrize('failure', ['limit', 'disk', 'interrupt'])
def test_spool_failure_never_applies_thresholds(tmp_path, monkeypatch, failure):
    import sqlite3
    from stash_ai.rethreshold.spool import SpoolUsage, SpoolLimitExceeded
    with StateStore(tmp_path / 'state.sqlite3', 'scope') as store:
        store.put('1', 'wd', outputs())
        before = blobs(store)
        report = RunReport(write=lambda _: None)
        if failure != 'limit':
            def stop(*args):
                if failure == 'interrupt': raise KeyboardInterrupt()
                raise sqlite3.OperationalError('database or disk is full')
            monkeypatch.setattr(SpoolUsage, 'flushed', stop)
        with pytest.raises((SpoolLimitExceeded, sqlite3.OperationalError, KeyboardInterrupt)):
            run(app, store, ['1'], [config(.8)], report, spool_limit_bytes=1 if failure == 'limit' else 0)
        assert not threshold_rows(store) and blobs(store) == before
        assert report.threshold_preflight['applied'] == 0


@pytest.mark.parametrize('failed_scene,committed', [('2', 0), ('130', 128)])
def test_failed_batch_only_reports_previously_committed_pairs(tmp_path, monkeypatch, failed_scene, committed):
    from stash_ai.rethreshold import execution
    with StateStore(tmp_path / 'state.sqlite3', 'scope') as store:
        count = 2 if failed_scene == '2' else 130
        ids = [str(i) for i in range(1, count+1)]
        for scene in ids: store.put(scene, 'wd', outputs())
        report, original = RunReport(write=lambda _: None), execution.write_plan
        def fail(store, scene, *args):
            original(store, scene, *args)
            if scene == failed_scene: raise KeyboardInterrupt()
        monkeypatch.setattr(execution, 'write_plan', fail)
        with pytest.raises(KeyboardInterrupt):
            run(app, store, ids, [config(.8)], report)
        assert len({row[1] for row in threshold_rows(store)}) == committed
        assert report.threshold_preflight['applied'] == committed
        assert report.workload['rethreshold_apply']['committed'] == committed
        assert report.workload['rethreshold_apply']['pending_batch'] == 0
        assert report.counts['threshold_pairs_committed'] == committed


@pytest.mark.parametrize('scenes,markers', MODES)
def test_late_failure_prevents_every_threshold_and_index_update(tmp_path, scenes, markers):
    with StateStore(tmp_path / 'state.sqlite3', 'scope') as store:
        store.put('1', 'wd', outputs())
        invalid = outputs()
        kind = 'scene' if scenes else 'markers'
        invalid[kind]['meta']['scene_score_floor' if scenes else 'marker_score_floor'] = .95
        store.put('2', 'wd', invalid)
        before = blobs(store), store.connection.execute('SELECT * FROM result_index').fetchall()
        report = RunReport(write=lambda _: None)
        target = config(.8) if scenes else replace(config(.8), scene_threshold=.4)

        assert run(app, store, ['1', '2', '3'], [target], report,
                   scenes=scenes, markers=markers) == {'errors': 1}

        assert not threshold_rows(store)
        assert (blobs(store), store.connection.execute('SELECT * FROM result_index').fetchall()) == before
        assert report.threshold_preflight == dict(
            selected=3, checked=2, skipped_no_evidence=1, failed=1, applied=0,
            reasons={'scene cache floor too high' if scenes else 'marker cache floor too high': 1})


@pytest.mark.parametrize('scenes,markers', MODES)
@pytest.mark.parametrize('changed', [False, True])
def test_pending_publication_preserves_mode_specific_noop_rules(tmp_path, scenes, markers, changed):
    with StateStore(tmp_path / 'state.sqlite3', 'scope') as store:
        store.put('1', 'wd', outputs())
        store.prepare('1', {'markers': {'contributions': []}})
        report = RunReport(write=lambda _: None)
        target = config(.8 if changed else .4)
        if not scenes:
            target = replace(target, scene_threshold=.4)
        before = blobs(store)

        result = run(app, store, ['1'], [target], report, scenes=scenes, markers=markers)

        blocked = changed or not scenes
        assert result == ({'errors': 1} if blocked else {'scene_thresholds_updated': 0})
        assert report.threshold_preflight['failed'] == int(blocked)
        assert report.threshold_preflight['applied'] == 0
        if blocked:
            phase = 'scene thresholds' if scenes else 'marker thresholds'
            message = ('Finish the pending merge or use --abandon-pending before changing thresholds.'
                       if scenes else 'Finish the pending merge or use --abandon-pending first.')
            assert report.groups[phase, 'wd', 'PendingPublication']['examples'] == ['scene 1: ' + message]
        assert not threshold_rows(store)
        assert blobs(store) == before

