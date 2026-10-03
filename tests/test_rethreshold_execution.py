"""Threshold execution preserves all-target preflight and pending/no-op behavior."""
from dataclasses import replace

import pytest

from stash_ai import cli as app
from stash_ai.rethreshold.execution import run
from stash_ai.runtime import RunReport
from stash_ai.state import StateStore
from tests.test_result_thresholds import blobs, config, outputs, threshold_rows

pytestmark = pytest.mark.usefixtures('reset_temporal')
MODES = [(True, False), (False, True), (True, True)]


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
            assert report.groups[phase, 'wd', 'RuntimeError']['examples'] == ['scene 1: ' + message]
        assert not threshold_rows(store)
        assert blobs(store) == before
