"""Frame-free Apply preserves completion indexes, history and preview isolation."""
from copy import deepcopy
from dataclasses import replace
import sqlite3

import pytest

from stash_ai.rethreshold import markers as marker_updates, execution as rethreshold_execution, scenes as scene_updates
from stash_ai import db
from tests.helpers.runtime import runtime as app
import stash_ai.evidence.identity as scene_cache_identity
import stash_ai.evidence.identity as thresholds_identity
from stash_ai.storage import progress
from stash_ai.storage import metadata as state_metadata
from stash_ai.runtime.report import RunReport
from stash_ai.storage.store import StateStore
from tests.test_result_thresholds import blobs, config, outputs

pytestmark = pytest.mark.usefixtures('reset_temporal')


def target(mode, value=.8):
    result = config(value)
    if mode == 'markers':
        result = replace(result, scene_threshold=.4)
    return result


def update(store, mode, report):
    if mode == 'markers':
        return rethreshold_execution.run(app, store, ['1'], [target(mode)], report, scenes=False, markers=True)
    return rethreshold_execution.run(app, store, ['1'], [target(mode)], report,
                                  markers=mode == 'combined')


def index_rows(store):
    return store.connection.execute('SELECT * FROM ' + store.table('result_index') +
                                    ' ORDER BY scope,scene,model,kind').fetchall()


@pytest.mark.parametrize('mode', ['combined', 'scene', 'markers'])
@pytest.mark.parametrize('dry_run', [False, True])
@pytest.mark.parametrize('history_populated', [(False, False), (True, False), (False, True), (True, True)])
def test_metadata_apply_matches_full_payload_index_and_preserves_history(tmp_path, monkeypatch, mode, dry_run, history_populated):
    paths = [tmp_path/'full.sqlite3', tmp_path/'metadata.sqlite3']
    for path in paths:
        with StateStore(path, 'scope') as store:
            for populated, signature in zip(history_populated, ('older', 'latest')):
                saved = outputs(signature=signature)
                if not populated:
                    saved['scene']['items'] = saved['scene']['frames'] = []
                    saved['markers']['items'] = []
                store.put('1', 'wd', saved, replace=dict(scene=False, markers=False))
    # Reference the established full-evidence path, including preview overlays.
    with StateStore(paths[0], 'scope', dry_run=dry_run) as store:
        if mode == 'markers':
            assert marker_updates.apply(app, store, '1', target(mode))
        else:
            planned = scene_updates.plan(app, store, '1', target(mode), markers=mode == 'combined')
            scene_updates.apply(store, '1', 'wd', planned)
        expected_index = index_rows(store)
        expected_records = {kind: store.records('1', kind) for kind in ('scene', 'markers')}
    source = paths[1].read_bytes()
    with StateStore(paths[1], 'scope', dry_run=dry_run) as store:
        before = blobs(store)
        report = RunReport(write=lambda _: None)
        store.report = report
        def forbidden(*args, **kwargs):
            pytest.fail('Apply read or encoded frame evidence')
        with monkeypatch.context() as guard:
            guard.setattr(store, 'unpack', forbidden)
            guard.setattr(store, 'pack', forbidden)
            guard.setattr(store, 'records', forbidden)
            guard.setattr(store, 'model_records', forbidden)
            if not dry_run:
                def authorize(action, table, column, database, trigger):
                    if action == sqlite3.SQLITE_READ and table == 'results' and column == 'data':
                        return sqlite3.SQLITE_DENY
                    return sqlite3.SQLITE_OK
                store.connection.set_authorizer(authorize)
            try:
                result = update(store, mode, report)
            finally:
                store.connection.set_authorizer(None)
        assert result == {('marker_thresholds_updated' if mode == 'markers' else 'scene_thresholds_updated'): 1}
        assert report.timing_calls['state_json_decode'] == 0
        assert index_rows(store) == expected_index
        assert blobs(store) == before
        assert {kind: store.records('1', kind) for kind in expected_records} == expected_records
        for row in index_rows(store):
            assert row[4:6] == (1, int(any(history_populated)))
            assert progress.completed(store, '1', 'wd', [row[3]], row[6])
    if dry_run:
        assert paths[1].read_bytes() == source
    assert db.check_database(paths[1], 'scope')['ok']


@pytest.mark.parametrize('history', ['no_inference', 'legacy_tail', 'empty_inference_tail', 'both_overrides'])
def test_small_index_plan_retains_latest_inference_and_all_history_population(tmp_path, history):
    saved = [outputs(signature='first')['scene'], outputs(signature='last')['scene']]
    if history == 'no_inference':
        for segment in saved:
            segment['meta']['origin'] = 'legacy'
    elif history == 'legacy_tail':
        saved[-1]['meta']['origin'] = 'legacy'
    elif history == 'empty_inference_tail':
        saved[-1]['items'] = saved[-1]['frames'] = []
    else:
        saved[-1][scene_cache_identity.SCENE_OVERRIDE] = .8
        saved[-1][thresholds_identity.OVERRIDE] = .9
    with StateStore(tmp_path/'state.sqlite3', 'scope') as store:
        store.replace_records('1', 'wd', 'scene', saved)
        expected = index_rows(store)
        small = state_metadata.summarize(saved)
        before = deepcopy(small)
        assert all('items' not in segment and 'frames' not in segment for segment in small)
        with store.connection:
            progress.index_result(store, '1', 'wd', 'scene', small, metadata_only=True)
        assert index_rows(store) == expected
        assert small == before


@pytest.mark.parametrize('mode', ['combined', 'scene', 'markers'])
@pytest.mark.parametrize('damage', ['missing', 'json', 'shape', 'format'])
def test_legacy_or_damaged_metadata_is_backfilled_once_before_apply(tmp_path, mode, damage):
    with StateStore(tmp_path/'state.sqlite3', 'scope') as store:
        store.put('1', 'wd', outputs())
        raw = blobs(store)
        with store.connection:
            if damage == 'missing':
                store.connection.execute('DELETE FROM result_metadata')
            elif damage == 'format':
                store.connection.execute('UPDATE result_metadata SET format=99')
            else:
                store.connection.execute('UPDATE result_metadata SET data=?', ('{' if damage == 'json' else '{}',))
        report = RunReport(write=lambda _: None)
        store.report = report
        assert 'errors' not in update(store, mode, report)
        assert report.timing_calls['state_json_decode'] == (1 if mode == 'scene' else 2)
        assert blobs(store) == raw

