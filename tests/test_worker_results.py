"""Worker failures keep a stable wire contract and count no completed scenes."""
import pickle
from types import SimpleNamespace

import pytest

from stash_ai.runtime.profile import execute_worker, ProfileMetrics
from stash_ai.runtime.worker_result import WorkerResult
from stash_ai.publication.parallel import iter_parallel_evidence
from stash_ai.reports.model.workers import build_report_parallel
from stash_ai.storage.store import StateStore
from tests.helpers.runtime import runtime as app
from tests.helpers.evidence import empty_policies
from tests.test_merge_parallel import _outputs


@pytest.mark.parametrize('profile', [False, True])
@pytest.mark.parametrize('failure', [None, ValueError, KeyboardInterrupt])
def test_worker_envelope_restores_report_and_counts_failure(profile, failure):
    previous = object()
    store = SimpleNamespace(report=previous)
    def callback():
        if failure:
            raise failure('invalid frame')
        return {'frames': [1, 2]}
    if failure is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt):
            execute_worker('1', store, callback, 'compute', profile=profile)
        assert store.report is previous
        return
    result = execute_worker('1', store, callback, 'compute', profile=profile)
    assert isinstance(result, WorkerResult)
    assert pickle.loads(pickle.dumps(result)) == result
    assert store.report is previous and result.scene_id == '1'
    assert result.error == ('ValueError: invalid frame' if failure else None)
    assert result.payload == (None if failure else {'frames': [1, 2]})
    if not profile:
        assert result.metrics is None
        return
    assert {row['kind'] for row in result.metrics} == {'stage', 'worker_cpu'}
    for row in result.metrics:
        assert row['calls'] == 1
        assert row['errors'] == int(failure is not None)
        assert row['units'] == int(failure is None)
        assert row['outcome'] == ('error' if failure else 'ok')


def corrupt_first(path):
    with StateStore(path, 'scope') as store:
        for scene in ('1', '2'):
            store.put(scene, 'wd', _outputs('wd', scene))
        with store.connection:
            store.connection.execute("UPDATE results SET data=? WHERE scene='1'", (b'broken',))


@pytest.mark.parametrize('profile', [False, True])
def test_spawned_merge_returns_failed_scene_in_order_and_continues(tmp_path, profile):
    path = tmp_path / 'state.sqlite3'
    corrupt_first(path)
    before = path.read_bytes()
    policies = empty_policies()
    results = list(iter_parallel_evidence(app, path, 'scope', ['1', '2'], policies, policies,
        workers=2, cache_bytes=0, memory_budget_bytes=128 * 2**20, profile=profile))
    assert [row.scene_id for row in results] == ['1', '2']
    assert results[0].error and results[0].payload is None
    assert results[1].error is None and results[1].payload is not None
    assert all(isinstance(row, WorkerResult) for row in results)
    if profile:
        for result in results:
            compute = [row for row in result.metrics if row['name'] == 'merge_worker_compute']
            assert len(compute) == 2
            assert all(row['units'] == int(result.error is None) for row in compute)
            assert all(row['errors'] == int(result.error is not None) for row in compute)
    assert path.read_bytes() == before


def test_spawned_model_report_ingests_failure_metrics_before_raising(tmp_path):
    path = tmp_path / 'state.sqlite3'
    corrupt_first(path)
    before = path.read_bytes()
    profile, policies = ProfileMetrics(), empty_policies()
    with pytest.raises(RuntimeError):
        build_report_parallel(app, path, 'scope', ['1', '2'], policies, policies,
                              workers=2, cache_bytes=0, memory_budget_bytes=128 * 2**20,
                              profile=profile)
    compute = [row for row in profile.export() if row['name'] == 'model_report_worker_compute']
    assert len(compute) == 2
    assert all(row['errors'] == 1 and row['units'] == 0 and row['outcome'] == 'error' for row in compute)
    assert path.read_bytes() == before
