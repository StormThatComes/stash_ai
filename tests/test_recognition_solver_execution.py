from dataclasses import replace
import pytest

from benchmarks.benchmark_merge_solver import fixture
from stash_ai.recognition.solver_execution import plan, scope, chunks
from stash_ai.recognition.compute import Compute
from stash_ai.recognition.merge_solver import solve
from stash_ai.recognition.fragmentation_report import _open_readonly


def test_joint_cpu_and_memory_admission(tmp_path, monkeypatch):
    monkeypatch.setattr('stash_ai.recognition.solver_execution.available_memory', lambda: 1024**4)
    # Python 3.13+ uses affinity-aware process_cpu_count rather than cpu_count.
    monkeypatch.setattr('stash_ai.recognition.solver_execution.os.process_cpu_count', lambda: 16, raising=False)
    config, _ = fixture(tmp_path, 'face')
    first = plan(config, dict(cpus=16, trials=4))
    assert first.trials == 4 and first.inner_workers == 3
    with pytest.raises(ValueError, match='Cannot start 4 simultaneous solver trials'):
        plan(config, dict(cpus=16, trials=4, memory_bytes=first.trial_bytes*2))
    limited = plan(config, dict(cpus=16, trials=2, memory_bytes=first.trial_bytes*2))
    assert limited.trials == 2 and limited.inner_workers == 7
    with scope(limited):
        pool = Compute(config)
        assert pool.workers == 7 and pool.native_threads == 1
    assert Compute(config).workers == config.data['execution']['rebuild_workers']


@pytest.mark.parametrize('options', [dict(cpus=17), dict(cpus=True), dict(trials=0),
    dict(memory_bytes=1), dict(profile=1), dict(scheduling='bad'), dict(unknown=1)])
def test_invalid_options(tmp_path, options):
    config, _ = fixture(tmp_path, 'face')
    with pytest.raises(ValueError):
        plan(config, options)


def test_cost_chunks_are_bounded():
    assert list(chunks([3, 4, 5, 1], int, target=7, maximum=9)) == [[3, 4], [5, 1]]
    with pytest.raises(ValueError):
        list(chunks([10], int, target=7, maximum=9))


def test_solver_profile_does_not_change_winner(tmp_path):
    config, goals = fixture(tmp_path, 'face')
    with _open_readonly(config.database) as db:
        before = list(db.iterdump())
        serial = solve(config, goals, db, maximum=4)
        measured = solve(config, goals, db, maximum=4, execution=dict(cpus=2, profile=True))
        assert list(db.iterdump()) == before
    assert serial['best_result'] == measured['best_result']
    for trial in measured['simulation']['trials'].values():
        assert trial['performance']['cpu_seconds'] >= 0
        assert trial['performance']['rebuild']['cpu']['native_threads'] == 1
