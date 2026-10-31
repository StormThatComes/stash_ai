"""Admission must be visible before costly preparation; workers really overlap."""
from contextlib import closing
from pathlib import Path
import os
import time

import pytest

from benchmarks.benchmark_merge_solver import fixture
from stash_ai.recognition.fragmentation_report import _open_readonly
from stash_ai.recognition.merge_solver import solve


def simultaneous_work(task):
    # An actual spawned trial cannot finish before its peer has started. Merely
    # seeing two PIDs across a whole run would also pass for sequential work.
    root = Path(task.config.database).parent
    if task.name in ('trial-1', 'trial-2'):
        (root / (task.name + '.started')).write_text(str(os.getpid()))
        peer = 'trial-2' if task.name == 'trial-1' else 'trial-1'
        deadline = time.monotonic() + 20
        while not (root / (peer + '.started')).exists():
            if time.monotonic() >= deadline:
                raise AssertionError('Second solver trial did not start concurrently.')
            time.sleep(.01)
    from stash_ai.recognition.solver_pool import work
    return work(task)


@pytest.mark.parametrize('scheduling', ['batch', 'async'])
def test_two_production_trials_start_before_either_can_finish(tmp_path, monkeypatch, scheduling):
    from stash_ai.recognition import solver_pool
    from stash_ai.recognition import solver_execution
    config, goals = fixture(tmp_path, 'disjoint')
    monkeypatch.setattr(solver_execution.os, 'process_cpu_count', lambda: 4)
    monkeypatch.setattr(solver_pool, 'work', simultaneous_work)
    with closing(_open_readonly(config.database)) as source:
        result = solve(config, goals, source, maximum=3, targeted_trials=0,
            feasibility_trials=0, execution=dict(cpus=4, trials=2,
                memory_bytes=128 * 1024**3, scheduling=scheduling))
    assert result['simulation']['execution']['trials'] == 2
    trials = result['simulation']['trials']
    assert set(trials) == {'baseline', 'trial-1', 'trial-2'}
    assert len({trials[name]['worker']['worker_pid'] for name in ('trial-1', 'trial-2')}) == 2


def test_plan_is_emitted_before_checkpoint_and_measurements(tmp_path, monkeypatch):
    from stash_ai.recognition import merge_solver
    config, goals = fixture(tmp_path, 'face')
    events = []
    def stop_at_measurements(*args, **kwargs):
        plans = [item['text'] for name, stage, item in events if (name, stage) == ('execution', 'plan')]
        assert len(plans) == 1, 'The admission plan is hidden until after preparation.'
        assert 'requested_trials=4' in plans[0]
        raise KeyboardInterrupt
    monkeypatch.setattr(merge_solver, 'measured_options', stop_at_measurements)
    with closing(_open_readonly(config.database)) as source:
        result = solve(config, goals, source, maximum=3,
            execution=dict(cpus=4, trials=4, memory_bytes=128 * 1024**3),
            progress=lambda *event: events.append(event))
    assert result['status'] == 'no_completed_trials'
    stages = [(name, stage) for name, stage, _ in events]
    assert stages.index(('execution', 'plan')) < stages.index(('initialization', 'stage: Validate solver checkpoint'))


def test_admission_explains_requested_cpu_and_memory_limits(tmp_path, monkeypatch):
    from stash_ai.recognition import solver_execution as execution
    config, _ = fixture(tmp_path, 'face')
    monkeypatch.setattr(execution.os, 'process_cpu_count', lambda: 16)
    monkeypatch.setattr(execution, 'available_memory', lambda: 24 * 1024**3)
    limited = execution.plan(config, dict(cpus=16, trials=4))
    assert limited.trials == 1  # Two configured 8 GiB replay ceilings per trial.
    assert limited.requested_trials == 4
    text = execution.plan_text(limited, maximum=500)
    assert 'trials=1 requested_trials=4' in text
    assert 'limited_by=memory' in text
    assert 'memory_budget_gib=19.20' in text and 'trial_memory_gib=' in text
    monkeypatch.setattr(execution.os, 'process_cpu_count', lambda: 1)
    cpu_limited = execution.plan(config, dict(cpus=16, trials=4, memory_bytes=128 * 1024**3))
    assert 'limited_by=cpu' in execution.plan_text(cpu_limited, maximum=500)
    assert cpu_limited.requested_cpus == 16 and cpu_limited.available_cpus == 1
    default = execution.plan(config, dict(cpus=16, memory_bytes=128 * 1024**3))
    assert default.trials == 1 and default.requested_trials == 1
    assert 'use --solver-trials' in execution.plan_text(default, maximum=500)
    # Diagnostic-only requests must not invalidate an otherwise identical plan.
    assert not {'requested_trials', 'requested_cpus', 'available_cpus'} & default.signature().keys()
