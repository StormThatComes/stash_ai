"""Admission must be visible before costly preparation; workers really overlap."""
from contextlib import closing
from pathlib import Path
import os
import subprocess
import sys
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
    monkeypatch.setattr('stash_ai.recognition.solver_execution.os.process_cpu_count', lambda: 4)
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
    with pytest.raises(ValueError) as failure:
        execution.plan(config, dict(cpus=16, trials=4))
    text = str(failure.value)
    assert 'Cannot start 4 simultaneous solver trials' in text
    assert 'resources admit only 1' in text
    assert 'RAM:' in text and 'budget=19.20 GiB' in text and 'per_trial=' in text
    assert '80% of 24.00 GiB available' in text and 'required=' in text
    assert 'CPU:' not in text
    monkeypatch.setattr(execution.os, 'process_cpu_count', lambda: 1)
    with pytest.raises(ValueError) as failure:
        execution.plan(config, dict(cpus=16, trials=4, memory_bytes=128 * 1024**3))
    text = str(failure.value)
    assert 'CPU:' in text and 'budget=16' in text and 'available=1' in text
    assert 'RAM:' not in text
    default = execution.plan(config, dict(cpus=16, memory_bytes=128 * 1024**3))
    assert default.trials == 1 and default.requested_trials == 1
    assert 'use --solver-trials' in execution.plan_text(default, maximum=500)
    # Diagnostic-only requests must not invalidate an otherwise identical plan.
    assert not {'requested_trials', 'requested_cpus', 'available_cpus'} & default.signature().keys()


def test_all_insufficient_resources_are_reported(tmp_path, monkeypatch):
    from stash_ai.recognition import solver_execution as execution
    config, _ = fixture(tmp_path, 'face')
    monkeypatch.setattr(execution.os, 'process_cpu_count', lambda: 16)
    with pytest.raises(ValueError) as failure:
        execution.plan(config, dict(cpus=2, trials=4, memory_bytes=20 * 1024**3))
    text = str(failure.value)
    assert 'resources admit only 1' in text
    assert 'CPU:' in text and 'budget=2' in text and 'available=16' in text
    assert 'RAM:' in text and 'budget=20.00 GiB (explicit)' in text


def test_memory_boundary_is_strict_and_unknown_ram_does_not_invent_a_limit(tmp_path, monkeypatch):
    from stash_ai.recognition import solver_execution as execution
    config, _ = fixture(tmp_path, 'face')
    monkeypatch.setattr(execution.os, 'process_cpu_count', lambda: 4)
    monkeypatch.setattr(execution, 'available_memory', lambda: None)
    admitted = execution.plan(config, dict(cpus=4, trials=4))
    assert admitted.trials == 4
    memory = admitted.trial_bytes * 4
    assert execution.plan(config, dict(cpus=4, trials=4, memory_bytes=memory)).trials == 4
    with pytest.raises(ValueError, match='resources admit only 3'):
        execution.plan(config, dict(cpus=4, trials=4, memory_bytes=memory-1))
    with pytest.raises(ValueError, match='resources admit only 0'):
        execution.plan(config, dict(cpus=4, trials=1, memory_bytes=admitted.trial_bytes-1))
    monkeypatch.setattr(execution.os, 'process_cpu_count', lambda: 2)
    with pytest.raises(ValueError) as failure:
        execution.plan(config, dict(cpus=4, trials=4))
    assert 'CPU:' in str(failure.value) and 'RAM:' not in str(failure.value)


@pytest.mark.parametrize('style', ['auto', 'off'])
def test_cli_rejects_reduced_concurrency_before_trials_or_output(tmp_path, style):
    from tests.test_recognition_merge_trial_progress import arguments
    config, goals = fixture(tmp_path/'input', 'disjoint')
    original = config.database.read_bytes()
    checkpoint = tmp_path/'checkpoint.json'
    output = tmp_path/'report.json'
    argv = arguments(config, goals, tmp_path, style) + [
        '--solver-cpus', '4', '--solver-trials', '4', '--solver-memory-gib', '20',
        '--checkpoint', str(checkpoint), '--output', str(output)]
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(repo/'stash_ai.py'), 'recognition', *argv],
        cwd=repo, env=dict(os.environ, PYTHON_CPU_COUNT='4'), capture_output=True, text=True, timeout=30)
    assert result.returncode == 1, result.stderr
    assert 'Cannot start 4 simultaneous solver trials' in result.stderr
    assert 'RAM:' in result.stderr and 'resources admit only 1' in result.stderr
    assert 'baseline: start' not in result.stderr and 'trial-1: start' not in result.stderr
    assert result.stdout == ''
    assert not checkpoint.exists() and not output.exists() and not (tmp_path/'winner.yml').exists()
    assert config.database.read_bytes() == original
