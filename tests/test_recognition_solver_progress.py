"""Live worker telemetry must remain truthful, bounded and outside scheduling."""
from concurrent.futures import Future
from queue import Queue
from types import SimpleNamespace
import multiprocessing
import os
import time

import pytest

from stash_ai.recognition.progress import observe_updates, stage, tracking, work
from stash_ai.recognition.solver_progress import LiveProgress, ReplayProgress, worker_text


def test_observed_work_without_console_restores_nested_rows_and_counts_only_completed(monkeypatch):
    from stash_ai.recognition import progress as module
    from stash_ai.recognition import solver_progress
    clock = [10.]
    fake_time = SimpleNamespace(monotonic=lambda: clock[0])
    monkeypatch.setattr(module, 'time', fake_time)
    monkeypatch.setattr(solver_progress, 'time', fake_time)
    snapshots = []
    reporter = ReplayProgress('trial-1', lambda n, s, v: snapshots.append(v))
    with observe_updates(reporter.update), stage('Rebuild', scene='42'):
        with work('Scenes', 3, position=0, unit='scene') as outer:
            clock[0] += 1
            outer.advance()
            with stage('Inner'), work('Observations', 2, position=0) as inner:
                clock[0] += 1
                inner.advance()
                assert snapshots[-1]['work'][0]['label'] == 'Observations'
            assert snapshots[-1]['stage'] == 'Rebuild'
            assert snapshots[-1]['scene'] == '42'
            assert snapshots[-1]['work'][0]['completed'] == 1
            with pytest.raises(ValueError), tracking([1, 2], 'Validate', position=1) as items:
                for item in items:
                    clock[0] += 1
                    if item == 2:
                        raise ValueError('failed second record')
            checks = [r for s in snapshots for r in s['work'] if r['label'] == 'Validate']
            assert max(r['completed'] for r in checks) == 1
    assert reporter.stages == [] and reporter.rows == {}
    count = len(snapshots)
    with stage('Unobserved'), work('Unused', 1) as row:
        row.advance()
    assert len(snapshots) == count
    text = worker_text(snapshots[-1], clock[0] + 60)
    assert 'last_update=60s ago' in text
    assert 'last_completed=61s ago' in text  # The failed record did not advance work.


def test_full_transport_retries_latest_snapshot_without_changing_work_time():
    queue = Queue(maxsize=1)
    queue.put('occupied')
    live = LiveProgress('trial-1', queue)
    initial = live.snapshot
    live.publish()  # Must return even when the receiver is not draining.
    latest = dict(initial, stage='Blocked SQL', updated=initial['updated'] + 1)
    live('trial-1', 'stage: Blocked SQL', latest)
    live.publish()
    assert queue.get_nowait() == 'occupied'
    live.publish()
    name, snapshot = queue.get_nowait()
    assert name == 'trial-1'
    assert snapshot.pop('reported') >= initial['started']
    assert snapshot == latest
    live.publish()
    assert queue.get_nowait()[1]['updated'] == latest['updated']
    with live:
        pass
    assert not live.thread.is_alive()


def publish_while_blocked(queue, gate):
    # Real spawn transport, with no stdout/stderr or worker console presenter.
    queue.cancel_join_thread()
    with LiveProgress('trial-1', queue) as publisher:
        reporter = ReplayProgress('trial-1', publisher)
        with observe_updates(reporter.update), stage('Slow proof', scene='7'):
            with work('Prepared proofs', 25) as row:
                row.advance(25)
                if not gate.wait(10):
                    raise RuntimeError('test coordinator did not release worker')
    return os.getpid()


def test_spawned_worker_reports_stage_during_blocking_work_and_shuts_down():
    context = multiprocessing.get_context('spawn')
    queue, gate = context.Queue(maxsize=4), context.Event()
    # Spawn args must be inherited through initialization, not pickled as a job.
    process = context.Process(target=publish_while_blocked, args=(queue, gate))
    process.start()
    try:
        snapshots = []
        deadline = time.monotonic() + 10
        while len(snapshots) < 2 and time.monotonic() < deadline:
            name, snapshot = queue.get(timeout=5)
            if snapshot['stage'] == 'Slow proof':
                snapshots.append(snapshot)
        assert len(snapshots) == 2
        assert snapshots[0]['pid'] == process.pid != os.getpid()
        assert snapshots[0]['work'][0]['completed'] == 25
        assert snapshots[0]['updated'] == snapshots[1]['updated']
        assert snapshots[0]['last_completed'] == snapshots[1]['last_completed']
    finally:
        gate.set()
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join()
        queue.close()
        queue.join_thread()
    assert process.exitcode == 0


def publish_without_reader(queue):
    from stash_ai.recognition.solver_pool import initialize
    initialize(None, queue)
    with LiveProgress('trial-1', queue) as publisher:
        for _ in range(1000):
            publisher.publish()


def test_full_spawn_transport_does_not_block_worker_exit():
    context = multiprocessing.get_context('spawn')
    queue = context.Queue(maxsize=2)
    process = context.Process(target=publish_without_reader, args=(queue,))
    process.start()
    try:
        process.join(10)
        assert not process.is_alive()
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.kill()
            process.join()
        queue.close()
        queue.join_thread()


@pytest.mark.parametrize('scheduling', ['batch', 'async'])
def test_ready_result_is_visible_before_application_without_changing_order(tmp_path, monkeypatch, scheduling):
    from stash_ai.recognition import solver_pool
    from stash_ai.recognition.solver_execution import Execution
    clock = [100.]
    monkeypatch.setattr(solver_pool, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    events = []
    execution = Execution(4, 3, 1, 1024**3, 1024, scheduling=scheduling)
    with solver_pool.Trials(execution, None, dict(seed_path=str(tmp_path/'seed')),
            progress=lambda n, s, v: events.append((n, s, v))) as pool:
        first, second, third = Future(), Future(), Future()
        # Future.running() is not proof the spawned process has started a trial.
        third.set_running_or_notify_cancel()
        pool.pending.update({'trial-1': (None, first), 'trial-2': (None, second), 'trial-3': (None, third)})
        pool.dispatched.update({n: 90. for n in pool.pending})
        pool.snapshots['trial-1'] = dict(pid=123, started=90., stage='Compare bodies',
            stage_started=95., updated=96., last_completed=None, scene=None, work=[])
        second.set_result(dict(elapsed_seconds=2.5))
        pool.harvest()
        ready = next(v['text'] for n, s, v in events if n == 'trial-2' and s == 'ready')
        assert 'awaiting_application' in ready
        assert ('waiting_for=trial-1' in ready) == (scheduling == 'batch')
        summary = next(v['text'] for n, s, v in events if s == 'pool')
        assert 'running=1 queued/starting=1 ready=1 retired=0' in summary
        assert list(pool.pending) == ['trial-1', 'trial-2', 'trial-3']
        before = len(events)
        pool.harvest()
        assert len(events) == before  # Periodic output and readiness are not spammed.
        clock[0] += 60
        pool.harvest()
        stale = [v['text'] for n, s, v in events if n == 'trial-1' and s == 'status'][-1]
        assert 'last_update=64s ago' in stale and 'last_completed=none observed' in stale
        first.set_result(dict(elapsed_seconds=70.))
        third.set_result(dict(elapsed_seconds=70.))
        assert pool.retire('trial-1')[0] == 'trial-1'
        assert 'trial-1' not in pool.snapshots and pool.retired == 1
        assert sum(s == 'ready' and n == 'trial-2' for n, s, _ in events) == 1


def test_disabled_telemetry_creates_no_transport(tmp_path):
    from stash_ai.recognition.solver_execution import Execution
    from stash_ai.recognition.solver_pool import Trials
    def callback(*args):
        pytest.fail('no console callback expected')
    callback.telemetry_enabled = False
    with Trials(Execution(2, 2, 1, 1024**3, 1024), None,
                dict(seed_path=str(tmp_path/'seed')), progress=callback) as pool:
        assert pool.updates is None
        pool.harvest()


@pytest.mark.parametrize('style', ['auto', 'off'])
def test_parallel_cli_keeps_json_clean_and_reports_ready_and_completion(tmp_path, capsys, style):
    import json
    from benchmarks.benchmark_merge_solver import fixture
    from stash_ai.recognition.cli import main
    from tests.test_recognition_merge_trial_progress import arguments
    config, goals = fixture(tmp_path/'input', 'disjoint')
    argv = arguments(config, goals, tmp_path, style) + ['--solver-cpus', '4', '--solver-trials', '2', '--solver-memory-gib', '128']
    assert main(argv) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert len(result['simulation']['trials']) == 3
    if style == 'off':
        assert captured.err == ''
    else:
        request = captured.err.index('execution: request cpus=4 trials=2')
        preparation = captured.err.index('stage=Prepare report inputs')
        admitted = captured.err.index('execution: plan cpus=')
        measurements = captured.err.index('stage: Measure retained evidence')
        assert request < preparation < admitted < measurements
        assert captured.err.count('execution: plan ') == 1
        assert 'trials=2 requested_trials=2' in captured.err
        assert 'limited_by=none' in captured.err
        assert 'execution: plan cpus=' in captured.err
        assert 'workers: pool running=' in captured.err
        assert 'trial-1: ready evaluation=finished' in captured.err
        assert 'trial-1: complete disjoint=' in captured.err
        assert 'completed=3/3' in captured.err
        assert 'running/queued=' not in captured.err
