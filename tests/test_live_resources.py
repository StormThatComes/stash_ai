"""Actual process transitions distinguish ordered blocking and ready work."""
from multiprocessing import get_context
from time import monotonic

import pytest

from stash_ai.runtime.cache import RamCache
from stash_ai.runtime.resources import WorkerTelemetry, RollingRates, ResourceMonitor
from stash_ai.runtime.report import RunReport


def compute(telemetry, index, started, release):
    telemetry.started(index)
    started.set()
    release.wait(10)
    telemetry.scene_finished()
    telemetry.finished(index)


def hold_lock(telemetry, entered, release):
    telemetry.lock.acquire()
    entered.set()
    release.wait(10)


def test_terminated_lock_owner_makes_telemetry_unavailable_without_hanging():
    context = get_context('spawn')
    telemetry = WorkerTelemetry(context, 2, 4)
    entered, release = context.Event(), context.Event()
    child = context.Process(target=hold_lock, args=(telemetry, entered, release))
    child.start()
    try:
        assert entered.wait(5)
        child.kill()
        child.join(5)
        started = monotonic()
        assert telemetry.snapshot() is None
        telemetry.queued(1)
        assert monotonic()-started < .25
    finally:
        if child.is_alive(): child.kill()
        child.join(5)


@pytest.mark.parametrize('stage', ['acquire', 'release'])
def test_broken_telemetry_semaphore_disables_observation(stage):
    telemetry = WorkerTelemetry(get_context('spawn'), 2, 4)
    class BrokenLock:
        def acquire(self, **kwargs):
            if stage == 'acquire': raise OSError('semaphore unavailable')
            return True
        def release(self):
            raise OSError('semaphore unavailable')
    telemetry.lock = BrokenLock()
    telemetry.snapshot()
    assert telemetry.snapshot() is None
    telemetry.scene_finished()
    assert not telemetry.available.value


def test_busy_workers_and_ordered_ready_chunks_are_measured_independently():
    context = get_context('spawn')
    telemetry = WorkerTelemetry(context, 2, 4)
    started, release = [context.Event() for _ in range(2)], [context.Event() for _ in range(2)]
    processes = []
    try:
        for index in range(2):
            telemetry.queued(index)
            process = context.Process(target=compute, args=(telemetry, index, started[index], release[index]))
            process.start()
            processes.append(process)
        assert all(event.wait(5) for event in started)
        assert telemetry.snapshot()['busy'] == 2
        release[1].set()
        processes[1].join(5)
        assert processes[1].exitcode == 0
        row = telemetry.snapshot()
        assert row['busy'] == 1 and row['ready_chunks'] == 1
        assert row['computed_scenes'] == 1 and row['queued_chunks'] == 0
        assert row['head_running_seconds'] is not None
        assert row['worker_rss'] is None
        release[0].set()
        processes[0].join(5)
        assert processes[0].exitcode == 0
        assert telemetry.snapshot()['computed_scenes'] == 2
        telemetry.received(0)
        telemetry.received(1)
        assert telemetry.snapshot()['busy'] == telemetry.snapshot()['ready_chunks'] == 0
    finally:
        for event in release: event.set()
        for process in processes:
            if process.is_alive(): process.kill()
            process.join(5)


def test_rolling_rates_warm_up_reset_gaps_and_keep_bounded_storage():
    rates = RollingRates()
    assert rates.sample(0., evidence=0, publication=0, backlog=0)['state'] == 'warming up'
    row = rates.sample(30., evidence=12, publication=2, backlog=10)
    assert row['evidence'] == .4 and row['publication'] == 2/30
    assert row['backlog'] == 10/30
    assert rates.sample(100., evidence=12, publication=2, backlog=10)['state'] == 'warming up'
    for second in range(101, 300): rates.sample(float(second), evidence=second)
    assert len(rates.samples) <= 61
    assert 30 <= rates.sample(300., evidence=300)['window_seconds'] <= 31


def test_cache_snapshot_does_not_touch_entries_or_pressure_and_sampler_is_throttled(monkeypatch):
    cache = RamCache(10000)
    cache.put('a', 'data')
    before = list(cache.entries), cache.size, cache.stats.copy()
    monkeypatch.setattr(cache, 'pressure', lambda **kw: (_ for _ in ()).throw(AssertionError('pressure called')))
    assert cache.snapshot()['retained_bytes'] == before[1]
    assert (list(cache.entries), cache.size, cache.stats) == before
    rss = []
    monkeypatch.setattr('stash_ai.runtime.resources.memory.process_rss', lambda: rss.append(True) or 1234)
    monitor = ResourceMonitor(RunReport(), cache)
    for _ in range(100): monitor.sample()
    assert len(rss) == 1
