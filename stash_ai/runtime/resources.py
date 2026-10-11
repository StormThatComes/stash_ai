"""Sparse process transitions and bounded rolling scalar resource observations."""
from collections import deque
from contextlib import contextmanager
from time import monotonic
from functools import wraps

from . import memory


def guarded(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        if not self.available.value:
            return None
        # A terminated process can die holding a semaphore. Diagnostics must
        # never prevent pool shutdown or publication from continuing.
        if not self.lock.acquire(timeout=.02):
            self.available.value = False
            return None
        try:
            return method(self, *args, **kwargs)
        finally:
            self.lock.release()
    return wrapped


class WorkerTelemetry:
    """One fixed ring for map's bounded chunk window; never stores payloads."""
    def __init__(self, context, workers, capacity):
        self.workers, self.capacity = workers, capacity
        self.lock = context.RLock()
        self.available = context.Value('b', True, lock=False)
        self.ids = context.Array('q', [-1] * capacity, lock=False)
        self.states = context.Array('b', capacity, lock=False)
        self.times = context.Array('d', capacity, lock=False)
        # offered chunks, finished chunks, successful scenes, failed scenes
        self.counts = context.Array('q', 4, lock=False)
        self.last_completion = context.Value('d', 0., lock=False)
        self.next_chunk = 0

    @guarded
    def queued(self, index):
        slot = index % self.capacity
        self.ids[slot], self.states[slot], self.times[slot] = index, 1, 0.
        self.counts[0] += 1

    @guarded
    def started(self, index):
        slot = index % self.capacity
        self.states[slot], self.times[slot] = 2, monotonic()

    @guarded
    def scene_finished(self, failed=False):
        self.counts[3 if failed else 2] += 1
        self.last_completion.value = monotonic()

    @guarded
    def finished(self, index):
        self.states[index % self.capacity] = 3
        self.counts[1] += 1

    @guarded
    def received(self, index):
        self.states[index % self.capacity] = 0
        self.next_chunk = index+1

    @guarded
    def snapshot(self):
        states = tuple(self.states)
        slot = self.next_chunk % self.capacity
        head_age = (max(0., monotonic()-self.times[slot])
                    if self.ids[slot] == self.next_chunk and states[slot] == 2 else None)
        return dict(workers=self.workers, busy=states.count(2), queued_chunks=states.count(1),
                    running_chunks=states.count(2), ready_chunks=states.count(3), capacity_chunks=self.capacity-1,
                    offered_chunks=self.counts[0], finished_chunks=self.counts[1],
                    computed_scenes=self.counts[2], failed_scenes=self.counts[3],
                    last_completion=self.last_completion.value or None, head_running_seconds=head_age,
                    worker_rss=None)


class RollingRates:
    def __init__(self, window=30.):
        self.window = window
        self.samples = deque(maxlen=61)

    def sample(self, now, **counts):
        if self.samples and now-self.samples[-1][0] > self.window:
            self.samples.clear()
        if not self.samples or now-self.samples[-1][0] >= 1.:
            self.samples.append((now, counts))
        while len(self.samples) > 2 and now-self.samples[1][0] >= self.window:
            self.samples.popleft()
        if len(self.samples) < 2 or now-self.samples[0][0] < self.window:
            return dict(state='warming up', window_seconds=self.window)
        then, before = self.samples[0]
        elapsed = now-then
        return dict(state='measured', window_seconds=elapsed,
                    **{key: (value-before.get(key, value))/elapsed for key, value in counts.items()})


class ResourceMonitor:
    def __init__(self, report, cache, telemetry=None, live=None):
        self.report, self.cache, self.telemetry, self.live = report, cache, telemetry, live
        self.rates, self.last_sample, self.row = RollingRates(), 0., {}
        self.previous = cache.snapshot()
        self.rates.sample(monotonic(), evidence=0, publication=0, backlog=0)

    def sample(self):
        now = monotonic()
        if now-self.last_sample < 1.:
            return self.row
        self.last_sample = now
        cache = self.cache.snapshot()
        workers = self.telemetry.snapshot() if self.telemetry is not None else None
        received = self.live['evidence_received'] if self.live is not None else 0
        published = self.report.counts['publication_committed']
        computed = workers['computed_scenes'] if workers is not None else 0
        failed = workers['failed_scenes'] if workers is not None else 0
        # Received is an envelope count, including failed envelopes.
        backlog = max(0, computed+failed-received) if workers is not None else None
        self.row = dict(cache=cache, parent_rss=memory.process_rss(), workers=workers,
                        ready_scenes=backlog,
                        evictions_delta=cache['evictions']-self.previous['evictions'],
                        pressure_trims_delta=cache['pressure_trims']-self.previous['pressure_trims'],
                        rates=self.rates.sample(now, evidence=computed, publication=published, backlog=backlog or 0)
                        if workers is not None else None)
        self.previous = cache
        self.report.workload['resources'] = self.row
        return self.row


@contextmanager
def observe(report, key, sample):
    ui = getattr(report, 'console', None)
    if ui is None or ui.closed or ui.style == 'off':
        yield
        return
    ui.register_sampler(key, sample)
    try:
        yield
    finally:
        ui.remove_sampler(key)


def resource_line(row):
    if not row:
        return ''
    parts, workers = [], row.get('workers')
    if workers is not None:
        parts.append(f'workers={workers["busy"]}/{workers["workers"]} busy')
        parts.append(f'chunks q/run/ready={workers["queued_chunks"]}/{workers["running_chunks"]}/{workers["ready_chunks"]} capacity={workers["capacity_chunks"]}')
        parts.append(f'ready scene results={row["ready_scenes"]}')
        if workers['head_running_seconds'] is not None:
            parts.append(f'ordered head running={workers["head_running_seconds"]:.1f}s')
        rates = row.get('rates') or {}
        parts.append(('rates=warming up' if rates.get('state') != 'measured' else
                      f'evidence/publish={rates["evidence"]:.2f}/{rates["publication"]:.2f} scene/s backlogΔ={rates["backlog"]:+.2f}/s'))
    cache = row['cache']
    mib = 2**20
    parts.append(f'parent cache={cache["retained_bytes"]/mib:.0f}/{cache["effective_bytes"]/mib:.0f}/{cache["configured_bytes"]/mib:.0f}MiB retained/effective/configured')
    parts.append('parent RSS=' + (f'{row["parent_rss"]/mib:.0f}MiB' if row.get('parent_rss') is not None else 'unavailable'))
    parts.append(f'evictionsΔ={row["evictions_delta"]} trimsΔ={row["pressure_trims_delta"]}')
    return 'Resources: ' + ' | '.join(parts)
