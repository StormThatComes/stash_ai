"""Opt-in host elapsed time between snapshots, excluding snapshot bookkeeping."""
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter


@dataclass(frozen=True)
class Timing:
    seconds: float
    total_seconds: float


class Clock:
    def __init__(self):
        self.started = None
        self.excluded = self.previous = 0.
        self.paused_at = None
        self.final = None

    def start(self):
        self.started = perf_counter()

    @contextmanager
    def snapshot(self):
        # Nested helpers and History.save share the timestamp taken before
        # copying points, packing masks and checking the history memory budget.
        if self.started is None or self.paused_at is not None or self.final is not None:
            yield
            return
        started = perf_counter()
        self.paused_at = max(self.previous, started - self.started - self.excluded)
        try:
            yield
        finally:
            self.excluded += perf_counter() - started
            self.paused_at = None

    def checkpoint(self):
        if self.final is not None:
            return self.final
        if self.started is None:
            return Timing(0., 0.)
        total = self.paused_at
        if total is None:
            total = max(self.previous, perf_counter() - self.started - self.excluded)
        result = Timing(total - self.previous, total)
        self.previous = total
        return result

    def finish(self):
        if self.final is None:
            self.final = self.checkpoint()
        return self.final


def duration(seconds):
    return f'{seconds * 1000:.1f} ms' if seconds < 1 else f'{seconds:.3f} s'


def caption(history, position):
    if history is None:
        return ''
    final = position == len(history.phases)
    timing = history.clock.final if final else history.phases[position].timing
    if timing is None:
        return ' | Timing unavailable'
    label = 'Remaining (includes omitted phases)' if final and history.omitted else 'Step'
    return f' | {label}: {duration(timing.seconds)} | Total: {duration(timing.total_seconds)}'
