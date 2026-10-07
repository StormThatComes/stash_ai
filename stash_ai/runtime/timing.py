"""Exclusive per-report stage timers and nested request accounting."""
from collections import Counter
from contextlib import contextmanager, nullcontext
from functools import wraps
import threading
import time


def stage(report, name):
    return report.measure(name) if report is not None else nullcontext()


def measured(stage):
    """Report a method's exclusive time when a run report is attached."""
    def decorate(method):
        @wraps(method)
        def wrapped(self, *args, **kwargs):
            report = self.report
            if report is None:
                return method(self, *args, **kwargs)
            with report.measure(stage):
                return method(self, *args, **kwargs)
        return wrapped
    return decorate


class RunTimings:
    """Each report owns its counters, thread-local stacks and diagnostic target."""
    def __init__(self, report):
        self.report = report
        self.timings, self.timing_calls = Counter(), Counter()
        self._timers = threading.local()
        self._timing_lock = threading.RLock()

    @contextmanager
    def measure(self, stage):
        stack = getattr(self._timers, 'stack', None)
        if stack is None:
            stack = self._timers.stack = []
        entry = [time.perf_counter(), 0.0]
        stack.append(entry)
        failed = True
        try:
            yield
            failed = False
        finally:
            elapsed = time.perf_counter() - entry[0]
            stack.pop()
            if stack:
                stack[-1][1] += elapsed
            with self._timing_lock:
                self.timings[stage] += max(0.0, elapsed-entry[1])
                self.timing_calls[stage] += 1
            if self.report.diagnostics is not None:
                self.report.diagnostics.record_stage(stage, elapsed, max(0.0, elapsed-entry[1]), failed)

    @contextmanager
    def request(self, operation, read):
        start, failed = time.perf_counter(), True
        details = {'response_bytes': 0}
        token, stopped = object(), threading.Event()
        with self._timing_lock:
            self.report.active_requests[token] = dict(operation=operation, read=read, started=start)

        def heartbeat():
            while not stopped.wait(self.report.heartbeat_seconds):
                with self._timing_lock:
                    active = len(self.report.active_requests)
                self.report.write(f'Waiting for Stash: {operation}, '
                    f'{time.perf_counter() - start:.1f}s, active={active}, '
                    f'{"read" if read else "write"}.')

        ticker = threading.Thread(target=heartbeat, name='stash-request-heartbeat', daemon=True)
        ticker.start()
        previous = getattr(self._timers, 'request', None)
        self._timers.request = details
        try:
            with self.measure('stash_read' if read else 'stash_write'):
                yield
            failed = False
        finally:
            stopped.set()
            ticker.join()
            with self._timing_lock:
                self.report.active_requests.pop(token, None)
            self._timers.request = previous
            if self.report.diagnostics is not None:
                self.report.diagnostics.record_api(operation, time.perf_counter()-start, failed, details['response_bytes'])

    def response_size(self, size):
        current = getattr(self._timers, 'request', None)
        if current is not None:
            current['response_bytes'] += size
