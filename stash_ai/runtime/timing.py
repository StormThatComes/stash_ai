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
        self._heartbeat_condition = threading.Condition(self._timing_lock)
        self._heartbeat_thread = None

    def bind(self, operation):
        """Carry attribution and response accounting into one owned HTTP worker."""
        names = ('model', 'work', 'request')
        captured = {name: getattr(self._timers, name, None) for name in names}
        def bound():
            previous = {name: getattr(self._timers, name, None) for name in names}
            for name, value in captured.items():
                setattr(self._timers, name, value)
            try:
                return operation()
            finally:
                for name, value in previous.items():
                    setattr(self._timers, name, value)
        diagnostics = self.report.diagnostics
        return diagnostics.bind(bound) if diagnostics is not None else bound

    def _start_heartbeat(self):
        # Called under the condition lock. Reuse one ticker across short reads;
        # idle reports release it after a short grace period without a finalizer.
        if self._heartbeat_thread is not None:
            return
        ticker = threading.Thread(target=self._heartbeat, name='stash-request-heartbeat', daemon=True)
        try:
            ticker.start()
        except Exception:
            self.report.counts['heartbeat_start_failures'] += 1
            if ticker.is_alive():
                self._heartbeat_thread = ticker
            return
        self._heartbeat_thread = ticker

    def _heartbeat(self):
        with self._heartbeat_condition:
            while True:
                now = time.perf_counter()
                requests = self.report.active_requests
                if not requests:
                    self._heartbeat_condition.wait(.5)
                    if not requests:
                        self._heartbeat_thread = None
                        return
                    continue
                due = min(value['next_heartbeat'] for value in requests.values())
                if due > now:
                    self._heartbeat_condition.wait(due - now)
                    continue
                for value in tuple(requests.values()):
                    if value['next_heartbeat'] > now:
                        continue
                    value['next_heartbeat'] = now + self.report.heartbeat_seconds
                    try:
                        from .console import status
                        presenter = getattr(self.report, 'console', None)
                        if presenter is not None and not presenter.closed:
                            status(self.report, request=value['operation'], wait='Stash ' + ('read' if value['read'] else 'write'),
                                   request_elapsed=f"{now-value['started']:.1f}s")
                            continue
                        self.report.write(f"Waiting for Stash: {value['operation']}, "
                            f"{now - value['started']:.1f}s, active={len(requests)}, "
                            f"{'read' if value['read'] else 'write'}.")
                    except Exception:
                        self.report.counts['heartbeat_write_failures'] += 1

    @contextmanager
    def measure(self, stage, *, model=None, work=None, units=0, unit=''):
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
                self.report.optional('diagnostics_stage', self.report.diagnostics.record_stage,
                    stage, elapsed, max(0.0, elapsed-entry[1]), failed)
            if self.report.profile is not None:
                self.report.optional('profile_stage', self.report.profile.record,
                    'stage', stage, elapsed, max(0.0, elapsed-entry[1]), failed,
                    model=model if model is not None else getattr(self._timers, 'model', ''),
                    work=work if work is not None else getattr(self._timers, 'work', ''), units=units, unit=unit)

            self.report.record_progress()

    @contextmanager
    def model(self, name):
        previous = getattr(self._timers, 'model', '')
        previous_work = getattr(self._timers, 'work', '')
        self._timers.model = name
        self._timers.work = ''
        try:
            yield
        finally:
            self._timers.model = previous
            self._timers.work = previous_work

    @contextmanager
    def request(self, operation, read):
        start, failed = time.perf_counter(), True
        details = {'response_bytes': 0, 'operation': operation, 'read': read}
        token = object()
        with self._heartbeat_condition:
            self.report.active_requests[token] = dict(operation=operation, read=read, started=start,
                next_heartbeat=start + self.report.heartbeat_seconds)
            try:
                self._start_heartbeat()
            except BaseException:
                self.report.active_requests.pop(token, None)
                raise
            self._heartbeat_condition.notify_all()
        previous = getattr(self._timers, 'request', None)
        self._timers.request = details
        self.request_state(state='request')
        try:
            with self.measure('stash_read' if read else 'stash_write'):
                yield
            failed = False
        finally:
            with self._heartbeat_condition:
                self.report.active_requests.pop(token, None)
                self._heartbeat_condition.notify_all()
            self._timers.request = previous
            from .console import status
            status(self.report, completed=not failed, request=None, wait=None,
                   request_elapsed=None, attempt=None, backoff=None, budget=None, previous_error=None,
                   _backoff_until=None, _deadline=None)
            if self.report.diagnostics is not None:
                self.report.optional('diagnostics_api', self.report.diagnostics.record_api,
                    operation, time.perf_counter()-start, failed, details['response_bytes'])
            if self.report.profile is not None:
                elapsed = time.perf_counter()-start
                self.report.optional('profile_api', self.report.profile.record,
                    'api', operation, elapsed, elapsed,
                    failed, model=getattr(self._timers, 'model', ''), work=getattr(self._timers, 'work', ''),
                    units=details['response_bytes'], unit='bytes')

    def response_size(self, size):
        current = getattr(self._timers, 'request', None)
        if current is not None:
            current['response_bytes'] += size

    def request_state(self, **values):
        details = getattr(self._timers, 'request', None)
        if details is None:
            return
        details.update(values)
        from .console import status
        operation = details['operation']
        with self._timing_lock:
            if values.get('state') == 'request' and 'attempt' in values:
                self.report.counts['http_attempt/' + operation] += 1
            if 'http_status' in values:
                self.report.counts['http_status/' + operation + '/' + str(values['http_status'])] += 1
            reason = values.get('reason')
            if reason:
                self.report.counts['http_reason/' + operation + '/' + reason] += 1
        status(self.report, request=operation, wait=values.get('state'),
               attempt=(f"{values['attempt']}/{values['attempts']}" if 'attempt' in values else None),
               backoff=(f"{values['delay']:.1f}s" if 'delay' in values else None),
               budget=(f"{values['budget']:.1f}s" if 'budget' in values else None),
               previous_error=reason,
               _backoff_until=(time.monotonic()+values['delay'] if 'delay' in values else None),
               _deadline=(time.monotonic()+values['budget'] if 'budget' in values else None))
