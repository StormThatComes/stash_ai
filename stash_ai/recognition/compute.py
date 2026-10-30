"""Coordinator-owned, bounded CPU work; workers never inherit replay state."""
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import os
import threading
import time

from stash_ai.runtime.cancellation import Cancellation, use, current as current_cancellation
from stash_ai.runtime.cleanup import cleanup
from stash_ai.runtime import shutdown

_active = ContextVar('recognition_compute', default=None)


@dataclass(frozen=True)
class Work:
    ordinal: int
    generation: object
    payload: object
    size: int


@dataclass(frozen=True)
class Result:
    ordinal: int
    generation: object
    value: object
    seconds: float
    cpu_seconds: float
    backend: str = 'threads'


def available_cpus():
    return max(1, getattr(os, 'process_cpu_count', os.cpu_count)() or 1)


class LinkedCancellation(Cancellation):
    """Inner numerical work observes both local shutdown and trial cancellation."""
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

    def check(self):
        super().check()
        if self.parent is not None:
            self.parent.check()


class Compute:
    def __init__(self, config):
        settings = config.data['execution']
        available = available_cpus()
        requested = settings.get('rebuild_workers', 1)
        self.requested_workers, self.available = requested, available
        native = settings.get('rebuild_native_threads', 1)
        self.workers = max(1, min(requested or min(16, max(1, available-2)),
                                  max(1, available//native)))
        from .solver_execution import current as solver_execution
        solver = solver_execution()
        if solver is not None:
            self.workers = max(1, solver.inner_workers - int(settings.get('rebuild_prefetch_scenes', False)))
            native = 1
        self.native_threads = native
        from .resource_plan import plan
        self.resources = plan(config)
        self.limit = self.resources.queue
        self.owner = threading.get_ident()
        self.cancel = LinkedCancellation(current_cancellation())
        self.executor = self.native = self.process_executor = self.process_event = None
        self.process_blocks = 0
        self.shared_inputs = {}
        self.pending = deque()
        self.closed = False
        self.bytes = self.peak_bytes = 0
        self.submitted = self.applied = 0
        self.worker_seconds = self.worker_cpu_seconds = self.wait_seconds = 0.
        self.lock = threading.Lock()
        self.running = self.completed = self.failed = 0
        self.fallbacks, self.records, self.preparation = Counter(), {}, Counter()
        self.scenes_applied = 0
        self.last_status = 0.

    def _account_queue(self, size):
        from .run_cache import current
        cache = current()
        if cache is not None:
            cache.reserve(('compute',id(self)),size)

    def fallback(self, reason):
        self.fallbacks[reason] += 1
        self.publish('serial: ' + reason.replace('_', ' '))

    def record(self, kind, *, computed=0, consumed=0, discarded=0):
        row = self.records.setdefault(kind, dict(computed=0, consumed=0, discarded=0))
        for key, value in (('computed', computed), ('consumed', consumed), ('discarded', discarded)):
            row[key] += value
        self.publish()

    @contextmanager
    def preparing(self, kind):
        started = time.perf_counter()
        self.publish('prepare ' + kind)
        try:
            yield
        finally:
            self.preparation[kind + '_seconds'] += time.perf_counter()-started

    def publish(self, reason='coordinator', *, clear=False):
        """Only the coordinator updates UI; worker counters are lock-protected."""
        if threading.get_ident() != self.owner:
            raise RuntimeError('Only the recognition coordinator may publish CPU progress.')
        now = time.monotonic()
        if not clear and now-self.last_status < .25:
            return
        self.last_status = now
        from stash_ai.runtime.console import current_report, status
        with self.lock:
            running, completed = self.running, self.completed
        fields = dict(cpu_workers=f'{running}/{self.workers}',
                      cpu_blocks=f'{completed} computed/{self.applied} retired',
                      cpu_queue=f'{self.bytes}/{self.limit} B', cpu_wait=reason,
                      cpu_scenes=self.scenes_applied)
        status(current_report(), **{k: None if clear else v for k, v in fields.items()})

    def summary(self):
        with self.lock:
            running, completed, failed = self.running, self.completed, self.failed
        return dict(format=2, requested_workers=self.requested_workers,
                    effective_workers=self.workers, available_cpus=self.available,
                    native_threads=self.native_threads, backend='threads',
                    process_cohort_blocks=self.process_blocks,
                    queue_limit_bytes=self.limit, peak_queued_bytes=self.peak_bytes,
                    shared_working_budget_bytes=self.resources.working,
                    cache_budget_bytes=self.resources.cache,
                    submitted_blocks=self.submitted, applied_blocks=self.applied,
                    successful_worker_wall_seconds=self.worker_seconds,
                    successful_worker_thread_cpu_seconds=self.worker_cpu_seconds,
                    ordered_wait_seconds=self.wait_seconds,
                    completed_blocks=completed, failed_blocks=failed, active_workers=running,
                    retired_blocks=self.applied, pending_blocks=len(self.pending),
                    strict_scenes_applied=self.scenes_applied,
                    fallbacks=dict(self.fallbacks), records={k: dict(v) for k,v in self.records.items()},
                    preparation=dict(self.preparation))

    def _owned(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError('Only the recognition coordinator may schedule CPU work.')
        if self.closed:
            raise RuntimeError('Recognition CPU owner is closed.')
        self.cancel.check()

    def _start(self):
        if self.executor is not None:
            return
        shutdown.require_idle()
        from threadpoolctl import threadpool_limits
        self.native = threadpool_limits(limits=self.native_threads, user_api='blas')
        try:
            self.executor = ThreadPoolExecutor(max_workers=self.workers,
                                                thread_name_prefix='recognition-cpu')
        except BaseException:
            cleanup([self.native.restore_original_limits])
            self.native = None
            raise

    def _start_processes(self):
        if self.process_executor is not None:
            return
        import multiprocessing
        shutdown.require_idle()
        context = multiprocessing.get_context('spawn')
        self.process_event = context.Event()
        self.cancel.subscribe(self.process_event.set)
        self.process_executor = ProcessPoolExecutor(max_workers=self.workers, mp_context=context,
            initializer=_initialize_process, initargs=(self.process_event, self.native_threads))

    def _invoke(self, function, job):
        with self.lock:
            self.running += 1
        success = False
        try:
            with use(self.cancel):
                start, cpu = time.perf_counter(), time.thread_time()
                value = function(job.payload)
                self.cancel.check()
                success = True
                return Result(job.ordinal, job.generation, value,
                              time.perf_counter()-start, time.thread_time()-cpu)
        finally:
            with self.lock:
                self.running -= 1
                self.completed += int(success)
                self.failed += int(not success)

    def _take(self):
        future, job = self.pending[0]
        start = time.perf_counter()
        try:
            while not future.done():
                self.cancel.check()
                self.publish('ordered CPU result')
                wait((future,), timeout=.25)
            result = future.result()
        finally:
            self.wait_seconds += time.perf_counter()-start
        if (result.ordinal, result.generation) != (job.ordinal, job.generation):
            raise RuntimeError('Stale or out-of-order recognition CPU result.')
        if result.backend == 'processes':
            with self.lock:
                self.completed += 1
            self.process_blocks += 1
        self.release_input(job.ordinal)
        self.pending.popleft()
        self.bytes -= job.size
        self._account_queue(self.bytes)
        self.applied += 1
        self.worker_seconds += result.seconds
        self.worker_cpu_seconds += result.cpu_seconds
        return result

    def ordered(self, function, jobs, *, backend='threads'):
        """Consume a bounded ordered stream, retiring results before advancing.

        Exhaust or close the iterator before scheduling another stream. Sizes
        reserve input owners, native scratch and results, not just arguments.
        """
        self._owned()
        if backend not in ('threads', 'processes'):
            raise ValueError('Unsupported recognition CPU backend.')
        if self.pending:
            raise RuntimeError('Recognition CPU result stream is still active.')
        previous = -1
        try:
            for job in jobs:
                self._owned()
                if (type(job.size) is not int or not 0 < job.size <= self.limit
                        or type(job.ordinal) is not int or job.ordinal <= previous):
                    raise ValueError('Invalid recognition CPU work size/order.')
                previous = job.ordinal
                if self.workers == 1:
                    yield self._invoke(function, job)
                    continue
                while self.pending and (len(self.pending) >= self.workers*2
                                        or self.bytes + job.size > self.limit):
                    yield self._take()
                self._account_queue(self.bytes+job.size)
                if backend == 'processes':
                    self._start_processes()
                    future = self.process_executor.submit(_invoke_process, function, job)
                else:
                    self._start()
                    future = self.executor.submit(self._invoke, function, job)
                self.pending.append((future, job))
                self.bytes += job.size
                self.peak_bytes = max(self.peak_bytes, self.bytes)
                self.submitted += 1
            while self.pending:
                yield self._take()
        finally:
            if self.pending:
                self.cancel.cancel()
                for future, _ in self.pending:
                    future.cancel()
            self.publish('stopping' if self.pending else 'coordinator')

    def release_input(self, ordinal):
        segment = self.shared_inputs.pop(ordinal, None)
        if segment is not None:
            from .shared_payload import release
            release(segment)

    def release_inputs(self):
        for ordinal in tuple(self.shared_inputs):
            self.release_input(ordinal)

    def _release(self):
        if self.native is not None:
            self.native.restore_original_limits()
            self.native = None
        self.pending.clear()
        self.bytes = 0
        self._account_queue(0)

    def close(self):
        self.closed = True
        def stop():
            self.cancel.cancel()
            for future, _ in self.pending:
                future.cancel()
            actions = []
            for executor in (self.executor, self.process_executor):
                if executor is not None:
                    actions.append(lambda executor=executor: shutdown.call(
                        lambda: executor.shutdown(wait=True, cancel_futures=True),
                        self if executor is self.executor else executor,
                        timeout=None if executor is self.executor else 5.))
            cleanup(actions)
        cleanup([('stop recognition CPU work', stop),
                 ('release recognition CPU owners', lambda: shutdown.release(self, self._release)),
                 ('release process inputs', lambda: shutdown.release(self.process_executor or self, self.release_inputs))])


def current():
    return _active.get()


@contextmanager
def compute_scope(config):
    if current() is not None:
        yield current()
        return
    value = Compute(config)
    token = _active.set(value)
    try:
        yield value
    finally:
        _active.reset(token)
        from .profiling import compute_summary
        cleanup([('close recognition CPU owner', value.close),
                 ('record recognition CPU work', lambda: compute_summary(value.summary())),
                 ('clear recognition CPU status', lambda: value.publish(clear=True))])


_process_cancellation = _process_native = None


def _initialize_process(event, native_threads):
    # Spawned workers receive no SQLite connection, CUDA model or coordinator
    # ContextVars. Only an event, immutable scene input and plain config cross.
    global _process_cancellation, _process_native
    import numpy  # Load BLAS before applying this process-local thread limit.
    from threadpoolctl import threadpool_limits
    _process_native = threadpool_limits(limits=native_threads, user_api='blas')
    _process_cancellation = Cancellation()
    _process_cancellation.event = event


def _invoke_process(function, job):
    with use(_process_cancellation):
        start, cpu = time.perf_counter(), time.process_time()
        value = function(job.payload)
        _process_cancellation.check()
        return Result(job.ordinal, job.generation, value, time.perf_counter()-start,
                      time.process_time()-cpu, 'processes')
