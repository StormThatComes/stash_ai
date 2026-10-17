"""Coordinator-owned, bounded CPU work; workers never inherit replay state."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import os
import threading
import time

from stash_ai.runtime.cancellation import Cancellation, use
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


def available_cpus():
    return max(1, getattr(os, 'process_cpu_count', os.cpu_count)() or 1)


class Compute:
    def __init__(self, config):
        settings = config.data['execution']
        available = available_cpus()
        requested = settings.get('rebuild_workers', 1)
        native = settings.get('rebuild_native_threads', 1)
        self.workers = max(1, min(requested or min(16, max(1, available-2)),
                                  max(1, available//native)))
        self.native_threads = native
        self.limit = min(settings.get('rebuild_queue_bytes', 67108864),
                         settings['rebuild_memory_bytes']//16)
        self.owner = threading.get_ident()
        self.cancel = Cancellation()
        self.executor = self.native = None
        self.pending = deque()
        self.closed = False
        self.bytes = self.peak_bytes = 0
        self.submitted = self.applied = 0
        self.worker_seconds = self.worker_cpu_seconds = self.wait_seconds = 0.

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

    def _invoke(self, function, job):
        with use(self.cancel):
            start, cpu = time.perf_counter(), time.thread_time()
            value = function(job.payload)
            self.cancel.check()
            return Result(job.ordinal, job.generation, value,
                          time.perf_counter()-start, time.thread_time()-cpu)

    def _take(self):
        future, job = self.pending[0]
        start = time.perf_counter()
        try:
            result = future.result()
        finally:
            self.wait_seconds += time.perf_counter()-start
        if (result.ordinal, result.generation) != (job.ordinal, job.generation):
            raise RuntimeError('Stale or out-of-order recognition CPU result.')
        self.pending.popleft()
        self.bytes -= job.size
        self.applied += 1
        self.worker_seconds += result.seconds
        self.worker_cpu_seconds += result.cpu_seconds
        return result

    def ordered(self, function, jobs):
        """Consume a bounded ordered stream, retiring results before advancing.

        Exhaust or close the iterator before scheduling another stream. Sizes
        reserve input owners, native scratch and results, not just arguments.
        """
        self._owned()
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

    def _release(self):
        if self.native is not None:
            self.native.restore_original_limits()
            self.native = None
        self.pending.clear()
        self.bytes = 0

    def close(self):
        self.closed = True
        def stop():
            self.cancel.cancel()
            for future, _ in self.pending:
                future.cancel()
            if self.executor is not None:
                shutdown.call(lambda: self.executor.shutdown(wait=True, cancel_futures=True), self)
        cleanup([('stop recognition CPU work', stop),
                 ('release recognition CPU owners', lambda: shutdown.release(self, self._release))])


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
        cleanup([('close recognition CPU owner', value.close)])
