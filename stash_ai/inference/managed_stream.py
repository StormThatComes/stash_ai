"""Bounded producer/consumer lifecycle shared by input and result streams."""
from dataclasses import dataclass
from queue import Empty, Full
from threading import Event

from stash_ai.runtime.cancellation import Cancellation
from stash_ai.runtime.cleanup import cleanup
from .buffer_usage import BatchQueue


@dataclass(frozen=True)
class StreamFailure:
    error: BaseException


class ManagedStream:
    def __init__(self, limit):
        if type(limit) is not int or not 1 <= limit <= 128:
            raise ValueError('Stream limit must be an integer in 1..128.')
        self.queue, self.stopped = BatchQueue(maxsize=limit), Event()
        self.cancellation, self.end, self.worker = Cancellation(), object(), None

    def send(self, value):
        while not self.stopped.is_set():
            try:
                self.queue.put(value, timeout=.05)
                return
            except Full:
                pass

    def receive(self):
        while not self.stopped.is_set():
            try:
                return self.queue.get(timeout=.05)
            except Empty:
                worker = self.worker
                if worker is not None and hasattr(worker, 'done') and worker.done():
                    if not worker.cancelled():
                        worker.result()
                    return self.end
                if (worker is not None and hasattr(worker, 'is_alive')
                        and worker.ident is not None and not worker.is_alive()):
                    raise RuntimeError('Producer thread exited without completing its stream.')
        return self.end

    def stop(self):
        # Stop queue writers before a cancellation callback can fail.
        self.stopped.set()
        cleanup([self.cancellation.cancel,
                 lambda: self.worker.cancel() if self.worker is not None and hasattr(self.worker, 'cancel') else None])

    def join(self):
        if self.worker is None:
            return
        if hasattr(self.worker, 'result'):
            if not self.worker.cancelled():
                self.worker.result()
        else:
            if self.worker.ident is not None:
                self.worker.join()

    def drain(self):
        while True:
            try:
                self.queue.get_nowait()
            except Empty:
                self.queue.consuming(False)
                return

    def close(self):
        cleanup([self.stop, self.join, self.drain])
