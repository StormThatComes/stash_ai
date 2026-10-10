"""Read-only snapshots of bounded input futures and queued batch payloads."""
from threading import Lock
from dataclasses import dataclass
from queue import Queue


@dataclass(frozen=True)
class BufferUsage:
    ready: int
    in_flight: int
    capacity: int

    @property
    def occupied(self):
        return self.ready + self.in_flight


class BatchQueue(Queue):
    """A producer and consumer can each own a batch outside the ready queue."""
    producer = consumer = 0

    def producing(self, active):
        with self.mutex:
            self.producer = int(active)

    def consuming(self, active):
        with self.mutex:
            self.consumer = int(active)

    def _put(self, value):
        super()._put(value)
        if isinstance(value, tuple):
            # Transfer ownership atomically with enqueue, so a snapshot cannot
            # count the same batch as both ready and in flight.
            self.producer = 0

    def _get(self):
        value = super()._get()
        if isinstance(value, tuple):
            self.consumer = 1
        return value


def queued_batches(queue):
    # End markers and failures occupy queue slots but are not buffered batches.
    with queue.mutex:
        return BufferUsage(sum(isinstance(value, tuple) for value in queue.queue),
                           getattr(queue, 'producer', 0) + getattr(queue, 'consumer', 0),
                           queue.maxsize + 2 * int(isinstance(queue, BatchQueue)))


class PreparedBatchUsage:
    """Retain only futures still owned by the bounded input lookahead."""
    def __init__(self, limit):
        self.limit = limit
        self._pending = set()
        self._consumer = 0
        self._lock = Lock()

    def submitted(self, future):
        with self._lock:
            self._pending.add(future)

    def taken(self, future):
        with self._lock:
            self._pending.discard(future)
            self._consumer = int(future.done() and not future.cancelled() and future.exception() is None
                                 and bool(len(future.result()[0])))

    def release(self):
        with self._lock:
            self._consumer = 0

    def clear(self):
        with self._lock:
            self._pending.clear()
            self._consumer = 0

    def snapshot(self):
        with self._lock:
            pending = tuple(self._pending)
            in_flight = self._consumer
        ready = 0
        for future in pending:
            # Never wait for preprocessing, rethrow a producer error or count
            # an empty batch used to mark the end of input.
            if not future.done():
                in_flight += 1  # Scheduled or currently executing preparation.
            elif (not future.cancelled() and future.exception() is None
                    and len(future.result()[0])):
                ready += 1
        return BufferUsage(ready, in_flight, self.limit + 1)
