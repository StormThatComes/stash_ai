"""Read-only snapshots of bounded input futures and queued batch payloads."""
from threading import Lock


def queued_batches(queue):
    # End markers and failures occupy queue slots but are not buffered batches.
    with queue.mutex:
        return sum(isinstance(value, tuple) for value in queue.queue), queue.maxsize


class PreparedBatchUsage:
    """Retain only futures still owned by the bounded input lookahead."""
    def __init__(self, limit):
        self.limit = limit
        self._pending = set()
        self._lock = Lock()

    def submitted(self, future):
        with self._lock:
            self._pending.add(future)

    def taken(self, future):
        with self._lock:
            self._pending.discard(future)

    def clear(self):
        with self._lock:
            self._pending.clear()

    def snapshot(self):
        with self._lock:
            pending = tuple(self._pending)
        ready = 0
        for future in pending:
            # Never wait for preprocessing, rethrow a producer error or count
            # an empty batch used to mark the end of input.
            if (future.done() and not future.cancelled() and future.exception() is None
                    and len(future.result()[0])):
                ready += 1
        return ready, self.limit
