"""Cooperative cancellation scoped to the producer and its read operations."""
from contextlib import contextmanager
from threading import Event, Lock, local

_context = local()


class Cancelled(KeyboardInterrupt):
    pass


class Cancellation:
    def __init__(self):
        self.event, self.lock, self.callbacks = Event(), Lock(), set()

    def check(self):
        if self.event.is_set():
            raise Cancelled('Preparation was cancelled.')

    def subscribe(self, callback):
        with self.lock:
            self.callbacks.add(callback)
            cancelled = self.event.is_set()
        if cancelled:
            callback()

    def unsubscribe(self, callback):
        with self.lock:
            self.callbacks.discard(callback)

    def cancel(self):
        with self.lock:
            self.event.set()
            callbacks = tuple(self.callbacks)
        from .cleanup import cleanup
        cleanup(callbacks)


def current():
    return getattr(_context, 'token', None)


@contextmanager
def use(token):
    previous, _context.token = current(), token
    try:
        token.check()
        yield
    finally:
        _context.token = previous
