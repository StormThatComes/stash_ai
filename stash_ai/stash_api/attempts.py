"""One mutation dispatch gate; expiry cannot authorize a later network write."""
from threading import Lock

from .deadlines import ReadDeadlineExceeded, remaining


class MutationAttempt:
    def __init__(self):
        self.lock = Lock()
        self.entered = self.expired = False

    def dispatch(self, callback, timeout, deadline):
        with self.lock:
            remaining(deadline)
            if self.expired:
                raise ReadDeadlineExceeded('Mutation expired before dispatch.')
            self.entered = True
        return callback(timeout)

    def expire(self):
        with self.lock:
            self.expired = True
            return self.entered
