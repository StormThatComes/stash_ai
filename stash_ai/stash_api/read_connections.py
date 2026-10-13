"""Per-operation socket tracking for bounded reads and single write attempts."""
from contextlib import contextmanager
import socket
from threading import Event, Lock, local

from requests.adapters import HTTPAdapter

_context = local()


def shutdown(sock):
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


class ReadControl:
    def __init__(self):
        self.cancelled, self.lock, self.sockets = Event(), Lock(), {}

    def track(self, sock, owner=None):
        if sock is None:
            return
        with self.lock:
            self.sockets[sock] = owner
            if owner is not None:
                owner._stash_read_control = self
            cancelled = self.cancelled.is_set()
        if cancelled:
            shutdown(sock)

    def cancel(self):
        with self.lock:
            self.cancelled.set()
            sockets = tuple(self.sockets.items())
        for sock, owner in sockets:
            if owner is None or getattr(owner, '_stash_read_control', None) is self:
                shutdown(sock)

    @contextmanager
    def use(self):
        previous, _context.control = getattr(_context, 'control', None), self
        try:
            yield
        finally:
            _context.control = previous


class TrackedConnection:
    def _track(self, sock):
        control = getattr(_context, 'control', None)
        if control is not None:
            control.track(sock, self)
        else:
            self._stash_read_control = None

    def _new_conn(self):
        sock = super()._new_conn()
        self._track(sock)
        return sock

    def request(self, *args, **kwargs):
        self._track(self.sock)
        return super().request(*args, **kwargs)

    def getresponse(self):
        self._track(self.sock)
        return super().getresponse()


def instrument(manager):
    if getattr(manager, '_stash_read_tracking', False):
        return
    pools = {}
    for scheme, pool in manager.pool_classes_by_scheme.items():
        connection = type('ReadConnection', (TrackedConnection, pool.ConnectionCls), {})
        pools[scheme] = type('ReadPool', (pool,), {'ConnectionCls': connection})
    manager.pool_classes_by_scheme = pools
    manager._stash_read_tracking = True


class ReadAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        super().init_poolmanager(*args, **kwargs)
        instrument(self.poolmanager)

    def proxy_manager_for(self, *args, **kwargs):
        manager = super().proxy_manager_for(*args, **kwargs)
        instrument(manager)
        return manager
