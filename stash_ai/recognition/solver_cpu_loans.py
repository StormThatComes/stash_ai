"""Bounded CPU loans shared by the spawned workers of one solver."""
from contextlib import contextmanager

_state = None


def initialize(state):
    global _state
    _state = state


@contextmanager
def active():
    if _state is None:
        yield
        return
    with _state.get_lock():
        _state[1] += 1
    try:
        yield
    finally:
        with _state.get_lock():
            _state[1] -= 1


def capacity():
    return max(1, _state[0]) if _state is not None else None


def _available():
    helpers, active_workers, in_use = _state[:]
    return min(helpers // max(1, active_workers), helpers - in_use)


def available():
    with _state.get_lock():
        return max(1, _available())


@contextmanager
def borrow():
    with _state.get_lock():
        amount = _available()
        # A one-worker block executes directly on its coordinator.
        amount = amount if amount > 1 else 0
        _state[2] += amount
    try:
        yield max(1, amount)
    finally:
        with _state.get_lock():
            _state[2] -= amount


def prefetch():
    """Reserve one asynchronous reader, or let the caller prepare synchronously."""
    if _state is None:
        return True
    with _state.get_lock():
        if _available() <= 0:
            return False
        _state[2] += 1
    return True


def release_prefetch():
    if _state is not None:
        with _state.get_lock():
            _state[2] -= 1
