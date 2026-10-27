"""Optional read-only diagnostics at the frozen query stage, before gate pruning."""
from contextlib import contextmanager
from contextvars import ContextVar

_observer = ContextVar('hybrid_query_observer', default=None)


@contextmanager
def observe_queries(observer):
    token = _observer.set(observer)
    try:
        yield
    finally:
        _observer.reset(token)


def query_observer(db, banks, config, births, selected=()):
    observer = _observer.get()
    if observer is None:
        return None
    observer.selected = selected
    return observer.begin(db, banks, config, births)


def query_decision(observation, stage, reason, **details):
    observer = _observer.get()
    if observer is not None:
        observer.decision(observation, stage, reason, details)
