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


def query_observer(db, banks, config, births):
    observer = _observer.get()
    return observer.begin(db, banks, config, births) if observer is not None else None
