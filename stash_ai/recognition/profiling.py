"""Opt-in bounded rebuild measurements. Never retain SQL, IDs or embeddings."""
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import time

_active = ContextVar('recognition_rebuild_profile', default=None)
_tables = ('candidates', 'observations', 'members', 'people', 'scans',
           'merge_support', 'identity_merges', 'origins', 'sample_keys')


def count(name, amount=1):
    metrics = _active.get()
    if metrics is not None:
        metrics['counts'][name] += amount


def measured(name):
    def decorate(function):
        @wraps(function)
        def call(*args, **kwargs):
            metrics = _active.get()
            if metrics is None:
                return function(*args, **kwargs)
            start, failed = time.perf_counter(), True
            try:
                result = function(*args, **kwargs)
                failed = False
                return result
            finally:
                row = metrics['stages'].setdefault(name, dict(calls=0, errors=0, seconds=0.))
                row['calls'] += 1
                row['errors'] += int(failed)
                row['seconds'] += time.perf_counter() - start
        return call
    return decorate


class Connection:
    """Borrow one coordinator connection without changing its trace callback."""
    def __init__(self, connection):
        self.raw = connection

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def _call(self, method, sql, *args):
        words = sql.lower().split()
        verb = words[0] if words and words[0] in ('select', 'insert', 'update', 'delete') else 'other'
        table = next((name for name in _tables if name in words), 'other')
        start = time.perf_counter()
        try:
            return getattr(self.raw, method)(sql, *args)
        finally:
            metrics = _active.get()
            if metrics is not None:
                key = verb + ':' + table
                row = metrics['sql'].setdefault(key, dict(calls=0, seconds=0.))
                row['calls'] += 1
                row['seconds'] += time.perf_counter() - start

    def execute(self, sql, *args):
        return self._call('execute', sql, *args)

    def executemany(self, sql, *args):
        return self._call('executemany', sql, *args)

    def executescript(self, sql):
        return self._call('executescript', sql)

    def __enter__(self):
        self.raw.__enter__()
        return self

    def __exit__(self, *error):
        pending = self.raw.in_transaction
        start = time.perf_counter()
        try:
            return self.raw.__exit__(*error)
        finally:
            metrics = _active.get()
            if pending and metrics is not None:
                key = 'rollback:transaction' if error[0] else 'commit:transaction'
                row = metrics['sql'].setdefault(key, dict(calls=0, seconds=0.))
                row['calls'] += 1
                row['seconds'] += time.perf_counter() - start


def connection(raw):
    return Connection(raw) if _active.get() is not None else raw


@contextmanager
def profile(database, enabled):
    if not enabled:
        yield
        return
    from stash_ai.runtime.console import current_report
    import platform
    import sqlite3
    metrics = dict(format=1, counts=Counter(), stages={}, sql={},
                   environment=dict(python=platform.python_version(), sqlite=sqlite3.sqlite_version))
    raw, start = database.connection, time.perf_counter()
    token = _active.set(metrics)
    database.connection = connection(raw)
    try:
        yield
    finally:
        database.connection = raw
        _active.reset(token)
        metrics['elapsed_seconds'] = time.perf_counter() - start
        metrics['counts'] = dict(metrics['counts'])
        database.last_rebuild_profile = metrics
        report = current_report()
        if report is not None:
            report.recognition_rebuild_profile = metrics
