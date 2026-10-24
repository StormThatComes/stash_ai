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


def compute_summary(value):
    """Called only by the coordinator; worker contexts are never shared."""
    metrics = _active.get()
    if metrics is not None:
        metrics['cpu'] = value


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


class Cursor:
    """Measure lazy SQLite work, separately from executing a statement."""
    def __init__(self, raw, metrics, key):
        self.raw, self.metrics, self.key = raw, metrics, key

    def __getattr__(self, name):
        return getattr(self.raw, name)

    @property
    def arraysize(self):
        return self.raw.arraysize

    @arraysize.setter
    def arraysize(self, value):
        self.raw.arraysize = value

    def _fetch(self, method, *args):
        start = time.perf_counter()
        count = 0
        try:
            result = getattr(self.raw, method)(*args)
            count = int(result is not None) if method == 'fetchone' else len(result)
            return result
        finally:
            row = self.metrics['sql_fetch'].setdefault(self.key, dict(calls=0, rows=0, seconds=0.))
            row['calls'] += 1
            row['rows'] += count
            row['seconds'] += time.perf_counter() - start

    def fetchone(self):
        return self._fetch('fetchone')

    def fetchmany(self, *args):
        return self._fetch('fetchmany', *args)

    def fetchall(self):
        return self._fetch('fetchall')

    def __iter__(self):
        return self

    def __next__(self):
        row = self.fetchone()
        if row is None:
            raise StopIteration
        return row


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
        metrics = _active.get()
        key = verb + ':' + table
        start = time.perf_counter()
        try:
            cursor = getattr(self.raw, method)(sql, *args)
            return Cursor(cursor, metrics, key) if metrics is not None else cursor
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
    metrics = dict(format=1, counts=Counter(), stages={}, sql={}, sql_fetch={},
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


_pipeline = ContextVar('recognition_pipeline_profile', default=None)


@contextmanager
def pipeline_scope(enabled):
    if not enabled:
        yield
        return
    metrics = dict(format=1, stages={}, scope='coordinator elapsed time; nested stages overlap')
    token = _pipeline.set((metrics, time.perf_counter()))
    try:
        yield
    finally:
        _pipeline.reset(token)


def pipeline_snapshot():
    active = _pipeline.get()
    if active is None:
        return None
    metrics, start = active
    return dict(metrics, elapsed_seconds=time.perf_counter()-start,
                stages={key: dict(value) for key, value in metrics['stages'].items()})


def pipeline_stage(name):
    def decorate(function):
        @wraps(function)
        def call(*args, **kwargs):
            active = _pipeline.get()
            if active is None:
                return function(*args, **kwargs)
            start, failed = time.perf_counter(), True
            try:
                result = function(*args, **kwargs)
                failed = False
                return result
            finally:
                row = active[0]['stages'].setdefault(name, dict(calls=0, errors=0, seconds=0.))
                row['calls'] += 1
                row['errors'] += int(failed)
                row['seconds'] += time.perf_counter()-start
        return call
    return decorate
