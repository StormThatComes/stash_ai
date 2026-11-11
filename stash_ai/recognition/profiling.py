"""Opt-in bounded rebuild measurements. Never retain SQL, IDs or embeddings."""
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import time
from threading import Lock, get_ident

_active = ContextVar('recognition_rebuild_profile', default=None)
_tables = ('candidates', 'observations', 'members', 'people', 'scans',
           'merge_support', 'identity_merges', 'origins', 'sample_keys')


def count(name, amount=1):
    metrics = _active.get()
    if metrics is not None:
        metrics['counts'][name] += amount
    else:
        pipeline_count(name, amount)


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


def timed_stream(name, values):
    """Time coordinator/worker execution up to each yield, excluding its caller."""
    metrics=_active.get()
    if metrics is None:
        yield from values
        return
    row=metrics['stages'].setdefault(name,dict(calls=0,errors=0,seconds=0.))
    iterator=iter(values)
    while True:
        start=time.perf_counter()
        try:
            value=next(iterator)
        except StopIteration:
            return
        except BaseException:
            row['errors']+=1
            raise
        finally:
            row['seconds']+=time.perf_counter()-start
        row['calls']+=1
        yield value


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


class _Pipeline:
    def __init__(self, cuda_sample_every=0):
        self.started = time.perf_counter()
        self.owner = get_ident()
        self.lock = Lock()
        self.closed = False
        self.stages, self.worker_stages, self.counts = {}, {}, {}
        self.cuda_sample_every=cuda_sample_every
        self.cuda_stages={};self.cuda_pending=[];self.cuda_dropped=0

    def record(self, name, seconds, failed):
        with self.lock:
            if self.closed:
                return
            stages = self.stages if get_ident() == self.owner else self.worker_stages
            row = stages.setdefault(name, dict(calls=0, errors=0, seconds=0.))
            row['calls'] += 1
            row['errors'] += int(failed)
            row['seconds'] += seconds


@contextmanager
def pipeline_scope(enabled, *, cuda_sample_every=0):
    if not enabled:
        yield
        return
    state = _Pipeline(cuda_sample_every)
    token = _pipeline.set(state)
    try:
        yield
    finally:
        with state.lock:
            state.closed = True
        _pipeline.reset(token)


def pipeline_bind(function):
    """Capture only telemetry for an owned thread task, never DB/model contexts."""
    state = _pipeline.get()
    if state is None:
        return function

    @wraps(function)
    def call(*args, **kwargs):
        token = _pipeline.set(state)
        try:
            return function(*args, **kwargs)
        finally:
            _pipeline.reset(token)
    return call


def pipeline_count(name, amount=1):
    state = _pipeline.get()
    if state is not None:
        with state.lock:
            if not state.closed:
                state.counts[name] = state.counts.get(name, 0) + amount


def pipeline_max(name, value):
    state = _pipeline.get()
    if state is not None:
        with state.lock:
            if not state.closed:
                state.counts[name] = max(state.counts.get(name, 0), value)


def pipeline_snapshot():
    state = _pipeline.get()
    if state is None:
        return None
    gpu=_cuda_snapshot(state) if state.cuda_sample_every else {}
    with state.lock:
        return dict(format=2, elapsed_seconds=time.perf_counter()-state.started,
            scope='host elapsed time; nested stages overlap; no forced CUDA synchronization',
            stages={key: dict(value) for key, value in state.stages.items()},
            worker_stages={key: dict(value) for key, value in state.worker_stages.items()},
            worker_scope='summed background task elapsed time; overlaps coordinator and other workers',
            counts=dict(state.counts), **gpu)


@contextmanager
def pipeline_timing(name):
    state = _pipeline.get()
    if state is None:
        yield
        return
    started, failed = time.perf_counter(), False
    try:
        yield
    except (StopIteration, StopAsyncIteration):
        raise  # Normal iterator exhaustion is not an operational failure.
    except BaseException:
        failed = True
        raise
    finally:
        state.record(name, time.perf_counter()-started, failed)


def pipeline_stage(name):
    def decorate(function):
        @wraps(function)
        def call(*args, **kwargs):
            if _pipeline.get() is None:
                return function(*args, **kwargs)
            with pipeline_timing(name):
                return function(*args, **kwargs)
        return call
    return decorate


@contextmanager
def pipeline_cuda_timing(name,device):
    """Optional bounded CUDA event sampling, with no synchronize calls."""
    state=_pipeline.get()
    if state is None or not state.cuda_sample_every or str(device).split(':')[0]!='cuda':
        yield
        return
    import torch
    with state.lock:
        row=state.cuda_stages.setdefault(name,dict(forwards=0,samples=0,seconds=0.,
            allocated_bytes_max=0,reserved_bytes_max=0))
        row['forwards']+=1
        sampled=(row['forwards']-1)%state.cuda_sample_every==0
    if not sampled:
        yield
        return
    _cuda_snapshot(state)
    with state.lock:
        full=len(state.cuda_pending)>=64
        if full:state.cuda_dropped+=1
    if full:
        yield
        return
    with torch.cuda.device(device):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        allocated=torch.cuda.memory_allocated(device);reserved=torch.cuda.memory_reserved(device)
    with state.lock:
        row['allocated_bytes_max']=max(row['allocated_bytes_max'],allocated)
        row['reserved_bytes_max']=max(row['reserved_bytes_max'],reserved)
        state.cuda_pending.append((name,start,end))


def _cuda_snapshot(state):
    with state.lock:
        pending=[]
        for name,start,end in state.cuda_pending:
            if end.query():
                row=state.cuda_stages[name]
                row['samples']+=1
                row['seconds']+=start.elapsed_time(end)/1000.
            else:pending.append((name,start,end))
        state.cuda_pending=pending
        return dict(cuda=dict(sample_every=state.cuda_sample_every,
            scope='completed sampled CUDA events; stage intervals can overlap; VRAM sampled after forward',
            stages={key:dict(row) for key,row in state.cuda_stages.items()},
            pending_samples=len(pending),dropped_samples=state.cuda_dropped))
