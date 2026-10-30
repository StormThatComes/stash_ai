"""Disposable content-addressed numerical arrays shared by trial processes.

Only numeric evidence is cached. No membership, thresholds or mutable decisions.
Atomic publication and a SQLite reservation ledger bound total published bytes.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile

_active = ContextVar('solver_shared_arrays', default=None)


def key(kind, arrays, extra=()):
    h = hashlib.sha256(json.dumps([kind, extra], sort_keys=True).encode())
    for array in arrays:
        if not array.flags.c_contiguous or array.dtype.hasobject:
            return None
        h.update(json.dumps([array.dtype.str, array.shape]).encode())
        h.update(memoryview(array).cast('B'))
    return h.hexdigest()


class Arrays:
    def __init__(self, directory, limit):
        self.directory, self.limit = Path(directory), limit
        self.db = sqlite3.connect(self.directory/'ledger.sqlite', timeout=.2)

    @classmethod
    def prepare(cls, directory):
        directory = Path(directory)
        directory.mkdir(exist_ok=True)
        with sqlite3.connect(directory/'ledger.sqlite') as db:
            db.execute('CREATE TABLE IF NOT EXISTS entries (key TEXT PRIMARY KEY, bytes INTEGER NOT NULL)')

    def get(self, signature):
        if signature is None:
            return None
        path = self.directory/(signature+'.npy')
        if not path.exists():
            return None
        import numpy as np
        from .profiling import count
        value = np.load(path, mmap_mode='r', allow_pickle=False)
        count('shared_numeric_hits')
        return value

    def put(self, signature, value):
        if signature is None or value.dtype.hasobject or value.nbytes + 512 > self.limit:
            return value
        # Avoid serializing every reader through the ledger on hits.
        found = self.get(signature)
        if found is not None:
            return found
        reserved = False
        try:
            self.db.execute('BEGIN IMMEDIATE')
            used = self.db.execute('SELECT COALESCE(SUM(bytes),0) FROM entries').fetchone()[0]
            exists = self.db.execute('SELECT 1 FROM entries WHERE key=?', (signature,)).fetchone()
            if not exists and used + value.nbytes + 512 <= self.limit:
                self.db.execute('INSERT INTO entries VALUES (?,?)', (signature, value.nbytes+512))
                reserved = True
            self.db.commit()
        except sqlite3.OperationalError:
            self.db.rollback()
            return value  # Cache contention changes cost only.
        if not reserved:
            return value
        filename = None
        try:
            import numpy as np
            from .profiling import count
            fd, filename = tempfile.mkstemp(dir=self.directory, suffix='.tmp')
            with os.fdopen(fd, 'wb') as stream:
                np.save(stream, value, allow_pickle=False)
            os.replace(filename, self.directory/(signature+'.npy'))
            filename = None
            count('shared_numeric_written_bytes', value.nbytes)
            return self.get(signature)
        except BaseException:
            with self.db:
                self.db.execute('DELETE FROM entries WHERE key=?', (signature,))
            raise
        finally:
            if filename is not None:
                Path(filename).unlink(missing_ok=True)

    def close(self):
        self.db.close()


def current():
    return _active.get()


@contextmanager
def scope(directory, limit):
    value = Arrays(directory, limit) if directory else None
    token = _active.set(value)
    try:
        yield value
    finally:
        _active.reset(token)
        if value:
            value.close()


def numerical(kind, arrays, calculate, extra=()):
    shared = current()
    if shared is None:
        return calculate()
    signature = key(kind, arrays, extra)
    value = shared.get(signature)
    return shared.put(signature, calculate()) if value is None else value
