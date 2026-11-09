"""Disposable content-addressed numerical arrays shared by trial processes.

Only numeric evidence is cached. No membership, thresholds or mutable decisions.
Atomic publication and a SQLite reservation ledger bound total published bytes.
"""
from contextlib import contextmanager
from contextlib import suppress
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
        if array.size:
            h.update(memoryview(array).cast('B'))
    return h.hexdigest()


class Arrays:
    def __init__(self, directory, limit):
        self.directory, self.limit = Path(directory), limit
        self.db = None
        self.disabled = False
        self.verified = {}
        self.pending = {}
        self.claims = {}
        try:
            self.db = sqlite3.connect(self.directory/'ledger.sqlite', timeout=.2)
        except sqlite3.Error:
            self.failed()

    def failed(self):
        from .profiling import count
        count('shared_numeric_errors')
        self.disabled = True

    def handle_error(self, error):
        if (getattr(error, 'sqlite_errorcode', 0) or 0) & 255 in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
            from .profiling import count
            count('shared_numeric_busy')
        else:
            self.failed()

    @classmethod
    def prepare(cls, directory):
        directory = Path(directory)
        try:
            directory.mkdir(exist_ok=True)
            with sqlite3.connect(directory/'ledger.sqlite') as db:
                db.execute('CREATE TABLE IF NOT EXISTS entries (key TEXT PRIMARY KEY, bytes INTEGER NOT NULL, checksum TEXT, shape TEXT, dtype TEXT)')
                columns = {r[1] for r in db.execute('PRAGMA table_info(entries)')}
                for column in ('checksum','shape','dtype'):
                    if column not in columns:db.execute(f'ALTER TABLE entries ADD COLUMN {column} TEXT')
                db.execute('CREATE TABLE IF NOT EXISTS usage (id INTEGER PRIMARY KEY CHECK(id=1), bytes INTEGER NOT NULL)')
                db.execute('INSERT OR IGNORE INTO usage SELECT 1,COALESCE(SUM(bytes),0) FROM entries')
                db.execute('CREATE TRIGGER IF NOT EXISTS reserve_bytes AFTER INSERT ON entries BEGIN UPDATE usage SET bytes=bytes+new.bytes WHERE id=1; END')
                db.execute('CREATE TRIGGER IF NOT EXISTS release_bytes AFTER DELETE ON entries BEGIN UPDATE usage SET bytes=bytes-old.bytes WHERE id=1; END')
            return True
        except (OSError, sqlite3.Error):
            from .profiling import count
            count('shared_numeric_errors')
            return False

    def flush_pending(self):
        if not self.pending or self.disabled:
            return
        try:
            with self.db:
                self.db.executemany('UPDATE entries SET checksum=?,shape=?,dtype=? WHERE key=? AND checksum IS NULL',
                                    self.pending.values())
        except sqlite3.Error as error:
            self.handle_error(error)
        else:
            self.pending.clear()

    def get(self, signature):
        if signature is None or self.disabled:
            return None
        self.flush_pending()
        if signature in self.pending or self.disabled:
            return None
        path = self.directory/(signature+'.npy')
        import numpy as np
        from .profiling import count
        try:
            row = self.db.execute('SELECT checksum,shape,dtype FROM entries WHERE key=?', (signature,)).fetchone()
            if row is None:
                if path.exists():self.failed()  # Orphaned publication has no integrity contract.
                return None
            if row[0] is None:
                return None  # Another worker's pending publication.
            stat = path.stat()
            stamp = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, tuple(row))
            if self.verified.get(signature) != stamp:
                with path.open('rb') as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() != row[0]:
                        raise ValueError('Shared numerical cache checksum mismatch.')
                if len(self.verified) >= 256:self.verified.clear()
                self.verified[signature] = stamp
            value = np.load(path, mmap_mode='r', allow_pickle=False)
            if list(value.shape) != json.loads(row[1]) or value.dtype.str != row[2]:
                raise ValueError('Shared numerical cache contract mismatch.')
        except FileNotFoundError:
            return None
        except (OSError, ValueError, EOFError, sqlite3.Error) as error:
            self.handle_error(error)
            return None
        count('shared_numeric_hits')
        return value

    def claim(self, signature):
        """One producer per key; never wait while holding another product claim."""
        from stash_ai.storage.writer_lock import acquire_writer_lock
        from stash_ai.runtime.cancellation import current
        import time
        if self.disabled or signature is None or signature in self.claims:
            return None
        deadline = time.monotonic() + 3
        while True:
            try:
                # Fixed stripes bound filesystem metadata across thousands of trials.
                lock = acquire_writer_lock(self.directory / ('compute-' + signature[:2]))
            except RuntimeError:
                value = self.get(signature)
                if value is not None or self.claims or time.monotonic() >= deadline:
                    return value
                token = current()
                if token is not None:
                    token.check()
                time.sleep(.02)
            except OSError:
                return None
            else:
                self.claims[signature] = lock
                value = self.get(signature)
                if value is not None:
                    self.release(signature)
                return value

    def release(self, signature):
        lock = self.claims.pop(signature, None)
        if lock is not None:
            lock.close()

    def put(self, signature, value):
        if self.disabled or signature is None or value.dtype.hasobject or value.nbytes + 512 > self.limit:
            return value
        # Avoid serializing every reader through the ledger on hits.
        found = self.get(signature)
        if found is not None:
            return found
        if self.disabled:
            return value
        reserved = False
        try:
            self.db.execute('BEGIN IMMEDIATE')
            used = self.db.execute('SELECT bytes FROM usage WHERE id=1').fetchone()[0]
            exists = self.db.execute('SELECT 1 FROM entries WHERE key=?', (signature,)).fetchone()
            if not exists and used + value.nbytes + 512 <= self.limit:
                self.db.execute('INSERT INTO entries(key,bytes) VALUES (?,?)', (signature, value.nbytes+512))
                reserved = True
            self.db.commit()
        except sqlite3.Error as error:
            with suppress(sqlite3.Error):
                self.db.rollback()
            self.handle_error(error)
            return value  # Cache contention changes cost only.
        if not reserved:
            return value
        filename = None
        published = False
        try:
            import numpy as np
            from .profiling import count
            fd, filename = tempfile.mkstemp(dir=self.directory, suffix='.tmp')
            with os.fdopen(fd, 'wb') as stream:
                np.save(stream, value, allow_pickle=False)
            with open(filename, 'rb') as stream:
                checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
            os.replace(filename, self.directory/(signature+'.npy'))
            published = True
            filename = None
            # Readers ignore reserved rows. A transient writer lock can defer
            # publication of the checksum without losing the byte reservation.
            self.pending[signature] = (checksum, json.dumps(list(value.shape)), value.dtype.str, signature)
            count('shared_numeric_written_bytes', value.nbytes)
            found = self.get(signature)
            return value if found is None else found
        except (OSError, ValueError, sqlite3.Error):
            self.failed()
            return value
        finally:
            if not published:
                with suppress(sqlite3.Error):
                    with self.db:
                        self.db.execute('DELETE FROM entries WHERE key=?', (signature,))
            if filename is not None:
                with suppress(OSError):
                    Path(filename).unlink(missing_ok=True)

    def close(self):
        for signature in tuple(self.claims):
            self.release(signature)
        if self.db is not None:
            try:
                self.flush_pending()
            finally:
                self.db.close()


def current():
    return _active.get()


@contextmanager
def scope(directory, limit):
    existing=current()
    if existing is not None and directory and existing.directory==Path(directory) and existing.limit==limit:
        yield existing
        return
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
    if value is None and sum(a.nbytes for a in arrays) >= 65536:
        value = shared.claim(signature)
    try:
        return shared.put(signature, calculate()) if value is None else value
    finally:
        shared.release(signature)
