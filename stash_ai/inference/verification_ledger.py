"""Disk-backed run verification with a 64 KiB SQLite page cache and hashed keys."""
import hashlib
import sqlite3
from threading import Lock


class VerificationLedger:
    """Temporary verification state; no media bytes or plaintext URLs are stored."""
    def __init__(self):
        self._lock = Lock()
        self._db = sqlite3.connect('', check_same_thread=False, isolation_level=None)
        self._db.execute('PRAGMA cache_size=-64')
        self._db.execute('PRAGMA mmap_size=0')
        self._db.execute('CREATE TABLE verification (key BLOB PRIMARY KEY, checksum TEXT NOT NULL) WITHOUT ROWID')

    @staticmethod
    def _key(key):
        return hashlib.sha256(key.encode()).digest()

    def __getitem__(self, key):
        with self._lock:
            row = self._db.execute('SELECT checksum FROM verification WHERE key=?', (self._key(key),)).fetchone() if self._db is not None else None
        if row is None:
            raise KeyError(key)
        return row[0]

    def __contains__(self, key):
        try:
            self[key]
        except KeyError:
            return False
        return True

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __setitem__(self, key, checksum):
        with self._lock:
            if self._db is None:
                raise RuntimeError('Verification ledger is closed.')
            self._db.execute('INSERT OR REPLACE INTO verification VALUES (?,?)', (self._key(key), checksum))

    def __len__(self):
        with self._lock:
            return self._db.execute('SELECT COUNT(*) FROM verification').fetchone()[0] if self._db is not None else 0

    def close(self):
        with self._lock:
            db, self._db = self._db, None
            if db is not None:
                db.close()
