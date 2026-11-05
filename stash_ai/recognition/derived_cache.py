"""Bounded, disposable replay results. Writes share the rebuild transaction."""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json

from .database import encoded
from .profiling import count
from .rebuild_cache import _hash_row

_active = ContextVar("recognition_derived_cache", default=None)
PREFIX = "recognition-derived-v1:"


def fingerprint(db, queries, seed):
    digest = hashlib.sha256(encoded(seed).encode())
    for statement, parameters in queries:
        digest.update(b"\xffquery\x00")
        for row in db.execute(statement, parameters):
            _hash_row(digest, row)
    return digest.hexdigest()


class Store:
    def __init__(self, db, budget):
        self.db = db
        self.budget = budget
        self.used = db.execute("SELECT COALESCE(SUM(length(CAST(value AS BLOB))+length(key)+128),0) "
                               "FROM metadata WHERE key GLOB ?", (PREFIX+"*",)).fetchone()[0]

        if self.used > self.budget:
            self.trim(self.budget*3//4)

    def trim(self, target):
        while self.used>target:
            rows=self.db.execute("SELECT key,length(CAST(value AS BLOB))+length(key)+128 FROM metadata "
                                 "WHERE key GLOB ? ORDER BY rowid LIMIT 256",(PREFIX+"*",)).fetchall()
            if not rows:break
            self.db.executemany("DELETE FROM metadata WHERE key=?",((key,) for key,_ in rows))
            self.used-=sum(charge for _,charge in rows)

    def get(self, kind, version):
        key = PREFIX+kind+":"+version
        row = self.db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        if row is None:
            count(kind+"_cache_misses")
            return None
        checksum, _, payload = row[0].partition(" ")
        if hashlib.sha256((key+payload).encode()).hexdigest() != checksum:
            return None
        count(kind+"_cache_hits")
        return json.loads(payload)

    def put(self, kind, version, value):
        key = PREFIX+kind+":"+version
        payload = encoded(value)
        record = hashlib.sha256((key+payload).encode()).hexdigest()+" "+payload
        size = len(record.encode())+len(key)+128
        if size > min(self.budget//8, 4*1024**2):
            return
        if self.used+size > self.budget:
            self.trim(self.budget*3//4-size)
        change = self.db.execute("INSERT OR IGNORE INTO metadata VALUES (?,?)", (key,record))
        if change.rowcount:
            self.used += size


@contextmanager
def scope(database, config):
    from .database import FaceDatabase
    from .solver_execution import current as solver
    budget = min(64*1024**2, config.data["execution"]["rebuild_cache_bytes"]//8)
    cache = Store(database.connection,budget) if budget and isinstance(database,FaceDatabase) and solver() is None else None
    token = _active.set(cache)
    try:
        yield
    finally:
        _active.reset(token)


def current():
    return _active.get()

