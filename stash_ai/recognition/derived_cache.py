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
    def __init__(self, db, budget, *, prefix=PREFIX):
        self.db = db
        self.prefix=prefix
        self.budget = budget
        self.hits=set()
        marker=self.prefix+"hybrid-endpoint:"
        self.previous_endpoints={row[0][len(marker):] for row in db.execute(
            "SELECT key FROM metadata WHERE key GLOB ?",(marker+"*",))}
        self.used = db.execute("SELECT COALESCE(SUM(length(CAST(value AS BLOB))+length(key)+128),0) "
                               "FROM metadata WHERE key GLOB ?", (self.prefix+"*",)).fetchone()[0]

        if self.used > self.budget:
            self.trim(self.budget*3//4)

    def trim(self, target):
        while self.used>target:
            rows=self.db.execute("SELECT key,length(CAST(value AS BLOB))+length(key)+128 FROM metadata "
                                 "WHERE key GLOB ? ORDER BY rowid LIMIT 256",(self.prefix+"*",)).fetchall()
            if not rows:break
            victims=[]
            for key,charge in rows:
                victims.append((key,))
                self.used-=charge
                if self.used<=target:break
            self.db.executemany("DELETE FROM metadata WHERE key=?",victims)

    def get_bytes(self,kind,version):
        key=self.prefix+kind+":"+version
        row=self.db.execute('SELECT value FROM metadata WHERE key=?',(key,)).fetchone()
        if row is None:count(kind+'_cache_misses');return None
        value=row[0].encode() if isinstance(row[0],str) else row[0]
        checksum,_,payload=value.partition(b' ')
        if hashlib.sha256(key.encode()+payload).hexdigest().encode()!=checksum:return None
        self.hits.add(kind);count(kind+'_cache_hits')
        return payload

    def get(self,kind,version):
        raw=self.get_bytes(kind,version)
        if raw is None:return None
        try:return json.loads(raw)
        except (ValueError,TypeError):return None

    def put_bytes(self,kind,version,payload):
        key=self.prefix+kind+":"+version
        record=hashlib.sha256(key.encode()+payload).hexdigest().encode()+b' '+payload
        self._put(key,record)

    def put(self,kind,version,value):
        key=self.prefix+kind+":"+version;payload=encoded(value)
        record=hashlib.sha256((key+payload).encode()).hexdigest()+" "+payload
        self._put(key,record)

    def _put(self,key,record):
        length=lambda value:len(value.encode()) if isinstance(value,str) else len(value)
        size=length(record)+len(key)+128
        if size>min(self.budget//2,4*1024**2):return
        if self.used+size<=self.budget:
            inserted=self.db.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)',(key,record))
            if inserted.rowcount:self.used+=size;return
        old=self.db.execute('SELECT value FROM metadata WHERE key=?',(key,)).fetchone()
        if old is not None:
            if old[0]==record:return
            self.db.execute('DELETE FROM metadata WHERE key=?',(key,))
            self.used-=length(old[0])+len(key)+128
        if self.used+size>self.budget:self.trim(self.budget*3//4-size)
        change=self.db.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)',(key,record))
        if change.rowcount:self.used+=size


class Pools:
    """Independent disk quotas prevent appearance blobs from evicting plans."""
    def __init__(self,db,budget):
        self.budget=budget
        self.stores={name:Store(db,budget*share//4,prefix=PREFIX+name+':')
                     for name,share in (('centroids',1),('appearances',2),('plans',1))}
        self.previous_endpoints=self.stores['plans'].previous_endpoints
    @property
    def hits(self):return set().union(*(s.hits for s in self.stores.values()))
    def store(self,kind):
        return self.stores['centroids' if kind.startswith('body-owner-') else
                           'appearances' if kind.startswith('body-appearance') else 'plans']
    def capacity(self,kind):return self.store(kind).budget
    def get(self,kind,version):return self.store(kind).get(kind,version)
    def put(self,kind,version,value):return self.store(kind).put(kind,version,value)
    def get_bytes(self,kind,version):return self.store(kind).get_bytes(kind,version)
    def put_bytes(self,kind,version,value):return self.store(kind).put_bytes(kind,version,value)


@contextmanager
def scope(database, config):
    from .database import FaceDatabase
    from .solver_execution import current as solver
    budget = config.data["execution"]["derived_cache_bytes"] if config.data["execution"]["rebuild_cache_bytes"] else 0
    cache = Pools(database.connection,budget) if budget and isinstance(database,FaceDatabase) and solver() is None else None
    token = _active.set(cache)
    try:
        yield
    finally:
        _active.reset(token)


def current():
    return _active.get()
