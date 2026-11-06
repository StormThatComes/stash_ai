"""Replay-owned vector pages; keep addresses, not the whole library, in RAM."""
from collections.abc import Sequence
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import sqlite3
import threading
import numpy as np

_active=ContextVar('recognition_paged_references',default=None)


class Samples(Sequence):
    def __init__(self,pool,keys,bytes,shape=None):
        self.pool,self.keys,self.nbytes,self.shape=pool,tuple(keys),bytes,shape
    def __len__(self):return len(self.keys)
    def _read(self,keys):
        # LEFT JOIN preserves order and duplicates and makes corrupt pages explicit.
        with self.pool.lock:
            rows=self.pool.db.execute('SELECT v.value FROM json_each(?) k LEFT JOIN vectors v '
                'ON v.id=k.value ORDER BY CAST(k.key AS INTEGER)',(json.dumps(keys),)).fetchall()
        for (raw,) in rows:
            if raw is None:raise ValueError('Recognition reference page is missing.')
            yield np.frombuffer(raw,dtype='<f4')
    def __iter__(self):
        for start in range(0,len(self.keys),256):
            yield from self._read(self.keys[start:start+256])
    def __getitem__(self,index):
        if isinstance(index,slice):
            keys=self.keys[index]
            return tuple(v for start in range(0,len(keys),256) for v in self._read(keys[start:start+256]))
        return next(self._read((self.keys[index],)))


class Pool:
    def __init__(self,config):
        from .memory_size import Reservation
        from .run_cache import current
        cache=current();limit=cache.working if cache and cache.working is not None else config.data['execution']['rebuild_memory_bytes']
        pages=min(config.data['execution']['rebuild_sqlite_cache_bytes'],max(16384,limit//32))
        self.reservation=Reservation(limit,label='reference-pages');self.reservation.set(pages*2)
        from .scratch_space import connect
        self.db=connect(check_same_thread=False);self.lock=threading.Lock()
        self.db.execute('PRAGMA cache_size=-'+str(pages//1024))
        self.db.execute('PRAGMA temp_store=FILE')
        self.db.execute('CREATE TABLE vectors(id TEXT PRIMARY KEY,value BLOB)')
        self.db.execute('CREATE TABLE cues(group_id TEXT,scan TEXT,cue INTEGER,PRIMARY KEY(group_id,scan,cue))')
        self.db.execute('CREATE TABLE records(group_id TEXT,ordinal INTEGER,payload TEXT,PRIMARY KEY(group_id,ordinal))')
        self.limit=config.data['execution']['replay_snapshot_bytes'];self.used=0
    def pack(self,vectors):
        if isinstance(vectors,Samples):return vectors
        keys=[];total=0;shapes=set()
        for vector in vectors:
            array=np.asarray(vector,dtype='<f4');raw=array.tobytes();key=hashlib.sha256(raw).hexdigest()
            shapes.add((array.size,))
            inserted=self.db.execute('INSERT OR IGNORE INTO vectors VALUES (?,?)',(key,raw)).rowcount
            if inserted:self.used+=len(raw)+128
            if self.used>self.limit:
                from .resource_errors import ResourceLimit
                raise ResourceLimit('Reference pages exceed execution.replay_snapshot_bytes.')
            keys.append(key);total+=len(raw)
        return Samples(self,keys,total,next(iter(shapes)) if len(shapes)==1 else None)
    def close(self):
        try:self.db.close()
        finally:self.reservation.set(0)


def pack(vectors):
    pool=_active.get()
    return pool.pack(vectors) if pool is not None else tuple(vectors)


@contextmanager
def scope(config):
    if _active.get() is not None or config.data['execution']['rebuild_storage']!='disk':
        yield
        return
    pool=Pool(config);token=_active.set(pool)
    try:yield
    finally:
        from stash_ai.runtime.cleanup import cleanup
        _active.reset(token)
        cleanup([('close reference pages',pool.close)])


class Rows(Sequence):
    """Stable ordered metadata pages, including every original veto witness."""
    def __init__(self,pool,group,count):self.pool,self.group,self.count=pool,group,count
    def __len__(self):return self.count
    def __iter__(self):
        import json
        for (value,) in self.pool.db.execute('SELECT payload FROM records WHERE group_id=? ORDER BY ordinal',(self.group,)):
            yield json.loads(value)
    def __getitem__(self,index):
        import json
        if isinstance(index,slice):return [self[i] for i in range(*index.indices(len(self)))]
        if index<0:index+=self.count
        row=self.pool.db.execute('SELECT payload FROM records WHERE group_id=? AND ordinal=?',(self.group,index)).fetchone()
        if row is None:raise IndexError(index)
        return json.loads(row[0])
    def __add__(self,other):
        from itertools import chain
        return chain(self,other)


def rows(values):
    pool=_active.get()
    if pool is None:return values
    if isinstance(values,Rows):return values
    import uuid
    from .database import encoded
    group=uuid.uuid4().hex;count=0
    for count,row in enumerate(values,1):
        value=encoded(dict(row) if hasattr(row,'keys') else row);pool.used+=len(value)+128
        if pool.used>pool.limit:
            from .resource_errors import ResourceLimit
            raise ResourceLimit('Reference metadata pages exceed execution.replay_snapshot_bytes.')
        pool.db.execute('INSERT INTO records VALUES (?,?,?)',(group,count-1,value))
    return Rows(pool,group,count)


class Cues:
    def __init__(self,pool):
        import uuid
        self.pool,self.group=pool,uuid.uuid4().hex
    def add(self,value):
        inserted=self.pool.db.execute('INSERT OR IGNORE INTO cues VALUES (?,?,?)',(self.group,*value)).rowcount
        if inserted:self.pool.used+=len(str(value[0]))+128
        if self.pool.used>self.pool.limit:
            from .resource_errors import ResourceLimit
            raise ResourceLimit('Reference cue pages exceed execution.replay_snapshot_bytes.')
    def __contains__(self,value):
        return self.pool.db.execute('SELECT 1 FROM cues WHERE group_id=? AND scan=? AND cue=?',(self.group,*value)).fetchone() is not None
    def __iter__(self):
        return iter(self.pool.db.execute('SELECT scan,cue FROM cues WHERE group_id=?',(self.group,)))
    def __len__(self):return self.pool.db.execute('SELECT COUNT(*) FROM cues WHERE group_id=?',(self.group,)).fetchone()[0]
    def isdisjoint(self,other):
        if isinstance(other,Cues) and other.pool is self.pool:
            return self.pool.db.execute('SELECT 1 FROM cues a JOIN cues b ON b.scan=a.scan AND b.cue=a.cue '
                'WHERE a.group_id=? AND b.group_id=? LIMIT 1',(self.group,other.group)).fetchone() is None
        return not any(value in self for value in other)


def cues():
    pool=_active.get()
    return Cues(pool) if pool is not None else set()
