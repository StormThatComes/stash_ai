"""Disposable stage cache: exact inputs, versioned contracts, bounded disk pages."""
from contextlib import contextmanager
from contextvars import ContextVar
from collections import OrderedDict
import base64
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
from functools import lru_cache

_active=ContextVar('recognition_inference_cache',default=None)


@lru_cache(maxsize=1)
def runtime_versions():
    from importlib.metadata import version,PackageNotFoundError
    result=[]
    for name in ('numpy','Pillow','opencv-python','opencv-python-headless','torch','torchvision','onnxruntime','onnxruntime-gpu','ultralytics','transformers'):
        try:result.append((name,version(name)))
        except PackageNotFoundError:pass
    return result


def current():return _active.get()


@contextmanager
def using(store):
    token=_active.set(store)
    try:yield
    finally:_active.reset(token)


def fingerprint(value):
    """Hash exact pixels/tensors and their shape, never cue IDs or timestamps."""
    import numpy as np
    h=hashlib.sha256()
    def add(v):
        if hasattr(v,'mode') and hasattr(v,'tobytes'):
            digest = getattr(v, '_recognition_pixel_digest', None)
            if digest is None:
                digest = hashlib.sha256(v.tobytes()).digest()
                if hasattr(v, '_recognition_pixel_digest'):
                    v._recognition_pixel_digest = digest
            h.update(str((v.mode,v.size)).encode())
            h.update(digest)
        elif isinstance(v,np.ndarray):
            h.update(str((v.dtype.str,v.shape)).encode());h.update(v.tobytes())
        elif hasattr(v,'detach'):
            add(v.detach().cpu().numpy())
        elif isinstance(v,(list,tuple)):
            h.update(b'[')
            for item in v:add(item);h.update(b',')
            h.update(b']')
        else:h.update(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
    add(value);return h.hexdigest()


def pack(value):
    import numpy as np
    raw=bytearray()
    def encode(v):
        if isinstance(v,np.ndarray):
            if v.dtype.hasobject:raise ValueError('Object arrays cannot be cached.')
            offset=len(raw);raw.extend(v.tobytes())
            return {'nd':[v.dtype.str,list(v.shape),offset,v.nbytes]}
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,tuple):return {'tuple':[encode(x) for x in v]}
        if isinstance(v,list):return [encode(x) for x in v]
        if isinstance(v,dict):return {k:encode(x) for k,x in v.items()}
        return v
    header=json.dumps(encode(value),separators=(',',':'),allow_nan=False).encode()
    return b'RIC2'+len(header).to_bytes(4,'big')+header+raw


def unpack(blob):
    import numpy as np
    binary=blob.startswith(b'RIC2')
    if binary:
        size=int.from_bytes(blob[4:8],'big')
        if len(blob)<8+size:raise ValueError('Truncated cache header.')
        data=memoryview(blob)[8+size:];blob=blob[8:8+size]
    def decode(v):
        if isinstance(v,list):return [decode(x) for x in v]
        if isinstance(v,dict):
            if binary and set(v)=={'nd'}:
                dtype,shape,start,size=v['nd'];dtype=np.dtype(dtype)
                if dtype.hasobject or start<0 or size<0 or start+size>len(data):
                    raise ValueError('Invalid cached array extent.')
                return np.frombuffer(data[start:start+size],dtype=dtype).reshape(shape).copy()
            if set(v)=={'array'}:
                dtype,shape,raw=v['array'];dtype=np.dtype(dtype)
                if dtype.hasobject:raise ValueError('Object arrays cannot be cached.')
                return np.frombuffer(base64.b64decode(raw,validate=True),dtype=dtype).reshape(shape).copy()
            if set(v)=={'tuple'}:return tuple(decode(x) for x in v['tuple'])
            return {k:decode(x) for k,x in v.items()}
        return v
    return decode(json.loads(blob))


class Store:
    def __init__(self,path,limit,*,queue_bytes=0):
        from concurrent.futures import ThreadPoolExecutor
        self.queue_limit = queue_bytes
        self.writer = ThreadPoolExecutor(1, thread_name_prefix="recognition-cache") if queue_bytes else None
        self.queued = OrderedDict()
        self.queued_bytes = 0
        self.queue_lock = threading.RLock()
        self.lock=threading.RLock();self.limit=limit;self.pending=0
        self.db=sqlite3.connect(path,check_same_thread=False)
        self.db.execute('PRAGMA cache_size=-1024')
        self.db.execute('PRAGMA journal_mode=TRUNCATE')
        self.db.execute(f'PRAGMA max_page_count={max(8,limit//4096)}')
        self.db.execute('CREATE TABLE IF NOT EXISTS entries(key TEXT PRIMARY KEY,payload BLOB,checksum TEXT,size INTEGER,sequence INTEGER)')
        self.db.execute('CREATE INDEX IF NOT EXISTS oldest ON entries(sequence)')
        self.used,self.sequence=self.db.execute('SELECT COALESCE(SUM(size),0),COALESCE(MAX(sequence),0) FROM entries').fetchone()
        if 'stage' not in [r[1] for r in self.db.execute('PRAGMA table_info(entries)')]:
            self.db.execute("ALTER TABLE entries ADD COLUMN stage TEXT NOT NULL DEFAULT 'legacy'")
        self.db.execute('CREATE INDEX IF NOT EXISTS stage_age ON entries(stage,sequence)')
        self.by_stage=dict(self.db.execute('SELECT stage,SUM(size) FROM entries GROUP BY stage'))
        self.prune(0);self.db.commit()
    def prune(self,required):
        while self.used+required>self.limit*0.75:
            rows=self.db.execute('SELECT key,size,stage FROM entries ORDER BY sequence LIMIT 64').fetchall()
            if not rows:break
            self.db.executemany('DELETE FROM entries WHERE key=?',((key,) for key,_,_ in rows))
            self.used-=sum(size for _,size,_ in rows)
            for _,size,stage in rows:self.by_stage[stage]-=size
    def get(self, key):
        return self.get_many([key]).get(key)

    def get_many(self, keys):
        result = {}
        if self.writer is not None:
            with self.queue_lock:
                for key in keys:
                    if key in self.queued:
                        self._retire(key)
        for start in range(0, len(keys), 32):
            group = keys[start:start+32]
            with self.lock:
                rows = list(self.db.execute('SELECT key,payload,checksum FROM entries WHERE key IN '
                                            '(SELECT value FROM json_each(?))', (json.dumps(group),)))
            for key, blob, checksum in rows:
                if hashlib.sha256(blob).hexdigest() != checksum:
                    continue
                try:
                    result[key] = unpack(blob)
                except (ValueError, TypeError, KeyError, OverflowError):
                    continue
        return result

    def put(self,key,value,stage='legacy'):
        blob=pack(value);size=len(blob)+512
        if size>min(4*1024**2,self.limit//4):return
        with self.lock:
            # Leave space for both geometry and the larger embedding stages.
            if stage!='legacy':
                fraction=.16 if 'embed' in stage or 'encoder' in stage else .10
                quota=int(self.limit*fraction)
                if size>quota:return
                while self.by_stage.get(stage,0)+size>quota:
                    rows=self.db.execute('SELECT key,size FROM entries WHERE stage=? ORDER BY sequence LIMIT 32',(stage,)).fetchall()
                    if not rows:break
                    self.db.executemany('DELETE FROM entries WHERE key=?',[(k,) for k,_ in rows])
                    freed=sum(n for _,n in rows);self.by_stage[stage]-=freed;self.used-=freed
            self.prune(size)
            if self.pending%128==0:
                from .disk_budget import check
                check(min(self.limit,2*1024**2),throttled=True)
            old=self.db.execute('SELECT size,stage FROM entries WHERE key=?',(key,)).fetchone()
            self.sequence+=1
            try:
                self.db.execute('INSERT OR REPLACE INTO entries VALUES (?,?,?,?,?,?)',
                    (key,blob,hashlib.sha256(blob).hexdigest(),size,self.sequence,stage))
            except sqlite3.OperationalError as error:
                if getattr(error,'sqlite_errorcode',None)!=sqlite3.SQLITE_FULL:raise
                # A cache capacity failure cannot invalidate collected evidence.
                self.db.rollback()
                self.used=self.db.execute('SELECT COALESCE(SUM(size),0) FROM entries').fetchone()[0]
                self.by_stage=dict(self.db.execute('SELECT stage,SUM(size) FROM entries GROUP BY stage'))
                return
            if old:self.by_stage[old[1]]-=old[0]
            self.by_stage[stage]=self.by_stage.get(stage,0)+size
            self.used+=size-(old[0] if old else 0);self.pending+=1
            if self.pending%128==0:self.db.commit()
    def _retire(self, key):
        future, charge = self.queued.pop(key)
        try:
            future.result()
        finally:
            self.queued_bytes -= charge

    def enqueue(self, key, value, stage):
        if self.writer is None:
            return self.put(key, value, stage)
        from contextvars import copy_context
        from copy import deepcopy
        from .memory_size import size
        charge = 3*size(value) + len(key) + 4096
        with self.queue_lock:
            if key in self.queued:
                return
            while self.queued and self.queued_bytes + charge > self.queue_limit:
                self._retire(next(iter(self.queued)))
            if charge > self.queue_limit:
                return self.put(key,value,stage)
            future = self.writer.submit(copy_context().run, self.put, key, deepcopy(value), stage)
            self.queued[key] = future, charge
            self.queued_bytes += charge

    def close(self):
        try:
            if self.writer is not None:
                self.writer.shutdown(wait=True)
                for key in tuple(self.queued):
                    self._retire(key)
        finally:
            with self.lock:
                try:self.db.commit()
                finally:self.db.close()


@contextmanager
def scope(config):
    limit=config.data['execution'].get('inference_cache_bytes',0)
    if current() is not None or limit<65536:
        yield current();return
    path=Path(str(config.database)+'.inference-cache.sqlite')
    path.parent.mkdir(parents=True,exist_ok=True)
    store=Store(path,limit,queue_bytes=min(limit//8,config.data['execution']['scan_queue_bytes']//8))
    try:
        with using(store):yield store
    finally:store.close()


def many(stage,contract,values,calculate,*,key=fingerprint):
    store=current()
    if store is None:return calculate(values)
    from .profiling import count
    prefix=fingerprint(['inference-cache-v2',runtime_versions(),stage,contract]);results={};missing=OrderedDict();keys=[]
    unique = {}
    for value in values:
        identity = prefix + ':' + key(value)
        keys.append(identity)
        unique.setdefault(identity, value)
    results = store.get_many(list(unique))
    for identity, value in unique.items():
        if identity not in results:
            missing[identity] = value
        else:
            count('inference_'+stage+'_hits')
    if missing:
        computed=calculate(list(missing.values()))
        if len(computed)!=len(missing):raise ValueError('Cached inference returned an incompatible batch.')
        for identity,value in zip(missing,computed):
            results[identity]=value;store.enqueue(identity,value,stage);count('inference_'+stage+'_misses')
    from copy import deepcopy
    if stage == 'body_embedding':
        from .models import certified_body
        return [v if certified_body(v) else deepcopy(v) for v in (results[k] for k in keys)]
    return [deepcopy(results[identity]) for identity in keys]
