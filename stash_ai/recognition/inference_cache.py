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
            h.update(str((v.mode,v.size)).encode());h.update(v.tobytes())
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
    def encode(v):
        if isinstance(v,np.ndarray):
            if v.dtype.hasobject:raise ValueError('Object arrays cannot be cached.')
            return {'array':[v.dtype.str,list(v.shape),base64.b64encode(v.tobytes()).decode()]}
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,tuple):return {'tuple':[encode(x) for x in v]}
        if isinstance(v,list):return [encode(x) for x in v]
        if isinstance(v,dict):return {k:encode(x) for k,x in v.items()}
        return v
    return json.dumps(encode(value),separators=(',',':'),allow_nan=False).encode()


def unpack(blob):
    import numpy as np
    def decode(v):
        if isinstance(v,list):return [decode(x) for x in v]
        if isinstance(v,dict):
            if set(v)=={'array'}:
                dtype,shape,raw=v['array'];dtype=np.dtype(dtype)
                if dtype.hasobject:raise ValueError('Object arrays cannot be cached.')
                return np.frombuffer(base64.b64decode(raw,validate=True),dtype=dtype).reshape(shape).copy()
            if set(v)=={'tuple'}:return tuple(decode(x) for x in v['tuple'])
            return {k:decode(x) for k,x in v.items()}
        return v
    return decode(json.loads(blob))


class Store:
    def __init__(self,path,limit):
        self.lock=threading.RLock();self.limit=limit;self.pending=0
        self.db=sqlite3.connect(path,check_same_thread=False)
        self.db.execute('PRAGMA cache_size=-1024')
        self.db.execute('PRAGMA journal_mode=TRUNCATE')
        self.db.execute(f'PRAGMA max_page_count={max(8,limit//4096)}')
        self.db.execute('CREATE TABLE IF NOT EXISTS entries(key TEXT PRIMARY KEY,payload BLOB,checksum TEXT,size INTEGER,sequence INTEGER)')
        self.db.execute('CREATE INDEX IF NOT EXISTS oldest ON entries(sequence)')
        self.used,self.sequence=self.db.execute('SELECT COALESCE(SUM(size),0),COALESCE(MAX(sequence),0) FROM entries').fetchone()
        self.prune(0);self.db.commit()
    def prune(self,required):
        while self.used+required>self.limit*0.75:
            rows=self.db.execute('SELECT key,size FROM entries ORDER BY sequence LIMIT 64').fetchall()
            if not rows:break
            self.db.executemany('DELETE FROM entries WHERE key=?',((key,) for key,_ in rows))
            self.used-=sum(size for _,size in rows)
    def get(self,key):
        with self.lock:
            row=self.db.execute('SELECT payload,checksum FROM entries WHERE key=?',(key,)).fetchone()
            if row is None:return None
            blob,checksum=row
            if hashlib.sha256(blob).hexdigest()!=checksum:return None
            try:return unpack(blob)
            except (ValueError,TypeError,KeyError,OverflowError):return None
    def put(self,key,value):
        blob=pack(value);size=len(blob)+512
        if size>min(4*1024**2,self.limit//4):return
        with self.lock:
            self.prune(size)
            if self.pending%128==0:
                from .disk_budget import check
                check(min(self.limit,2*1024**2),throttled=True)
            old=self.db.execute('SELECT size FROM entries WHERE key=?',(key,)).fetchone()
            self.sequence+=1
            try:
                self.db.execute('INSERT OR REPLACE INTO entries VALUES (?,?,?,?,?)',
                    (key,blob,hashlib.sha256(blob).hexdigest(),size,self.sequence))
            except sqlite3.OperationalError as error:
                if getattr(error,'sqlite_errorcode',None)!=sqlite3.SQLITE_FULL:raise
                # A cache capacity failure cannot invalidate collected evidence.
                self.db.rollback()
                self.used=self.db.execute('SELECT COALESCE(SUM(size),0) FROM entries').fetchone()[0]
                return
            self.used+=size-(old[0] if old else 0);self.pending+=1
            if self.pending%128==0:self.db.commit()
    def close(self):
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
    store=Store(path,limit)
    try:
        with using(store):yield store
    finally:store.close()


def many(stage,contract,values,calculate,*,key=fingerprint):
    store=current()
    if store is None:return calculate(values)
    from .profiling import count
    prefix=fingerprint(['inference-cache-v1',runtime_versions(),stage,contract]);results={};missing=OrderedDict();keys=[]
    for value in values:
        identity=prefix+':'+key(value);keys.append(identity)
        if identity in results or identity in missing:continue
        cached=store.get(identity)
        if cached is None:missing[identity]=value
        else:results[identity]=cached;count('inference_'+stage+'_hits')
    if missing:
        computed=calculate(list(missing.values()))
        if len(computed)!=len(missing):raise ValueError('Cached inference returned an incompatible batch.')
        for identity,value in zip(missing,computed):
            results[identity]=value;store.put(identity,value);count('inference_'+stage+'_misses')
    from copy import deepcopy
    return [deepcopy(results[identity]) for identity in keys]
