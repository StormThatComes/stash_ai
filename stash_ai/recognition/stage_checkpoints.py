"""Durable pure plans; interrupted rebuilds never publish partial assignments."""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import hashlib
import json
import os
import sqlite3
import stat

from .database import encoded
from .config import digest
from .profiling import count

_active=ContextVar('recognition_stage_checkpoints',default=None)
APPLICATION_ID=0x5253504C


class Plans:
    def __init__(self,path,limit):
        self.path=Path(str(path)+'.stage-plans.sqlite3');self.limit=limit;self.db=None
    def open(self):
        if self.db is not None:return
        created=False
        try:
            descriptor=os.open(self.path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        except FileExistsError:pass
        else:os.close(descriptor);created=True
        info=self.path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1:
            raise ValueError('Stage checkpoint must be an ordinary owned file.')
        db=sqlite3.connect(self.path.resolve().as_uri()+'?mode=rw',uri=True)
        try:
            if created:
                db.execute('PRAGMA application_id='+str(APPLICATION_ID))
                db.execute('CREATE TABLE plans(key TEXT PRIMARY KEY,payload TEXT,checksum TEXT,bytes INTEGER)')
                db.commit()
            elif db.execute('PRAGMA application_id').fetchone()[0]!=APPLICATION_ID:
                raise ValueError('Refusing to overwrite a foreign stage checkpoint.')
            db.execute('PRAGMA cache_size=-256');db.execute('PRAGMA mmap_size=0')
            db.execute('PRAGMA max_page_count='+str(max(16,self.limit//4096)))
        except BaseException:db.close();raise
        self.db=db
    def get(self,kind,key):
        self.open();name=kind+':'+key
        row=self.db.execute('SELECT payload,checksum FROM plans WHERE key=?',(name,)).fetchone()
        if row is None:return None
        payload,checksum=row
        if not isinstance(payload,str) or len(payload)>self.limit//2:return None
        if hashlib.sha256((name+payload).encode()).hexdigest()!=checksum:return None
        try:value=json.loads(payload)
        except (TypeError,ValueError):return None
        count('stage_'+kind+'_resumed')
        return value
    def put(self,kind,key,value):
        payload=encoded(value);name=kind+':'+key;size=len(payload.encode())+len(name)+256
        if size>self.limit//2:return
        self.open()
        try:
            with self.db:
                self.db.execute('DELETE FROM plans WHERE key=?',(name,))
                used=self.db.execute('SELECT COALESCE(SUM(bytes),0) FROM plans').fetchone()[0]
                for old,charge in self.db.execute('SELECT key,bytes FROM plans ORDER BY rowid').fetchall():
                    if used+size<=self.limit*3//4:break
                    self.db.execute('DELETE FROM plans WHERE key=?',(old,));used-=charge
                self.db.execute('INSERT INTO plans VALUES (?,?,?,?)',
                    (name,payload,hashlib.sha256((name+payload).encode()).hexdigest(),size))
        except sqlite3.OperationalError as error:
            if getattr(error,'sqlite_errorcode',None)!=sqlite3.SQLITE_FULL:raise
            # Checkpoints are optional; admission failure never drops evidence.
            count('stage_checkpoint_capacity_misses')
    def close(self):
        if self.db is not None:self.db.close();self.db=None


def current():return _active.get()


@contextmanager
def scope(database,config):
    from .solver_execution import current as solver
    if _active.get() is not None or not config.data['execution']['rebuild_checkpoint_scenes'] or solver() is not None:
        yield;return
    plans=Plans(config.database,config.data['execution']['rebuild_checkpoint_bytes']);token=_active.set(plans)
    try:yield
    finally:
        from stash_ai.runtime.cleanup import cleanup
        _active.reset(token)
        cleanup([('close stage checkpoints',plans.close)])


def report(db,selected,config,kind,build):
    plans=current()
    if plans is None:return build()
    # Strong signatures include every decision dependency. Raw body versions
    # only memoize content hashes inside the authoritative transaction.
    from .hybrid_merge import signature
    from .dependencies import cached_many
    from .derived_cache import fingerprint
    _,face=signature(db,selected,config)
    scans=[r[0] for r in db.execute('SELECT id FROM scans ORDER BY id')]
    def body(scan):
        return fingerprint(db,[('SELECT id,scan,cue,seconds,contract,face,geometry,quality,embedding,rejected '
            'FROM body_observations WHERE scan=? ORDER BY id',(scan,))],'stage-body-v1')
    bodies=cached_many(db,'stage-body-input',scans,'v1',body,dependency='raw-body')
    derived=fingerprint(db,[(
        'SELECT observation,person,role,status,policy,proof FROM body_members ORDER BY observation',()),
        ('SELECT observation,person FROM body_origins ORDER BY observation',())],'stage-derived-v1')
    key=digest(['stage-plan-v1',kind,face,bodies,derived])
    value=plans.get(kind,key)
    if value is None:
        value=build();plans.put(kind,key,value)
    return value
