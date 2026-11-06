"""Run-owned scratch leases shared by private SQLite stores and vector files.

Leases bound allocated capacity, not cumulative writes. Persistent caches and
checkpoints have their own retention budgets; SQLite internal sort files are not
application-owned stores and are outside this budget.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import os
import shutil
import sqlite3
import tempfile
from .resource_errors import ResourceLimit

_active=ContextVar('recognition_scratch',default=None)
_BLOCK=4*1024**2


class Budget:
    def __init__(self,limit,directory):
        self.limit,self.directory=limit,directory
        self.leases={};self.writes={};self.used=0;self.peak=0
    def set(self,owner,size):
        total=self.used-self.leases.get(owner,0)+size
        growth=max(0,size-self.leases.get(owner,0))
        if total>self.limit:
            raise ResourceLimit('Recognition scratch allocations exceed execution.rebuild.scratch_bytes.',
                budget='execution.rebuild.scratch_bytes',limit=self.limit,requested=total)
        if growth and shutil.disk_usage(self.directory).free<growth:
            raise ResourceLimit('Insufficient free space in recognition scratch directory.')
        self.used=total;self.peak=max(self.peak,total)
        if size:self.leases[owner]=size
        else:self.leases.pop(owner,None)


@contextmanager
def scope(config):
    if _active.get() is not None:
        yield _active.get();return
    settings=config.data['execution']
    directory=settings.get('rebuild_scratch_directory') or None
    if directory:
        directory=Path(directory).expanduser()
        if not directory.is_absolute():directory=(config.source_path.parent if config.source_path else Path.cwd())/directory
        directory.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='recognition-scratch-',dir=directory) as path:
        budget=Budget(settings.get('rebuild_scratch_bytes',256*1024**3),path)
        token=_active.set(budget)
        try:yield budget
        finally:
            from .profiling import count
            count('scratch_reserved_peak_bytes',budget.peak)
            _active.reset(token)


def directory(**kwargs):
    budget=_active.get()
    return tempfile.TemporaryDirectory(dir=budget.directory if budget else None,**kwargs)


class File:
    def __init__(self,stream,budget):self.stream,self.budget,self.capacity=stream,budget,0
    def __getattr__(self,name):return getattr(self.stream,name)
    def write(self,value):
        needed=self.stream.tell()+len(value)
        if needed>self.capacity:
            capacity=max(needed,min(needed+262144,self.budget.limit-self.budget.used+self.capacity))
            self.budget.set(self,capacity);self.capacity=capacity
        return self.stream.write(value)
    def truncate(self,size=None):
        size=self.stream.tell() if size is None else size
        self.budget.set(self,size);self.capacity=size
        return self.stream.truncate(size)
    def close(self):
        try:self.stream.close()
        finally:self.budget.set(self,0)
    def __enter__(self):return self
    def __exit__(self,*error):self.close()


def file(**kwargs):
    budget=_active.get()
    stream=tempfile.TemporaryFile(dir=budget.directory if budget else None,**kwargs)
    return File(stream,budget) if budget else stream


class Connection(sqlite3.Connection):
    """Grow private database capacity in leases, avoiding per-row page queries."""
    def __init__(self,*args,**kwargs):
        import weakref
        super().__init__(*args,**kwargs)
        self.scratch_cursors=weakref.WeakSet()
        self.budget=_active.get();self.capacity=0;self.estimate=0;self.page_limit=None
        self.page=super().execute('PRAGMA page_size').fetchone()[0]
    def _admit(self,sql,parameters,overhead=4096):
        if not self.budget or sql.lstrip().split(None,1)[0].upper() not in ('INSERT','REPLACE','UPDATE','CREATE','ALTER'):
            return False
        # Unknown INSERT SELECT growth receives all currently unleased space.
        bulk='SELECT' in sql.upper() and sql.lstrip().upper().startswith(('INSERT','REPLACE'))
        values=parameters.values() if isinstance(parameters,dict) else parameters
        needed=overhead+4*sum(len(v) if isinstance(v,(bytes,str,memoryview)) else 32 for v in values)
        if bulk or self.estimate+needed>self.capacity:
            actual=super().execute('PRAGMA page_count').fetchone()[0]*self.page
            available=(self.budget.limit-self.budget.used)//2+self.capacity
            capacity=min(available,actual+shutil.disk_usage(self.budget.directory).free//4) if bulk else min(available,max(actual+needed,actual+_BLOCK))
            if self.page_limit is not None:capacity=min(capacity,self.page_limit)
            self.budget.set(self,2*capacity)
            self.capacity=capacity;self.estimate=actual
            super().execute('PRAGMA max_page_count='+str(capacity//self.page))
            if capacity<actual+needed and not bulk:
                raise ResourceLimit('Recognition scratch cannot admit the next database batch.',
                    budget='execution.rebuild.scratch_bytes',limit=self.budget.limit,requested=self.budget.used+needed)
        self.estimate+=needed
        return bulk
    def execute(self,sql,parameters=()):
        bulk=self._admit(sql,parameters)
        try:
            cursor=super().execute(sql,parameters);self.scratch_cursors.add(cursor);return cursor
        except sqlite3.OperationalError as error:
            if self.budget and getattr(error,'sqlite_errorcode',None)==sqlite3.SQLITE_FULL:
                raise ResourceLimit('Recognition scratch SQLite capacity exceeded.',budget='execution.rebuild.scratch_bytes',limit=self.budget.limit) from error
            raise
        finally:
            if bulk:
                actual=super().execute('PRAGMA page_count').fetchone()[0]*self.page
                self.budget.set(self,2*actual);self.capacity=self.estimate=actual
    def executemany(self,sql,parameters):
        from itertools import islice,chain
        bulk=False
        def admitted():
            nonlocal bulk
            iterator=iter(parameters)
            while batch:=list(islice(iterator,256)):
                values=chain.from_iterable(row.values() if isinstance(row,dict) else row for row in batch)
                bulk=self._admit(sql,values,4096*len(batch)) or bulk
                yield from batch
        try:
            cursor=super().executemany(sql,admitted());self.scratch_cursors.add(cursor);return cursor
        finally:
            if bulk:
                actual=super().execute('PRAGMA page_count').fetchone()[0]*self.page
                self.budget.set(self,2*actual);self.capacity=self.estimate=actual
    def executescript(self,sql):
        self._admit('CREATE script',(sql,))
        cursor=super().executescript(sql);self.scratch_cursors.add(cursor);return cursor
    def cursor(self,*args,**kwargs):
        cursor=super().cursor(*args,**kwargs);self.scratch_cursors.add(cursor);return cursor
    def close(self):
        for cursor in tuple(self.scratch_cursors):cursor.close()
        self.scratch_cursors.clear()
        try:super().close()
        finally:
            path=getattr(self,'owned_path',None)
            if path is not None:
                for suffix in ('','-journal','-wal','-shm'):Path(path+suffix).unlink(missing_ok=True)
            if self.budget:self.budget.set(self,0)


def connect(**kwargs):
    budget=_active.get()
    if not budget:return sqlite3.connect('',**kwargs)
    descriptor,path=tempfile.mkstemp(suffix='.sqlite3',dir=budget.directory);os.close(descriptor)
    db=sqlite3.connect(path,factory=Connection,**kwargs)
    db.owned_path=path
    return db


def grow(path,size):
    budget=_active.get()
    if budget:
        key=str(path);needed=budget.writes.get(key,0)+size
        capacity=budget.leases.get(key,0)
        if needed>capacity:
            capacity=max(needed,min(needed+262144,budget.limit-budget.used+capacity))
            budget.set(key,capacity)
        budget.writes[key]=needed


def release_directory(path):
    budget=_active.get()
    if budget:
        prefix=str(path)+os.sep
        for key in tuple(budget.leases):
            if isinstance(key,str) and key.startswith(prefix):
                budget.set(key,0);budget.writes.pop(key,None)
