"""Per-volume accounting for durable evidence, caches and disposable stores."""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import shutil
import time
import threading

from .resource_errors import ResourceLimit

_active = ContextVar('recognition_disk_budget',default=None)


class Budget:
    def __init__(self,config):
        self.database=Path(config.database)
        self.limit=config.data['execution'].get('disk_total_bytes',0)
        self.headroom=config.data['execution'].get('disk_free_bytes',0)
        self.roots=set()
        self.volumes={}
        self.checked=0.0
        self.lock=threading.RLock()
        self.reservations={}

    def check(self, growth=0, path=None, *, throttled=False):
        with self.lock:
            return self._check(growth,path,throttled=throttled)

    def _check(self, growth=0, path=None, *, throttled=False):
        if throttled and time.monotonic()-self.checked<1:return self.volumes
        if not self.limit and not self.headroom:return {}
        parent=self.database.parent
        roots=[p for p in parent.glob(self.database.name+'*') if not p.is_symlink()]
        roots.extend(self.roots)
        volumes={};seen=set()
        for root in roots:
            files=root.rglob('*') if root.is_dir() else (root,)
            for item in files:
                if item.is_symlink() or not item.is_file():continue
                stat=item.stat();key=(stat.st_dev,stat.st_ino)
                if key in seen:continue
                seen.add(key)
                row=volumes.setdefault(stat.st_dev,dict(path=str(item.parent),used=0))
                row['used']+=stat.st_size
        target=Path(path or parent)
        while not target.exists():target=target.parent
        device=target.stat().st_dev
        volumes.setdefault(device,dict(path=str(target),used=0))
        for dev,row in volumes.items():
            held=sum(max(0,cap-p.stat().st_size) for p,(vol,cap) in self.reservations.items() if vol==dev)
            requested=row['used']+held+(growth if dev==device else 0)
            free=shutil.disk_usage(row['path']).free
            row.update(free=free,requested=requested)
            if self.limit and requested>self.limit:
                raise ResourceLimit('Recognition data, caches and scratch exceed the per-volume disk budget.',
                    budget='execution.rebuild.disk_total_bytes',limit=self.limit,requested=requested)
            required=self.headroom+held+(growth if dev==device else 0)
            if free<required:
                raise ResourceLimit('Recognition disk free-space reserve would be exhausted.',
                    budget='execution.rebuild.disk_free_bytes',limit=free,requested=required)
        self.volumes=volumes
        self.checked=time.monotonic()
        return volumes


def check(growth=0,path=None,*,throttled=False):
    budget=_active.get()
    return budget.check(growth,path,throttled=throttled) if budget is not None else {}


@contextmanager
def scope(config):
    if _active.get() is not None:
        yield _active.get();return
    budget=Budget(config);token=_active.set(budget)
    try:
        budget.check();yield budget
    finally:_active.reset(token)


@contextmanager
def temporary(path):
    budget=_active.get();path=Path(path)
    if budget is not None:budget.roots.add(path)
    try:yield
    finally:
        if budget is not None:budget.roots.discard(path)

@contextmanager
def reserve_file(path, maximum):
    """Admit a private SQLite file once; max_page_count enforces the grant."""
    budget=_active.get()
    if budget is None or not (budget.limit or budget.headroom):
        yield maximum
        return
    path=Path(path)
    with budget.lock:
        dev=path.stat().st_dev
        volume=budget.check(path=path)[dev]
        held=volume['requested']-volume['used']
        existing=path.stat().st_size
        cap=min(maximum,existing+volume['free']-budget.headroom-held)
        if budget.limit:cap=min(cap,existing+budget.limit-volume['requested'])
        cap=max(0,cap)
        budget.reservations[path]=(dev,cap)
    try:yield cap
    finally:
        with budget.lock:budget.reservations.pop(path,None)
