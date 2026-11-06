"""Per-volume accounting for durable evidence, caches and disposable stores."""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import shutil
import time

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

    def check(self, growth=0, path=None, *, throttled=False):
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
            requested=row['used']+(growth if dev==device else 0)
            free=shutil.disk_usage(row['path']).free
            row.update(free=free,requested=requested)
            if self.limit and requested>self.limit:
                raise ResourceLimit('Recognition data, caches and scratch exceed the per-volume disk budget.',
                    budget='execution.rebuild.disk_total_bytes',limit=self.limit,requested=requested)
            required=self.headroom+(growth if dev==device else 0)
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
