"""Per-volume accounting for durable evidence, caches and disposable stores."""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import errno
import os
import stat
import shutil
import time
import threading

from .resource_errors import ResourceLimit

if os.name == 'nt':
    from .windows_volume import volume_key as _windows_volume
else:
    _windows_volume = None


def _retry_access_denied(read, *args):
    # Windows can temporarily deny reads while a peer removes its scratch
    # directory. Retry that read locally and propagate a lasting denial.
    for delay in (0.01, 0.05, 0.2):
        try:
            return read(*args)
        except PermissionError:
            time.sleep(delay)
    return read(*args)


def _device(path, info, parent_device=None):
    if _windows_volume is None:
        return info.st_dev
    # Ordinary children share their parent's volume. Reparse points may
    # redirect to another volume underneath the same drive letter.
    if parent_device is not None and not _is_reparse_point(info):
        return parent_device
    return _windows_volume(path)


def _is_reparse_point(info):
    return getattr(info, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT


_active = ContextVar('recognition_disk_budget',default=None)


def _volume_space(path, device=None):
    """A peer may remove its scratch tree between stat and disk_usage."""
    def measure():
        nonlocal device
        current = _device(path, path.stat())
        if device is not None and current != device:
            raise OSError(errno.ENODEV, f'Recognition disk budget volume {device} is unavailable', str(path))
        device = current
        return device, dict(path=str(path), free=shutil.disk_usage(path).free)

    while True:
        try:
            return _retry_access_denied(measure)
        except (FileNotFoundError, NotADirectoryError):
            if path == path.parent:
                raise
            path = path.parent


def _entry_metadata(path, parent_device):
    info = path.lstat()
    if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
        return None
    device = _device(path, info, parent_device)
    if _is_reparse_point(info):
        # Use the target's file index with its volume when deduplicating.
        info = path.stat()
    return device, info


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
        self.directories = {}

    def check(self, growth=0, path=None, *, throttled=False):
        with self.lock:
            return self._check(growth,path,throttled=throttled)

    def _check(self, growth=0, path=None, *, throttled=False):
        if throttled and not growth and self.volumes and time.monotonic()-self.checked<1:return self.volumes
        if not self.limit and not self.headroom:return {}
        from .scratch_ledger import file_leases
        reservations = dict(self.reservations)
        with file_leases() as ledger:
            if ledger is not None:
                reservations.update((Path(name), (device, cap))
                                    for name, device, cap in ledger.execute('SELECT path,device,cap FROM files'))
        counted_sizes = {}
        parent=self.database.parent
        roots=[p for p in parent.glob(self.database.name+'*') if not p.is_symlink()]
        roots.extend(self.roots)
        if work := os.environ.get('RECOGNITION_WORK_ROOT'):
            roots.append(Path(work))
        volumes = {}
        seen = set()
        visited = set()
        pending = [(root, None) for root in dict.fromkeys(roots)]
        visited_paths = set()
        while pending:
            item, parent_device = pending.pop()
            if item in visited_paths:
                continue
            visited_paths.add(item)
            try:
                entry = _retry_access_denied(_entry_metadata, item, parent_device)
            except (FileNotFoundError, NotADirectoryError):
                continue
            if entry is None:
                continue
            dev, info = entry
            # A partial Windows stat may omit the file index as well as st_dev.
            # Distinct anonymous files must each be charged against the quota.
            key = (dev, info.st_ino or os.path.normcase(os.path.realpath(item)))
            if key in seen:
                continue
            if stat.S_ISDIR(info.st_mode):
                stamp = (key, info.st_mtime_ns, info.st_ctime_ns)
                cached = self.directories.get(item)
                if cached is None or cached[0] != stamp or time.monotonic() - cached[2] >= 5:
                    try:
                        children = _retry_access_denied(lambda: tuple(item.iterdir()))
                    except (FileNotFoundError, NotADirectoryError):
                        continue
                    cached = self.directories[item] = stamp, children, time.monotonic()
                visited.add(item)
                pending.extend((child, dev) for child in cached[1])
            elif stat.S_ISREG(info.st_mode):
                row = volumes.setdefault(dev, dict(path=str(item), used=0))
                row['used'] += info.st_size
                if item in reservations:
                    counted_sizes[item] = info.st_size
            seen.add(key)
        for directory in self.directories.keys() - visited:
            del self.directories[directory]
        device, target_space = _volume_space(Path(path or parent))
        volumes.setdefault(device, {'used': 0}).update(target_space)
        for dev,row in volumes.items():
            if dev != device:
                _, space = _volume_space(Path(row['path']), dev)
                row.update(space)
            held = 0
            for path,(volume,cap) in reservations.items():
                if volume == dev:
                    # A worker can grow this file during the scan. Subtract
                    # only bytes already included in the measured volume.
                    held += max(0, cap - counted_sizes.get(path, 0))
            requested=row['used']+held+(growth if dev==device else 0)
            target = row['path']
            free = row['free']
            row.update(requested=requested, reserved=held)
            if self.limit and requested>self.limit:
                raise ResourceLimit(f'Recognition data, caches and scratch exceed the per-volume disk budget at {target}: {requested} bytes required, limit {self.limit}.',
                    budget='execution.rebuild.disk_total_bytes',limit=self.limit,requested=requested)
            required=self.headroom+held+(growth if dev==device else 0)
            if free<required:
                raise ResourceLimit(f'Recognition disk free-space reserve at {target} would be exhausted: {required} bytes required, {free} free.',
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
    from .scratch_ledger import file_leases
    with budget.lock:
        info = path.stat()
        dev, existing = _device(path, info), info.st_size
        with file_leases() as ledger:
            while True:
                # A directory walk can outlast SQLite's busy timeout. Take the
                # writer lock only to validate the measurement and publish.
                version = ledger.execute('PRAGMA data_version').fetchone() if ledger is not None else None
                try:
                    volume = budget.check(path=path)[dev]
                except ResourceLimit:
                    if ledger is None or ledger.execute('PRAGMA data_version').fetchone() == version:
                        raise
                    continue
                held = volume['reserved']
                cap = min(maximum, existing + volume['free'] - budget.headroom - held)
                if budget.limit:
                    cap = min(cap, existing + budget.limit - volume['requested'])
                cap = max(0, cap)
                if ledger is None:
                    break
                ledger.execute('BEGIN IMMEDIATE')
                if ledger.execute('PRAGMA data_version').fetchone() == version:
                    ledger.execute('INSERT OR REPLACE INTO files VALUES (?,?,?)', (str(path), dev, cap))
                    break
                ledger.rollback()
        budget.reservations[path] = (dev, cap)
    try:yield cap
    finally:
        with budget.lock, file_leases(write=True) as ledger:
            budget.reservations.pop(path,None)
            if ledger is not None:
                ledger.execute('DELETE FROM files WHERE path=?',(str(path),))
