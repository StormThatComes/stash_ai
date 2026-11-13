"""Bounded, cancellable desktop reads. Worker threads never own Tk objects."""
from collections import OrderedDict, deque
from copy import deepcopy
from functools import wraps
from pathlib import Path
import threading

local = threading.local()


class Cancelled(Exception):
    pass


def check_cancelled():
    if getattr(local, 'cancel', None) is not None and local.cancel.is_set():
        raise Cancelled()


def open_readonly(path):
    from .fragmentation_report import _open_readonly
    db = _open_readonly(path)
    event = getattr(local, 'cancel', None)
    if event is not None:
        db.set_progress_handler(lambda: int(event.is_set()), 2000)
    return db


def stamp(path):
    result = []
    for suffix in ('', '-wal'):
        try:
            stat = Path(str(path)+suffix).stat()
            result.append((stat.st_ino, stat.st_size, stat.st_mtime_ns))
        except FileNotFoundError:
            result.append(None)
    return tuple(result)


def weight(value):
    if hasattr(value, 'size') and hasattr(value, 'getbands'):
        return value.width*value.height*len(value.getbands())
    if isinstance(value, dict):
        return sum(weight(k)+weight(v) for k,v in value.items())
    if isinstance(value, (list,tuple)):
        return sum(map(weight,value))
    return len(value) if isinstance(value,(str,bytes)) else 32


class Cache:
    def __init__(self, limit=32*1048576, entries=128):
        self.limit, self.entries, self.size = limit, entries, 0
        self.items, self.lock = OrderedDict(), threading.Lock()

    def get(self, key):
        with self.lock:
            if key not in self.items:
                return None
            value, size = self.items.pop(key)
            self.items[key] = value, size
            return deepcopy(value)

    def put(self, key, value):
        size = weight(value)
        if size > self.limit:
            return
        with self.lock:
            old = self.items.pop(key, None)
            if old: self.size -= old[1]
            self.items[key] = deepcopy(value), size
            self.size += size
            while self.size > self.limit or len(self.items) > self.entries:
                _, size = self.items.popitem(last=False)[1]
                self.size -= size

    def clear(self):
        with self.lock:
            self.items.clear()
            self.size = 0


metadata_cache = Cache(8*1048576)


def cached(function):
    @wraps(function)
    def read(self, *args, **kwargs):
        from .config import digest
        revision = stamp(self.config.database)
        selector = getattr(self,'selector',None) or getattr(self,'scan',None) or (self.scene,self.scope,self.namespace)
        key = (str(self.config.database),revision,digest(self.config.data),repr(selector),function.__qualname__,repr(args),repr(sorted(kwargs.items())))
        result = metadata_cache.get(key)
        if result is not None:
            return result
        result = function(self,*args,**kwargs)
        check_cancelled()
        if stamp(self.config.database) == revision:
            metadata_cache.put(key,result)
        return result
    return read


class Worker:
    """Fixed daemon pool, latest pending request, cooperative I/O/SQL cancellation."""
    def __init__(self, name, *, workers=1, resource=None):
        self.condition = threading.Condition()
        self.pending, self.completed, self.active = None, deque(), {}
        self.generation, self.closed = 0, False
        self.resource = resource
        for index in range(workers):
            threading.Thread(target=self.run, name=f'{name}-{index}', daemon=True).start()

    @property
    def future(self):
        with self.condition:
            return True if self.pending is not None or self.active else None

    def invalidate(self):
        with self.condition:
            self.generation += 1
            self.pending = None
            self.completed.clear()
            for event in self.active.values(): event.set()

    def request(self, operation, complete, failed):
        self.invalidate()
        with self.condition:
            if self.closed: return
            self.pending = self.generation, operation, complete, failed
            self.condition.notify()

    def run(self):
        service = None
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.closed or self.pending is not None)
                    if self.closed: return
                    generation, operation, complete, failed = self.pending
                    self.pending = None
                    event = self.active[generation] = threading.Event()
                local.cancel = event
                try:
                    if service is None and self.resource:
                        service = self.resource()
                    local.preview = service
                    result = operation()
                    error = None
                except Exception as exception:
                    result, error = None, exception
                with self.condition:
                    self.active.pop(generation, None)
                    if not self.closed and generation == self.generation:
                        self.completed.append((complete,failed,result,error))
        finally:
            if service is not None: service.close()

    def poll(self):
        with self.condition:
            results = list(self.completed)
            self.completed.clear()
        for complete,failed,result,error in results:
            if error is None: complete(result)
            elif not isinstance(error,Cancelled): failed(error)

    def close(self, release=lambda: None):
        self.invalidate()
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        release()
