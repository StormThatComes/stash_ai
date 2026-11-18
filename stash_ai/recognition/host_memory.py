"""Optional shared host admission for scan pools, queues and rebuild work.

RSS is sampled, not a hard OS limit. CUDA memory is handled by gpu_memory.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from threading import RLock
from .resource_errors import ResourceLimit
from stash_ai.runtime.memory import process_rss

_active=ContextVar('recognition_host_memory',default=None)


class Budget:
    def __init__(self,limit):
        self.limit=limit;self.baseline=process_rss() or 0
        self.owners={};self.used=0;self.lock=RLock()
    def replace(self,owner,amount,*,required=False):
        with self.lock:
            old=self.owners.get(owner,0)
            # Release must always work, including after unrelated native growth.
            if amount<=old:
                self.used+=amount-old
                if amount:self.owners[owner]=amount
                else:self.owners.pop(owner,None)
                return True
            requested=max(self.baseline+self.used,process_rss() or 0)+amount-old
            if requested>self.limit:
                if required:
                    raise ResourceLimit(f'Recognition host admission exceeds execution.host_memory_mb: owner={owner[0]}, requested_bytes={requested}, limit_bytes={self.limit}. Reduce batches/prefetch/cache limits or raise the host budget.',budget='execution.host_memory_mb',limit=self.limit,requested=requested)
                return False
            self.owners[owner]=amount;self.used+=amount-old
            return True
    def available(self):
        with self.lock:return max(0,self.limit-max(self.baseline+self.used,process_rss() or 0))


def current():return _active.get()


@contextmanager
def scope(config):
    if current() is not None or not config.data['execution'].get('host_memory_bytes',0):
        yield current();return
    budget=Budget(config.data['execution']['host_memory_bytes'])
    token=_active.set(budget)
    try:yield budget
    finally:_active.reset(token)
