"""Conservative Python/container ownership estimate for bounded work admission."""
import sys
from collections.abc import Mapping


def size(value, seen=None):
    seen=set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    total=sys.getsizeof(value)
    if isinstance(value,Mapping):
        total+=sum(size(k,seen)+size(v,seen) for k,v in value.items())
    elif isinstance(value,(tuple,list,set,frozenset)):
        total+=sum(size(v,seen) for v in value)
    elif hasattr(value,'nbytes'):
        total+=value.nbytes
    return total


class Reservation:
    def __init__(self, limit):
        from .run_cache import current
        import weakref
        self.cache, self.limit = current(), limit
        self.owner = ('metadata',id(self))
        if self.cache is not None:
            weakref.finalize(self,self.cache.reserve,self.owner,0)

    def set(self, required):
        if required > self.limit:
            raise ValueError('Recognition metadata exceeds execution.rebuild_memory_bytes.')
        if self.cache is not None:
            self.cache.reserve(self.owner,required)
