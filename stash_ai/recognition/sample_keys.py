"""Bounded canonical physical-key reuse; all dependency values address a key."""
from contextlib import contextmanager
from contextvars import ContextVar

from .config import digest
from .evidence_cache import MemoryCache, MISSING
from .profiling import count

_active = ContextVar('recognition_physical_keys', default=None)
_FIELDS = ('scope','namespace','scene','input_sha','cue','seconds','source','rect','geometry')


def sample_key(row):
    values = tuple(row[key] for key in _FIELDS)
    cache = _active.get()
    if cache is not None:
        found = cache.get(values)
        if found is not MISSING:
            count('physical_sample_key_reuses')
            return found
    result = digest(values)
    count('physical_sample_keys')
    if cache is not None:
        size = 512 + sum(128+len(v.encode('utf-8'))*4 if isinstance(v,str) else 128 for v in values)
        cache.put(values,result,size)
    return result


@contextmanager
def physical_keys(database, config):
    limit = config.data['execution']['rebuild_memory_bytes']//8
    cache = getattr(database,'_physical_sample_keys',None)
    if cache is None or cache.limit != limit:
        cache = database._physical_sample_keys = MemoryCache(limit)
    token = _active.set(cache)
    try:
        yield
    finally:
        _active.reset(token)
