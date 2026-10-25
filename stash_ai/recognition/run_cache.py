"""Disposable run-wide immutable evidence, under one conservative byte budget.

Keys contain content or a live database generation, never an identity label
alone. Eviction changes cost only: all callers have their original exact path.
"""
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from types import MappingProxyType
import json

from .profiling import count

MISSING = object()
_active = ContextVar('recognition_run_cache', default=None)


class Cache:
    def __init__(self, limit, matrix_limit):
        self.limit, self.matrix_limit = limit, min(limit, matrix_limit)
        self.entries, self.size, self.peak = OrderedDict(), 0, 0
        self.matrices, self.matrix_bytes = OrderedDict(), 0
        self.working = None
        self.reservations = {}
        self.reserved_size = 0
        self.ceiling, self.matrix_ceiling = limit, matrix_limit

    def configure_working(self, working, ceiling):
        self.working, self.ceiling = working, ceiling
        self._resize()

    def reserve(self, owner, size):
        if self.working is None:
            return
        total = self.reserved_size - self.reservations.get(owner, 0) + size
        if total > self.working:
            raise ValueError('Recognition working allocations exceed execution.rebuild_memory_bytes.')
        self.reserved_size = total
        if size:
            self.reservations[owner] = size
        else:
            self.reservations.pop(owner, None)
        self._resize()

    def _resize(self):
        if self.working is None:
            return
        self.limit = max(0, min(self.ceiling, self.working-self.reserved_size))
        self.matrix_limit = min(self.matrix_ceiling, self.limit)
        while self.entries and self.size > self.limit:
            self._remove(next(iter(self.entries)))
            count('ram_evictions')
        while self.matrices and self.matrix_bytes > self.matrix_limit:
            self._remove(next(iter(self.matrices)))
            count('ram_evictions')

    def get(self, key):
        found = self.entries.pop(key, MISSING)
        count('ram_' + key[0] + ('_misses' if found is MISSING else '_hits'))
        if found is MISSING:
            return MISSING
        self.entries[key] = found
        if key[0] == 'matrix':
            self.matrices.move_to_end(key)
        return found[0]

    def _remove(self, key):
        _, size = self.entries.pop(key)
        self.size -= size
        if key[0] == 'matrix':
            self.matrices.pop(key)
            self.matrix_bytes -= size

    def put(self, key, value, size):
        # Callers include retained key owners as well as payload and scratch.
        size += 512
        if key in self.entries:
            self._remove(key)
        limit = self.matrix_limit if key[0] == 'matrix' else self.limit
        if size > limit:
            count('ram_' + key[0] + '_oversize')
            return value
        while self.entries and self.size + size > self.limit:
            self._remove(next(iter(self.entries)))
            count('ram_evictions')
        if key[0] == 'matrix':
            while self.matrices and self.matrix_bytes + size > self.matrix_limit:
                self._remove(next(iter(self.matrices)))
                count('ram_evictions')
            self.matrices[key] = None
            self.matrix_bytes += size
        self.entries[key] = value, size
        self.size += size
        self.peak = max(self.peak, self.size)
        return value

    def discard(self, predicate):
        for key in list(self.entries):
            if predicate(key):
                self._remove(key)

    def clear(self):
        self.entries.clear()
        self.matrices.clear()
        self.size = self.matrix_bytes = 0


class Partition:
    """A sealed replay's evidence shares the run budget, with private keys."""
    def __init__(self, cache):
        self.cache, self.owner = cache, object()
        self.limit = cache.limit

    def get(self, key):
        return self.cache.get(('evidence', self.owner, key))

    def put(self, key, value, size):
        return self.cache.put(('evidence', self.owner, key), value, size)

    def discard_vectors(self):
        self.cache.discard(lambda k: k[0] == 'evidence' and k[1] is self.owner and k[2][0] == 'vector')

    def clear(self):
        self.cache.discard(lambda k: k[0] == 'evidence' and k[1] is self.owner)


def current():
    return _active.get()


@contextmanager
def scope(config):
    if current() is not None:
        yield current()
        return
    settings = config.data['execution']
    # Reserve the remaining memory for replay SQLite, worker queues and banks.
    from .resource_plan import plan
    limit = plan(config).cache
    value = Cache(limit, settings['rebuild_matrix_cache_bytes'])
    value.configure_working(plan(config).working, settings.get('rebuild_cache_bytes',0))
    token = _active.set(value)
    try:
        yield value
    finally:
        count('ram_cache_peak_bytes', value.peak)
        count('ram_cache_limit_bytes', value.limit)
        value.clear()
        _active.reset(token)


def immutable_samples(samples):
    """Exact keys only for complete, immutable float32 byte-backed vectors."""
    import numpy as np
    keys = []
    for sample in samples:
        if (not isinstance(sample, np.ndarray) or sample.dtype != np.dtype('<f4') or sample.ndim != 1
                or sample.flags.writeable or not isinstance(sample.base, bytes)
                or sample.nbytes != len(sample.base) or not sample.flags.c_contiguous):
            return None
        keys.append(sample.base)
    return tuple(keys)


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    return tuple(_freeze(v) for v in value) if isinstance(value, list) else value


def metadata(raw):
    cache = current()
    if cache is None or not isinstance(raw, str):
        return json.loads(raw)
    key = 'metadata', raw
    found = cache.get(key)
    if found is MISSING:
        found = _freeze(json.loads(raw))
        # JSON nodes can dwarf their source; reserve a deliberately generous bound.
        cache.put(key, found, len(raw)*128 + 256)
    return found


def cross_scores(left, right):
    """Cache only numerical evidence; gates, ownership and rivals stay live."""
    import numpy as np
    cache = current()
    a, b = immutable_samples(left), immutable_samples(right)
    key = ('pair', a, b) if a is not None and b is not None else None
    result = cache.get(key) if cache is not None and key is not None else MISSING
    if result is MISSING:
        scores = [float(np.dot(x, y)) for x in left for y in right]
        result = min(scores), sum(scores)/len(scores)
        if cache is not None and key is not None:
            cache.put(key, result, sum(len(x)+128 for x in a+b))
    return result


def database_key(db):
    """Rollback-safe generation, including writes through other connections."""
    from .dependencies import supported, version
    if current() is None or not supported(db):
        return None
    raw = getattr(db, 'raw', db)
    return raw, raw.total_changes, raw.execute('PRAGMA data_version').fetchone()[0], version(db)
