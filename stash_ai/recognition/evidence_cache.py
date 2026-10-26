"""RAM-only cue evidence for one coordinator's sealed scan catalogue."""
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import hashlib
import weakref

from .profiling import count

from .run_cache import MISSING
from .scan_heads import predicate as current_scan
_active = ContextVar('recognition_evidence_cache', default=None)
LATEST = 'AND ' + current_scan()


class MemoryCache:
    def __init__(self, limit):
        self.limit, self.size, self.entries = limit, 0, OrderedDict()

    def get(self, key):
        found = self.entries.pop(key, MISSING)
        if found is MISSING:
            return MISSING
        self.entries[key] = found
        count('evidence_cache_hits')
        return found[0]

    def put(self, key, value, size):
        size += 256
        old = self.entries.pop(key, None)
        if old is not None:
            self.size -= old[1]
        if size > self.limit:
            return value
        while self.entries and self.size + size > self.limit:
            _, old = self.entries.popitem(last=False)
            self.size -= old[1]
            count('evidence_cache_evictions')
        self.entries[key] = value, size
        self.size += size
        return value


class Evidence:
    def __init__(self, db, config):
        self.db = db
        from .run_cache import current, Partition
        shared = current()
        self.cache = (Partition(shared) if shared is not None else
                      MemoryCache(config.data['execution']['rebuild_memory_bytes']//4))

    def heads(self, scope, namespace):
        key = 'heads', scope, namespace
        value = self.cache.get(key)
        if value is not MISSING:
            return value
        total = self.db.execute('SELECT COUNT(*) FROM scans WHERE scope=? AND namespace=?',
                                (scope, namespace)).fetchone()[0]
        if total*512 > self.cache.limit//2:
            return self.cache.put(key, None, 0)
        heads = {}
        for row in self.db.execute('SELECT s.id,s.scene FROM scans s WHERE s.scope=? AND s.namespace=? AND '+current_scan(),
                                   (scope, namespace)):
            heads[row['scene']] = row['id']
        count('latest_scan_catalogues')
        return self.cache.put(key, heads, len(heads)*512)

    def peers(self, scan, cue):
        key = 'peers', scan, cue
        value = self.cache.get(key)
        if value is MISSING:
            sql = ('SELECT id,geometry,dimensions,embedding FROM observations '
                   'WHERE scan=? AND cue=? AND rejected IS NULL ORDER BY id')
            value, size = [], 0
            for row in self.db.execute(sql, (scan, cue)):
                size += _row_bytes(row)
                if size > self.cache.limit//2:
                    return self.db.execute(sql, (scan, cue))
                value.append(row)
            self.cache.put(key, value, size)
        return value

    def vector(self, row):
        from .clustering import vector
        key = 'vector', row['id']
        value = self.cache.get(key)
        if value is MISSING:
            from .replay_memory import prepared_vector, MISSING as UNPREPARED
            value = prepared_vector(row)
            if value is UNPREPARED:
                value = vector(row)
            self.cache.put(key, value, value.nbytes + 256)
        return value

    def discard_vectors(self):
        if hasattr(self.cache, 'discard_vectors'):
            self.cache.discard_vectors()
            return
        for key in list(self.cache.entries):
            if key[0] == 'vector':
                _, size = self.cache.entries.pop(key)
                self.cache.size -= size

    def signature(self, row):
        key = 'signature', row['id']
        value = self.cache.get(key)
        if value is MISSING:
            value = hashlib.sha256(row['geometry'].encode() + row['embedding']).digest()
            self.cache.put(key, value, 128)
        return value

    def scores(self, group, query, observation):
        from .clustering import _similarities
        # Only immutable byte-backed representatives can own cached scores.
        layout = getattr(group, '_evidence_layout', None)
        if isinstance(group.samples, tuple) and layout is not None and layout[0] is group.samples:
            immutable = layout[1]
        else:
            immutable = all(not sample.flags.writeable and isinstance(sample.base, bytes)
                            for sample in group.samples)
            if isinstance(group.samples, tuple):
                group._evidence_layout = group.samples, immutable
        if not immutable:
            return _similarities(group, query)
        version = (('tuple', id(group.samples)) if isinstance(group.samples, tuple)
                   else tuple(id(sample) for sample in group.samples))
        key = 'scores', observation, version
        value = self.cache.get(key)
        if value is MISSING or not same_owners(value[1], group.samples):
            scores = _similarities(group, query)
            value = scores, tuple(weakref.ref(sample) for sample in group.samples)
            self.cache.put(key, value, sum(sample.nbytes + 128 for sample in group.samples))
        return value[0]

    def remember_scores(self, group, observation, scores):
        """Import a verified immutable worker result on the coordinator only."""
        if not all(not sample.flags.writeable and isinstance(sample.base, bytes)
                   for sample in group.samples):
            return
        version = (('tuple', id(group.samples)) if isinstance(group.samples, tuple)
                   else tuple(id(sample) for sample in group.samples))
        self.cache.put(('scores', observation, version),
                       (scores, tuple(weakref.ref(sample) for sample in group.samples)),
                       sum(sample.nbytes + 128 for sample in group.samples))

    def contextual(self, sql, parameters, selected):
        key = 'context', selected['scope'], selected['namespace'], selected['policy_sha']
        value = self.cache.get(key)
        if value is MISSING:
            rows, size = [], 0
            for row in latest_rows(self.db, sql, parameters, selected):
                size += _row_bytes(row) + 256
                if size > self.cache.limit//2:
                    # Recompute from SQLite rather than spilling working RAM.
                    rows = None
                    break
                rows.append(row)
            value = rows
            self.cache.put(key, value, size if rows is not None else 0)
        return latest_rows(self.db, sql, parameters, selected) if value is None else iter(value)


def same_owners(refs, samples):
    return len(refs) == len(samples) and all(ref() is sample for ref, sample in zip(refs, samples))


def _row_bytes(row):
    return 512 + sum(len(row[name]) if isinstance(row[name], (str, bytes)) else 0
                     for name in ('embedding', 'geometry'))


def current(db=None):
    value = _active.get()
    return value if value is not None and (db is None or value.db is db) else None


def latest_rows(db, sql, parameters, selected):
    cache = current(db)
    heads = cache.heads(selected['scope'], selected['namespace']) if cache is not None else None
    if heads is None:
        yield from db.execute(sql.replace('{latest}', LATEST), parameters)
    else:
        for row in db.execute(sql.replace('{latest}', ''), parameters):
            if heads.get(row['scene']) == row['scan']:
                yield row


@contextmanager
def evidence_scope(db, config):
    if current(db) is not None:
        yield current(db)
        return
    value = Evidence(db, config)
    token = _active.set(value)
    try:
        yield value
    finally:
        if hasattr(value.cache, 'clear'):
            value.cache.clear()
        else:
            value.cache.entries.clear()
        _active.reset(token)


def scoped(function):
    @wraps(function)
    def call(database, scan, config, *args, **kwargs):
        with evidence_scope(database.connection, config):
            return function(database, scan, config, *args, **kwargs)
    return call
