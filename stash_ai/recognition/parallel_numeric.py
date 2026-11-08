"""Bounded pure numerical work, drained before exposing results to callers."""
from contextlib import nullcontext
from functools import wraps

from .compute import current, Work, compute_scope
from stash_ai.runtime.cleanup import closing


def diagnostic_scope(function):
    """Reuse a trial's CPU budget for measurements outside the rebuild itself."""
    @wraps(function)
    def wrapped(db, config, *args, **kwargs):
        from .solver_execution import current as execution
        with compute_scope(config) if execution() is not None else nullcontext():
            return function(db, config, *args, **kwargs)
    return wrapped


def map_bounded(function, items, *, cost, payload=lambda item: item, kind, cached=lambda item:None, stream=False):
    """Bound ordered result windows. Nested pool users can drain a suspended window."""
    pool = current()
    if pool is None or pool.workers == 1:
        for item in items:
            found=cached(item)
            yield item, found if found is not None else function(payload(item))
        return
    maximum = max(1, pool.limit//2)
    batch, used = [], 0

    def flush():
        from collections import deque
        if all(found is not None for _, _, found in batch):
            for item, _, found in batch:
                yield item, found
            return
        jobs = (Work(i, kind, found if found is not None else payload(item), amount, ready=found is not None)
                for i, (item, amount, found) in enumerate(batch))
        queued = deque()
        with closing(pool.ordered(function, jobs), name='close '+kind+' numerical jobs') as results:
            def drain():
                pool.drain_numeric = None
                queued.extend(results)
            if not stream:
                drain()
            try:
                for item, _, found in batch:
                    result = queued.popleft() if queued else next(results)
                    if found is None:
                        pool.record(kind, computed=1, consumed=1)
                    pool.drain_numeric = drain
                    try:
                        yield item, result.value
                    finally:
                        pool.drain_numeric = None
            finally:
                # Early termination of a bounded window must leave the pool
                # usable by the next phase. Workers still observe cancellation.
                drain()

    for item in items:
        found = cached(item)
        amount = max(1, cost(item))
        if found is not None:
            from .profiling import count
            count(kind+'_cached_jobs')
        if batch and (used + amount > maximum or len(batch) >= pool.workers*2):
            yield from flush()
            batch, used = [], 0
        if amount > maximum:
            if found is None:
                pool.fallback(kind+'_admission')
            yield item, found if found is not None else function(payload(item))
        else:
            batch.append((item, amount, found))
            used += amount
    if batch:
        yield from flush()


def _matrices(payload):
    from stash_ai.runtime.cancellation import current as cancellation
    token = cancellation()
    values = []
    for left, right, found in payload:
        if token is not None:
            token.check()
        values.append(found if found is not None else left @ right.T)
    return values


def matrix_tiles(tiles, *, kind):
    """(metadata, matrix-pairs) -> (metadata, products), with exact tile caching."""
    from .run_cache import product_input, product_output
    def prepare():
        for metadata, pairs in tiles:
            keys, payload, size = [], [], 4096
            for left, right in pairs:
                identity, found = product_input(left, right)
                keys.append(identity)
                payload.append((left, right, found))
                if found is None:size += 3*(left.nbytes+right.nbytes) + 2*len(left)*len(right)*8
            yield metadata, keys, payload, size
    for item, values in map_bounded(_matrices, prepare(), cost=lambda item: item[3],
                                    payload=lambda item: [p for p in item[2] if p[2] is None], kind=kind, stream=True,
                                    cached=lambda item: [] if all(p[2] is not None for p in item[2]) else None):
        metadata, keys, payload, _ = item
        values=iter(values)
        yield metadata, [inputs[2] if inputs[2] is not None else product_output(key, next(values))
                         for key, inputs in zip(keys, payload)]
