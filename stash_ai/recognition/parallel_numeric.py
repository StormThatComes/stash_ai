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


def map_bounded(function, items, *, cost, payload=lambda item: item, kind):
    """Yield (input, output) in order; consumers may schedule their own work.

    A whole batch retires before yielding, avoiding nested active CPU streams.
    Half the queue is reserved for a prepared lookahead item. Oversized jobs
    retain the exact scalar path; neither item count nor bytes grow unbounded.
    """
    pool = current()
    if pool is None or pool.workers == 1:
        for item in items:
            yield item, function(payload(item))
        return
    maximum = max(1, pool.limit//2)
    batch, used = [], 0

    def flush():
        jobs = (Work(i, kind, payload(item), amount)
                for i, (item, amount) in enumerate(batch))
        values = []
        with closing(pool.ordered(function, jobs), name='close '+kind+' numerical jobs') as stream:
            for result in stream:
                values.append(result.value)
        pool.record(kind, computed=len(values), consumed=len(values))
        return [(item, value) for (item, _), value in zip(batch, values)]

    for item in items:
        amount = cost(item)
        if batch and (used+amount > maximum or len(batch) >= pool.workers*2):
            yield from flush()
            batch, used = [], 0
        if amount > maximum:
            pool.fallback(kind+'_admission')
            yield item, function(payload(item))
        else:
            batch.append((item, amount))
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
                size += 3*(left.nbytes+right.nbytes) + 2*len(left)*len(right)*8
            yield metadata, keys, payload, size
    for item, values in map_bounded(_matrices, prepare(), cost=lambda item: item[3],
                                    payload=lambda item: item[2], kind=kind):
        metadata, keys, payload, _ = item
        yield metadata, [value if inputs[2] is not None else product_output(key, value)
                         for key, inputs, value in zip(keys, payload, values)]
