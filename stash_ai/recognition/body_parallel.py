"""Exact complete body/face witness proofs; workers receive vectors, never SQL."""
from itertools import islice
import numpy as np

from .compute import current
from .parallel_numeric import map_bounded


def _compare(payload):
    left, right = payload
    from stash_ai.runtime.cancellation import current as cancellation
    token = cancellation()
    total, minimum = 0., float('inf')
    for first in left:
        if token is not None:
            token.check()
        for second in right:
            if len(first) != len(second):
                raise ValueError('Incompatible face namespaces in body consolidation.')
            # Preserve production float32 dot and Python accumulation order.
            score = float(np.dot(first, second))
            total += score
            minimum = min(minimum, score)
    count = len(left)*len(right)
    return (total/count, minimum) if count else (-1., -1.)


def comparisons(evidence, requests):
    """Yield (caller metadata, score) with bounded deduplication and input banks.

    Keys use the immutable catalogue's witness IDs. Budgets, SQLite reads and
    cache publication belong to the coordinator; no original core is sampled.
    """
    pool = current()
    if pool is None or pool.workers == 1:
        for metadata, left, right, kind in requests:
            yield metadata, evidence.compare(left, right, kind)
        return
    source = iter(requests)
    while batch := list(islice(source, pool.workers*2)):
        resolved, pending = {}, {}
        def prepare():
            for position, (metadata, left, right, kind) in enumerate(batch):
                key = (kind, tuple(r['id'] for r in left), tuple(r['id'] for r in right))
                cacheable = len(left)+len(right) <= 128
                if key in evidence.scores:
                    evidence.stats['score_cache_hits'] += 1
                    evidence.scores.move_to_end(key)
                    resolved[position] = evidence.scores[key]
                    continue
                if cacheable and key in pending:
                    evidence.stats['score_cache_hits'] += 1
                    pending[key].append(position)
                    continue
                # Body dimensions are fixed; face dimensions are at most 1024.
                estimate = (len(left)+len(right))*(4096*4+512)+4096
                if estimate > pool.limit//4:
                    pool.fallback('body_proof_admission')
                    resolved[position] = evidence.compare(left, right, kind)
                    continue
                a = tuple(evidence.feature(row, kind) for row in left)
                b = tuple(evidence.feature(row, kind) for row in right)
                from .body_numeric import lookup
                identity, found = lookup(a, b, kind)
                if found is not None:
                    evidence.stats['score_cache_hits'] += 1
                    resolved[position] = found
                    if cacheable:evidence.remember(key,found)
                    continue
                evidence.charge(len(left)*len(right), kind)
                for vector in a+b:
                    vector.flags.writeable = False
                positions = [position]
                if cacheable:
                    pending[key] = positions
                yield key, positions, (a, b), estimate, cacheable, identity
        for item, value in map_bounded(_compare, prepare(), cost=lambda item: item[3],
                                      payload=lambda item: item[2], kind='body_witness_proofs'):
            key, positions, _, _, cacheable, identity = item
            from .body_numeric import publish
            value = publish(identity, value)
            if cacheable:
                evidence.remember(key, value)
            for position in positions:
                resolved[position] = value
        # A duplicate can be discovered after the original job retired.
        for key, positions in pending.items():
            value = resolved[positions[0]]
            for position in positions:
                resolved[position] = value
        for position, (metadata, _, _, _) in enumerate(batch):
            yield metadata, resolved[position]
