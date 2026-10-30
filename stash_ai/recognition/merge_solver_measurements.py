"""Bounded face measurements; cache only immutable numerical inputs."""
from .resource_errors import ResourceLimit
import math


def reference_stats(left, right, *, max_bytes=4194304):
    import numpy as np
    from .run_cache import current, immutable_samples, MISSING
    from .memory_size import Reservation
    if not len(left) or not len(right):
        raise ValueError('Reference statistics require nonempty banks.')
    dimensions = len(left[0])
    if max_bytes < 8*(2*dimensions+1):
        raise ResourceLimit('Face measurement working budget is too small for one pair.')
    cache = current()
    a, b = immutable_samples(left), immutable_samples(right)
    key = ('solver-face-stats', max_bytes, a, b) if a is not None and b is not None else None
    found = cache.get(key) if cache is not None and key is not None else MISSING
    if found is not MISSING:
        return found
    side = min(max(len(left), len(right)), max(1, int(math.sqrt(max_bytes/8))))
    while 8*(side*side+2*side*dimensions) > max_bytes:
        side = max(1, side//2)
    reservation = Reservation(max_bytes)
    reservation.set(8*(side*side+2*side*dimensions))
    minimum, maximum, weakest = math.inf, -math.inf, (0, 0)
    def tiles():
        nonlocal minimum, maximum, weakest
        for i in range(0, len(left), side):
            x = np.asarray(left[i:i+side], dtype='float64')
            for j in range(0, len(right), side):
                y = np.asarray(right[j:j+side], dtype='float64')
                matrix = x @ y.T
                row, col = np.unravel_index(matrix.argmin(), matrix.shape)
                value, position = float(matrix[row, col]), (i+int(row), j+int(col))
                if value < minimum or value == minimum and position < weakest:
                    minimum, weakest = value, position
                maximum = max(maximum, float(matrix.max()))
                total = float(matrix.sum(dtype='float64'))
                del matrix, y
                yield total
            del x
    try:
        # fsum consumes a stream; the number of tiles does not grow a Python list.
        mean = math.fsum(tiles())/(len(left)*len(right))
        result = minimum, mean, maximum, weakest
    finally:
        reservation.set(0)
    if cache is not None and key is not None:
        cache.put(key, result, sum(len(v)+128 for v in a+b)+256)
    return result


def _measure(payload):
    left, right, max_bytes, cached = payload
    return cached if cached is not None else reference_stats(left, right, max_bytes=max_bytes)


def measured_pairs(pairs, *, max_bytes=4194304):
    """Parallel complete face banks; only the coordinator reads/writes caches."""
    from .parallel_numeric import map_bounded
    from .run_cache import current, immutable_samples, MISSING
    from .solver_shared import current as shared_current, key as shared_key
    import numpy as np
    cache = current()
    shared = shared_current()
    def prepared():
        for left, right in pairs:
            a, b = immutable_samples(left.samples), immutable_samples(right.samples)
            key = ('solver-face-stats', max_bytes, a, b) if a is not None and b is not None else None
            found = cache.get(key) if cache is not None and key is not None else MISSING
            signature = shared_key('face-stats-v1', (*left.samples, *right.samples),
                extra=(max_bytes,len(left.samples))) if shared is not None and found is MISSING else None
            if found is MISSING and shared is not None:
                value = shared.get(signature)
                if value is not None:
                    found = (float(value[0]),float(value[1]),float(value[2]),(int(value[3]),int(value[4])))
                    if cache is not None and key is not None:
                        cache.put(key,found,sum(len(v)+128 for v in key[2]+key[3])+256)
            size = sum(v.nbytes+128 for v in (*left.samples, *right.samples)) + max_bytes + 4096
            yield left, right, key, None if found is MISSING else found, size, signature
    def payload(item):
        a, b, _, cached, _, _ = item
        return tuple(a.samples), tuple(b.samples), max_bytes, cached
    for item, result in map_bounded(_measure, prepared(), cost=lambda item: item[4],
                                    payload=payload, kind='face_measurements',cached=lambda item:item[3]):
        left, right, key, cached, _, signature = item
        if cached is None and shared is not None:
            shared.put(signature,np.asarray([*result[:3],*result[3]],dtype='float64'))
        if cached is None and cache is not None and key is not None:
            cache.put(key, result, sum(len(v)+128 for v in key[2]+key[3])+256)
        yield left, right, result


def diagnostic_pairs(groups, config):
    """Bound search hints separately from complete production acceptance.

    Small goal sets retain the historical order. Large sets visit adjacent
    owners around a deterministic ring before denser distances, giving sparse
    fragments a chance without allocating or walking every possible pair.
    """
    from itertools import combinations, islice
    limits = config.data['sapiensid']['execution']
    pair_limit = min(limits['max_candidate_pairs'], limits['plan_bytes']//8192)
    witness_limit = limits['max_witness_comparisons']
    total = len(groups)*(len(groups)-1)//2
    coverage = dict(total_pairs=total, examined_pairs=0, measured_pairs=0,
        witness_comparisons=0, pair_limit=pair_limit, witness_limit=witness_limit,
        complete=False, sampling='all_pairs' if total<=pair_limit else 'deterministic_ring',
        purpose='search_hints_only')
    def ring():
        n=len(groups)
        for distance in range(1,n//2+1):
            for i in range(n//2 if distance*2==n else n):
                yield groups[i],groups[(i+distance)%n]
    def admitted():
        source=combinations(groups,2) if total<=pair_limit else ring()
        for left,right in islice(source,pair_limit):
            coverage['examined_pairs']+=1
            cost=len(left.samples)*len(right.samples)
            if coverage['witness_comparisons']+cost>witness_limit:
                continue
            coverage['witness_comparisons']+=cost
            coverage['measured_pairs']+=1
            yield left,right
        coverage['complete']=coverage['measured_pairs']==total
    return admitted(),coverage
