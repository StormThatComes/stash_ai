"""Bounded face measurements; cache only immutable numerical inputs."""
import math


def reference_stats(left, right, *, max_bytes=4194304):
    import numpy as np
    from .run_cache import current, immutable_samples, MISSING
    from .memory_size import Reservation
    if not len(left) or not len(right):
        raise ValueError('Reference statistics require nonempty banks.')
    dimensions = len(left[0])
    if max_bytes < 8*(2*dimensions+1):
        raise ValueError('Face measurement working budget is too small for one pair.')
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
