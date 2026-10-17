"""Bounded native score blocks with conservative rejection and exact finalists.

BLAS reduction order may differ from scalar np.dot. Approximate blocks may only
prove rejection, never supply a decision score. Every remaining comparison uses
the original scalar reduction, including all threshold/tie boundary cases.
"""
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import weakref

from .profiling import count

_blocks = ContextVar('recognition_score_blocks', default=None)
_MAX_BLOCK = 8 * 1024 * 1024


def scoped(function):
    @wraps(function)
    def call(*args, **kwargs):
        with score_scope():
            return function(*args, **kwargs)
    return call


@contextmanager
def score_scope():
    if _blocks.get() is not None:
        yield
        return
    state = dict(entries=OrderedDict(), size=0, limit=64*1024*1024)
    token = _blocks.set(state)
    try:
        yield
    finally:
        state['entries'].clear()
        _blocks.reset(token)


def _prepared(groups, query):
    import numpy as np
    state = _blocks.get()
    key = tuple(('tuple', id(group.samples)) if isinstance(group.samples, tuple)
                else tuple(id(sample) for sample in group.samples) for group in groups)
    arrays = [sample for group in groups for sample in group.samples]
    if state is not None and key in state['entries']:
        value = state['entries'].pop(key)
        if len(value[4]) == len(arrays) and all(ref() is sample for ref, sample in zip(value[4], arrays)):
            state['entries'][key] = value
            count('reused_packed_bytes', value[0].nbytes)
            return value[:4]
        state['size'] -= value[-1]

    if (not arrays or any(not isinstance(sample, np.ndarray) or sample.dtype != np.float32
                          or sample.shape != query.shape for sample in arrays)
            or any(not group.samples for group in groups)):
        return None
    matrix = np.stack(arrays)
    count('repacked_bytes', matrix.nbytes)
    lengths = np.asarray([len(group.samples) for group in groups], dtype=np.int64)
    starts = np.concatenate(([0], lengths.cumsum()[:-1]))
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    size = (matrix.nbytes + norms.nbytes + starts.nbytes + lengths.nbytes
            + sum(sample.nbytes for sample in arrays) + 128*len(arrays))
    if (state is not None and size <= state['limit']
            and all(not sample.flags.writeable and isinstance(sample.base, bytes) for sample in arrays)):
        while state['entries'] and state['size'] + size > state['limit']:
            _, removed = state['entries'].popitem(last=False)
            state['size'] -= removed[-1]
        # Weak owners guard identity keys without retaining obsolete references.
        state['entries'][key] = matrix, lengths, starts, norms, tuple(weakref.ref(a) for a in arrays), size
        state['size'] += size
    return matrix, lengths, starts, norms


def comparisons(groups, query, threshold=float('-inf'), cohesion=float('-inf'), *, exact=None):
    """Yield unchanged scalar scores for every potentially relevant group."""
    import numpy as np
    if exact is None:
        from .clustering import _similarities
        exact = _similarities
    if (not isinstance(query, np.ndarray) or query.dtype != np.float32 or query.ndim != 1
            or hasattr(groups, '__len__') and len(groups) < 16):
        for group in groups:
            yield group, *exact(group, query)
        return
    block, size = [], 0
    for group in groups:
        layout = getattr(group, '_score_layout', None)
        if isinstance(group.samples, tuple) and layout is not None and layout[0] is group.samples:
            required = layout[1]
        else:
            required = sum(sample.nbytes for sample in group.samples)
            if isinstance(group.samples, tuple):
                group._score_layout = group.samples, required
        if block and size + required > _MAX_BLOCK:
            yield from _compare(block, query, threshold, cohesion, exact)
            block, size = [], 0
        block.append(group)
        size += required
    if block:
        yield from _compare(block, query, threshold, cohesion, exact)


def _compare(groups, query, threshold, cohesion, exact):
    import numpy as np
    prepared = _prepared(groups, query)
    if prepared is None:
        for group in groups:
            yield group, *exact(group, query)
        return
    matrix, lengths, starts, norms = prepared
    scores = matrix @ query
    # Each f32 dot can differ from the mathematical dot by gamma_d*||a||||b||.
    # Four gamma terms cover both reduction orders plus conservative margin.
    dimension = len(query)
    unit = np.finfo(np.float32).eps / 2
    gamma = dimension * unit / (1 - dimension * unit)
    errors = (4*gamma + 16*np.finfo(np.float64).eps) * norms * np.linalg.norm(query.astype(np.float64))
    errors += dimension * np.finfo(np.float32).tiny
    means = np.add.reduceat(scores.astype(np.float64), starts) / lengths
    mean_errors = np.add.reduceat(errors, starts) / lengths
    minimum_upper = np.minimum.reduceat(scores.astype(np.float64) + errors, starts)
    count('native_score_blocks')
    for index, group in enumerate(groups):
        if (np.isfinite(means[index]) and np.isfinite(mean_errors[index])
                and (means[index] + mean_errors[index] < threshold
                     or minimum_upper[index] < cohesion)):
            count('screened_groups')
            continue
        yield group, *exact(group, query)
