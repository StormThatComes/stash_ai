"""Batched frozen-reference CPU scores, with exact scalar decision values.

Only byte-backed immutable vectors and packed read-only arrays cross the worker
boundary. SQLite, Group mutation, caches and identity decisions stay outside.
"""
from dataclasses import dataclass
from itertools import islice

from stash_ai.runtime.cancellation import current as cancellation
from stash_ai.runtime.cleanup import closing
from .compute import current, Work
from .profiling import count


@dataclass(frozen=True)
class Reference:
    identifier: str
    samples: tuple


@dataclass(frozen=True)
class Shard:
    groups: tuple
    matrix: object
    lengths: object
    starts: object
    norms: object


def _score(payload):
    """No coordinator contexts are copied into this pure numerical kernel."""
    import numpy as np
    from .clustering import _similarities
    shard, queries, threshold, cohesion = payload
    matrix = shard.matrix
    query_matrix = np.stack(queries)
    scores = matrix @ query_matrix.T
    dimension = matrix.shape[1]
    unit = np.finfo(np.float32).eps / 2
    gamma = dimension * unit / (1 - dimension * unit)
    errors = (4*gamma + 16*np.finfo(np.float64).eps) * shard.norms[:, None] * np.linalg.norm(
        query_matrix.astype(np.float64), axis=1)[None, :]
    errors += dimension*np.finfo(np.float32).tiny
    means = np.add.reduceat(scores.astype(np.float64), shard.starts, axis=0) / shard.lengths[:, None]
    mean_errors = np.add.reduceat(errors, shard.starts, axis=0) / shard.lengths[:, None]
    minimum_upper = np.minimum.reduceat(scores.astype(np.float64)+errors, shard.starts, axis=0)
    rows, screened, samples = [], 0, 0
    token = cancellation()
    for index, query in enumerate(queries):
        if token is not None:
            token.check()
        values = []
        for position, group in enumerate(shard.groups):
            if (np.isfinite(means[position, index]) and np.isfinite(mean_errors[position, index])
                    and (means[position, index]+mean_errors[position, index] < threshold
                         or minimum_upper[position, index] < cohesion)):
                screened += 1
                continue
            # Preserve the original float32 dot and float64 mean/min order.
            values.append((position, *_similarities(group, query)))
            samples += len(group.samples)
        rows.append(tuple(values))
    return tuple(rows), screened, samples


class FrozenScores:
    """One fixed reference generation, reused for bounded query batches."""
    def __init__(self, groups):
        self.groups = tuple(groups)
        self.references = tuple(Reference(g.identifier, g.samples if isinstance(g.samples, tuple)
                                          else tuple(g.samples)) for g in self.groups)
        self.generation = tuple((g.identifier, tuple(id(a) for a in g.samples)) for g in self.references)

    def valid(self):
        return all(g.identifier == r.identifier and len(g.samples) == len(r.samples)
                   and all(a is b for a, b in zip(g.samples, r.samples))
                   for g, r in zip(self.groups, self.references))

    def eligible(self):
        import numpy as np
        pool = current()
        return bool(pool is not None and pool.workers > 1 and len(self.groups) >= 16
                    and all(isinstance(a, np.ndarray) and a.dtype == np.float32
                            and a.ndim == 1 and not a.flags.writeable and isinstance(a.base, bytes)
                            for g in self.references for a in g.samples))

    def score(self, queries, threshold, cohesion=float('-inf')):
        """Return sparse exact contributions, or None for the serial fallback."""
        if not queries or not self.eligible():
            pool = current()
            if pool is not None:
                pool.fallback('disabled' if pool.workers == 1 else
                              'small_catalogue' if len(self.groups) < 16 else 'unsupported_layout')
            return None
        import numpy as np
        from .scoring import _prepared
        pool = current()
        if not self.valid():
            raise RuntimeError('Recognition reference generation changed during scoring.')
        # Aggregated sparse Python results are bounded even when every group is
        # a finalist. The caller also caps query count before preparing inputs.
        if len(queries)*len(self.groups)*192 > pool.limit//4:
            pool.fallback('result_admission')
            return None
        target = max(1024, min(524288, pool.limit//(max(2, pool.workers)*12)))
        blocks, block, size = [], [], 0
        for group in self.references:
            required = sum(a.nbytes for a in group.samples)
            if block and size+required > target:
                blocks.append(tuple(block)); block, size = [], 0
            block.append(group); size += required
        if block:
            blocks.append(tuple(block))
        # Avoid starting a compute pool for one small shard.
        if len(blocks) < 2:
            pool.fallback('single_shard')
            return None
        dimension = queries[0].shape
        if any(q.shape != dimension for q in queries):
            pool.fallback('mixed_dimensions')
            return None
        for block in blocks:
            sample_count = sum(len(g.samples) for g in block)
            raw = sum(a.nbytes for g in block for a in g.samples)
            if (any(not g.samples or any(a.shape != dimension for a in g.samples) for g in block)
                    or 6*raw + sample_count*len(queries)*32 + len(block)*len(queries)*192 + 4096 > pool.limit):
                pool.fallback('layout_or_admission')
                return None
        def jobs():
            for ordinal, block in enumerate(blocks):
                with pool.preparing('packing'):
                    prepared = _prepared(block, queries[0])
                if prepared is None:
                    raise RuntimeError('Invalid immutable recognition score shard.')
                matrix, lengths, starts, norms = prepared
                for array in prepared:
                    array.flags.writeable = False
                shard = Shard(block, matrix, lengths, starts, norms)
                size = 6*matrix.nbytes + len(matrix)*len(queries)*32 + len(block)*len(queries)*192 + 4096
                yield Work(ordinal, self.generation, (shard, queries, threshold, cohesion), size)
        result = [[] for _ in queries]
        offset = 0
        with closing(pool.ordered(_score, jobs()), name='close parallel score results') as stream:
            for completed in stream:
                if completed.generation != self.generation:
                    raise RuntimeError('Stale recognition reference score result.')
                rows, screened, samples = completed.value
                for index, values in enumerate(rows):
                    result[index].extend((offset+position, mean, low) for position, mean, low in values)
                count('parallel_score_blocks')
                count('screened_groups', screened)
                count('group_comparisons', sum(len(row) for row in rows))
                count('sample_comparisons', samples)
                offset += len(blocks[completed.ordinal])
        if not self.valid():
            raise RuntimeError('Recognition reference generation changed during scoring.')
        return result


def scored_rows(rows, groups, threshold, cohesion=float('-inf')):
    """Yield (authoritative row, frozen contributions or None) in original order.

    Bad embedding errors are deferred to the caller's original vector() use.
    Only cold-replay numerical work uses this helper; it does not read SQLite.
    """
    from .clustering import vector
    pool = current()
    if pool is None or pool.workers == 1 or len(groups) < 16:
        if pool is not None:
            pool.fallback('disabled' if pool.workers == 1 else 'small_catalogue')
        for row in rows:
            yield row, None
        return
    plan = FrozenScores(groups)
    if not plan.eligible():
        pool.fallback('unsupported_layout')
        for row in rows:
            yield row, None
        return
    batch_size = max(1, min(16, pool.limit//max(1, len(groups)*192*8)))
    iterator = iter(rows)
    while batch := list(islice(iterator, batch_size)):
        queries, indices = [], []
        with pool.preparing('vectors'):
            for index, row in enumerate(batch):
                try:
                    query = vector(row)
                except ValueError:
                    continue  # Raised only when this row is consumed by the caller.
                indices.append(index); queries.append(query)
        values = plan.score(tuple(queries), threshold, cohesion)
        mapped = dict(zip(indices, values)) if values is not None else {}
        from .evidence_cache import current as current_evidence
        cache = current_evidence()
        pool.record('score_rows', computed=len(mapped))
        consumed = 0
        try:
            for index, row in enumerate(batch):
                frozen = mapped.get(index)
                if cache is not None and frozen is not None:
                    for position, mean, low in frozen:
                        cache.remember_scores(plan.groups[position], row['id'], (mean, low))
                if frozen is not None:
                    consumed += 1
                    pool.record('score_rows', consumed=1)
                yield row, frozen
        finally:
            pool.record('score_rows', discarded=len(mapped)-consumed)


def contributions(groups, query, frozen, fixed_count, threshold, cohesion=float('-inf'), *, exact=None):
    """Combine frozen known groups with fresh coordinator-owned local groups."""
    from .scoring import comparisons
    if frozen is None:
        yield from comparisons(groups, query, threshold, cohesion, exact=exact)
        return
    for position, mean, low in frozen:
        yield groups[position], mean, low
    yield from comparisons(groups[fixed_count:], query, threshold, cohesion, exact=exact)
