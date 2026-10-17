"""Demanded immutable merge proofs; only the coordinator owns their caches."""
from itertools import islice, product

from stash_ai.runtime.cancellation import current as cancellation
from stash_ai.runtime.cleanup import closing
from .compute import current, Work
from .evidence_cache import current as evidence, MISSING
from .merge_cache import cached_cross, cross_identity, find_cross, save_cross
from .parallel_scores import FrozenScores
from .profiling import count


def cross_scores(first, second, threshold, cohesion, *, counts=None):
    """Keep the original directional scalar dot/mean/min reduction order."""
    from .clustering import _similarities
    total, minimum = 0., 1.
    token = cancellation()
    for sample in first.samples:
        if token is not None:
            token.check()
        mean, low = _similarities(second, sample)
        if counts is not None:
            counts[0] += 1
            counts[1] += len(second.samples)
        minimum = min(minimum, low)
        if minimum < cohesion:
            return None
        total += mean
    mean = total / len(first.samples)
    return None if mean < threshold else {'mean': mean, 'minimum': minimum}


def _cross(payload):
    first, second, threshold, cohesion = payload
    counts = [0, 0]
    return cross_scores(first, second, threshold, cohesion, counts=counts), *counts


def _immutable(plan):
    import numpy as np
    samples = [a for g in plan.references for a in g.samples]
    return bool(samples and all(g.samples for g in plan.references)
                and all(isinstance(a, np.ndarray) and a.dtype == np.float32
                        and a.ndim == 1 and a.shape == samples[0].shape
                        and not a.flags.writeable and isinstance(a.base, bytes) for a in samples))


def cross_pairs(first, second, config, serial):
    """Retire a small demanded batch before exposing any proof to the caller.

    An incompatible pair can stop later batches without abandoning an active
    CPU stream. Never schedule a catalogue-wide all-pairs graph. Negative
    proofs use the same directional, weak-owner cache as the scalar path.
    """
    pool = current()
    if pool is None or pool.workers == 1 or len(first)*len(second) < 2:
        for left, right in product(first, second):
            yield cached_cross(left, right, config, serial)
        return
    plan = FrozenScores(tuple(first) + tuple(second))
    if not _immutable(plan):
        for left, right in product(first, second):
            yield cached_cross(left, right, config, serial)
        return
    cache = evidence()
    references = dict(zip(map(id, plan.groups), plan.references))
    threshold = config.data['recognition']['match_threshold']
    cohesion = config.data['clustering']['cohesion_threshold']
    pairs = iter(product(first, second))
    while batch := list(islice(pairs, pool.workers*2)):
        values, identities, jobs, demanded = [], [], [], {}
        for left, right in batch:
            identity = cross_identity(left, right, config)
            identities.append(identity)
            found = find_cross(cache, identity)
            if found is not MISSING:
                values.append((False, found))
                continue
            # Repeated demands in a batch share one immutable proof.
            key = identity[0]
            if key not in demanded:
                a, b = references[id(left)], references[id(right)]
                size = 4096 + sum(s.nbytes+128 for g in (a, b) for s in g.samples)
                if size > pool.limit:
                    values.append((False, cached_cross(left, right, config, serial)))
                    continue
                ordinal = len(jobs)
                demanded[key] = ordinal
                jobs.append(Work(ordinal, (plan.generation, threshold, cohesion),
                                 (a, b, threshold, cohesion), size))
            values.append((True, demanded[key]))
        completed = []
        with closing(pool.ordered(_cross, jobs), name='close parallel constituent proofs') as stream:
            for result in stream:
                scores, groups, samples = result.value
                completed.append(scores)
                count('group_comparisons', groups)
                count('sample_comparisons', samples)
        if not plan.valid():
            raise RuntimeError('Recognition constituent references changed during scoring.')
        count('parallel_constituent_pairs', len(jobs))
        resolved = []
        for identity, (pending, value) in zip(identities, values):
            if pending:
                value = completed[value]
                save_cross(cache, identity, value)
            resolved.append(value)
        yield from resolved


def _matches(payload):
    """Pure exact constituent predicates, with no component/identity mutation."""
    from .clustering import _similarities
    components, queries, threshold, cohesion = payload
    token = cancellation()
    rows, groups, samples = [], 0, 0
    for query in queries:
        mask = 0
        for position, component in enumerate(components):
            matched = True
            for group in component:
                if token is not None:
                    token.check()
                mean, low = _similarities(group, query)
                groups += 1
                samples += len(group.samples)
                if mean < threshold or low < cohesion:
                    matched = False
                    break
            if matched:
                mask |= 1 << position
        rows.append(mask)
    return tuple(rows), groups, samples


def matching_rows(rows, components, config):
    """Yield demanded evidence rows and exact component masks in source order.

    SQLite and bounded batch preparation stay on the coordinator. Invalid
    vectors/dimensions retain the caller's original validation/short circuit.
    All jobs in a batch retire before yielding: enough canonical votes or a
    co-occurrence proof may then stop the stream without cancelling the pool.
    These ephemeral masks are never cached as mutable union-level booleans.
    """
    pool = current()
    if pool is None or pool.workers == 1:
        for row in rows:
            yield row, None
        return
    plan = FrozenScores(tuple(group for component in components for group in component))
    if not _immutable(plan):
        for row in rows:
            yield row, None
        return
    from .clustering import vector
    offset, frozen = 0, []
    for component in components:
        frozen.append(plan.references[offset:offset+len(component)])
        offset += len(component)
    frozen = tuple(frozen)
    threshold = config.data['recognition']['match_threshold']
    cohesion = config.data['clustering']['cohesion_threshold']
    dimension = plan.references[0].samples[0].shape
    raw = sum(a.nbytes+128 for g in plan.references for a in g.samples)
    scratch = max(len(g.samples) for g in plan.references)*16 + 4096
    per_query = dimension[0]*4+512
    # Leave room for at least two concurrent blocks under byte admission.
    block_size = min(8, (pool.limit//2-raw-scratch)//per_query)
    if block_size < 1:
        for row in rows:
            yield row, None
        return
    batch_size = min(256, max(2, pool.workers)*block_size,
                     max(1, pool.limit//(per_query*2)))
    iterator = iter(rows)
    while batch := list(islice(iterator, batch_size)):
        if not plan.valid():
            raise RuntimeError('Recognition constituent references changed during scoring.')
        queries, indices = [], []
        for index, row in enumerate(batch):
            try:
                query = vector(row)
            except ValueError:
                continue
            if query.shape == dimension:
                queries.append(query)
                indices.append(index)
        mapped = {}
        if len(queries) > block_size:
            jobs = (Work(ordinal, (plan.generation, threshold, cohesion),
                         (frozen, tuple(queries[start:start+block_size]), threshold, cohesion),
                         raw+scratch+len(queries[start:start+block_size])*per_query)
                    for ordinal, start in enumerate(range(0, len(queries), block_size)))
            masks = []
            with closing(pool.ordered(_matches, jobs), name='close parallel merge evidence') as stream:
                for result in stream:
                    values, groups, samples = result.value
                    masks.extend(values)
                    count('parallel_merge_evidence_blocks')
                    count('group_comparisons', groups)
                    count('sample_comparisons', samples)
            if not plan.valid():
                raise RuntimeError('Recognition constituent references changed during scoring.')
            mapped = dict(zip(indices, masks))
        for index, row in enumerate(batch):
            yield row, mapped.get(index)
