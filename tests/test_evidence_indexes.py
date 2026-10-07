"""Compare optimized queries with direct evidence scanners."""
from copy import deepcopy
import random

from stash_ai.evidence.interval_geometry import union_intervals
from stash_ai.evidence.observations import ObservationIndex
from stash_ai.evidence.ranges import Coverage
from stash_ai.evidence.temporal.policy import TemporalPolicy
from stash_ai.evidence.timeline import Series


def test_coverage_matches_linear_queries():
    rng = random.Random(48)
    intervals = [[i * 3, i * 3 + 1] for i in range(200)]
    coverage = Coverage(intervals)
    for _ in range(1000):
        start = rng.uniform(-1, 605)
        end = start + rng.uniform(0, 5)
        assert coverage.contains(start, end) == any(a <= start and b >= end for a, b in intervals)
        assert coverage.overlaps(start, end) == any(a < end and b > start for a, b in intervals)


def test_dedup_preserves_exact_equality_and_borrowed_proofs(monkeypatch):
    import stash_ai.evidence.interval_geometry as geometry
    # Deliberate hash collisions must never remove distinct proofs.
    monkeypatch.setattr(geometry, '_content_hash', lambda value, memo: 7)
    proof = {'shared': [1, 2]}
    values = [dict(tag='core', source='wd', start=0, end=2, score=.9, hits=2,
                   meta={'index': i}, proof=proof) for i in range(30)]
    inputs = values + [deepcopy(value) for value in reversed(values)]
    result = union_intervals(inputs, borrow=True)[0]
    assert result['contributors'] == values
    assert all(item is original for item, original in zip(result['contributors'], values))
    copied = union_intervals(inputs)[0]['contributors']
    assert copied == values and copied[0] is not values[0]
    assert copied[0]['proof'] is copied[1]['proof']


def test_range_max_matches_scores_across_block_boundaries():
    rng = random.Random(49)
    series = Series()
    for i in range(1000):
        series.append(i, i, i + 1, rng.random())
    for _ in range(1000):
        a = rng.randrange(1000)
        b = rng.randrange(a + 1, 1001)
        assert series.max_score(a, b) == max(series.scores[a:b])


def test_sparse_tag_index_matches_reference_with_source_filters():
    rng = random.Random(50)
    events = [dict(start=float(i), end=float(i + 1), extras={
        f'tag_{rng.randrange(90)}': {'wd': rng.random(), 'dtq': rng.random()}
        for _ in range(3)}) for i in range(500)]
    policy = TemporalPolicy()
    with_index = ObservationIndex(events, {'wd': 'w', 'dtq': 'd'}, 'group')
    try:
        for _ in range(200):
            first = rng.randrange(500)
            last = rng.randrange(first + 1, 501)
            allowed = None if rng.randrange(2) else frozenset({'wd'})
            assert with_index.support(first, last, policy, allowed) == with_index.scan_support(
                first, last, last, policy, allowed, start=first)
    finally:
        with_index.close()
