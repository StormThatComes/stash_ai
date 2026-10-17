"""Parallel merge proofs retain scalar direction, versions and ordered decisions."""
from threading import Event, get_ident
import gc
import weakref

import numpy as np
import pytest

from stash_ai.recognition import compute, identity_merges as merges, parallel_merges
from stash_ai.recognition.clustering import Group
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.evidence_cache import evidence_scope, MISSING
from stash_ai.recognition.merge_cache import cross_identity, find_cross
from tests.test_recognition_clustering import setup
from tests.test_recognition_compute import configured
from tests.test_recognition_identity_merges import changed
from tests.test_recognition_score_blocks import immutable


@pytest.fixture(autouse=True)
def cpus(monkeypatch):
    monkeypatch.setattr(compute, 'available_cpus', lambda: 24)


def references():
    rng = np.random.default_rng(674)
    result = []
    for index in range(8):
        values = rng.normal(0, .015, (5, 128)).astype('float32')
        values[:, 0] += 1
        values /= np.linalg.norm(values, axis=1)[:, None]
        result.append(Group(str(index), tuple(immutable(v) for v in values), True))
    return result[:4], result[4:]


@pytest.mark.parametrize('boundary', ['pass', 'mean', 'cohesion', 'reject'])
def test_pair_proofs_match_directional_scalar_boundaries_with_reverse_completion(tmp_path, monkeypatch, boundary):
    first, second = references()
    config = setup(tmp_path)
    score = merges._cross(first[0], second[0], config)
    threshold = {'pass': .65, 'mean': score['mean'], 'cohesion': .65, 'reject': 1.}[boundary]
    cohesion = score['minimum'] if boundary == 'cohesion' else .6
    config = changed(config, recognition={'match_threshold': threshold}, clustering={'cohesion_threshold': cohesion})
    expected = merges._compatible(first, second, config)
    arrived, original, threads = Event(), parallel_merges._cross, set()
    owner = get_ident()
    def reordered(payload):
        threads.add(get_ident())
        if payload[0].identifier == '0' and payload[1].identifier == '4':
            assert arrived.wait(5)
        else:
            arrived.set()
        return original(payload)
    monkeypatch.setattr(parallel_merges, '_cross', reordered)
    with compute.compute_scope(configured(rebuild_workers=4, rebuild_queue_bytes=65536)) as pool:
        assert merges._compatible(first, second, config) == expected
        assert pool.submitted >= 2
        assert pool.peak_bytes <= pool.limit
        assert not pool.pending  # An early incompatible proof never poisons the pool.
        assert merges._compatible(first, second, config) == expected
    assert len(threads) >= 2 and owner not in threads


def test_cached_pairs_are_directional_versioned_and_do_not_retain_old_samples(tmp_path):
    first, second = references()
    config = setup(tmp_path)
    parallel = changed(config, execution={'rebuild_workers': 4, 'rebuild_queue_bytes': 65536})
    with FaceDatabase(config.database) as state, evidence_scope(state.connection, config) as cache:
        with compute.compute_scope(parallel) as pool:
            expected = merges._compatible(first, second, parallel)
            submitted = pool.submitted
            assert submitted == 16
            assert merges._compatible(first, second, parallel) == expected
            assert pool.submitted == submitted
            merges._compatible(second, first, parallel)
            assert pool.submitted == submitted*2
            old = weakref.ref(first[0].samples[0])
            first[0].samples = tuple(immutable(a) for a in first[1].samples)
            merges._compatible(first, second, parallel)
            assert pool.submitted == submitted*2+4
            stricter = changed(parallel, recognition={'match_threshold': .99999})
            assert merges._compatible(first, second, stricter) is None
            assert pool.submitted > submitted*2+4
        gc.collect()
        assert old() is None
        assert find_cross(cache, cross_identity(first[1], second[0], config)) is not MISSING


@pytest.mark.parametrize('failure', [RuntimeError, KeyboardInterrupt])
def test_parallel_pair_failure_does_not_import_partial_cache_proofs(tmp_path, monkeypatch, failure):
    first, second = references()
    config = setup(tmp_path)
    def fail(payload):
        raise failure('pair worker failed')
    monkeypatch.setattr(parallel_merges, '_cross', fail)
    with FaceDatabase(config.database) as state, evidence_scope(state.connection, config) as cache:
        with compute.compute_scope(configured(rebuild_workers=4, rebuild_queue_bytes=65536)):
            with pytest.raises(failure, match='pair worker failed'):
                merges._compatible(first, second, config)
        assert find_cross(cache, cross_identity(first[0], second[0], config)) is MISSING


def test_reference_change_rejects_completed_pair_proofs(tmp_path, monkeypatch):
    first, second = references()
    config = setup(tmp_path)
    original = parallel_merges._cross
    changed_once = Event()
    def mutate(payload):
        if not changed_once.is_set():
            changed_once.set()
            first[0].samples = first[1].samples
        return original(payload)
    monkeypatch.setattr(parallel_merges, '_cross', mutate)
    with FaceDatabase(config.database) as state, evidence_scope(state.connection, config) as cache:
        with compute.compute_scope(configured(rebuild_workers=4, rebuild_queue_bytes=65536)):
            with pytest.raises(RuntimeError, match='constituent references changed'):
                merges._compatible(first, second, config)
        assert not any(key[0] == 'cross' for key in cache.cache.entries)


def test_duplicate_demands_share_one_inflight_pair(tmp_path):
    first, second = references()
    config = setup(tmp_path)
    with compute.compute_scope(configured(rebuild_workers=4, rebuild_queue_bytes=65536)) as pool:
        actual = list(parallel_merges.cross_pairs([first[0]]*3, [second[0]]*2, config, merges._cross))
        assert actual == [merges._cross(first[0], second[0], config)]*6
        assert pool.submitted == 1


def test_negative_proofs_stop_later_batches_and_are_reused(tmp_path):
    first, second = references()
    config = changed(setup(tmp_path), recognition={'match_threshold': 1.})
    with FaceDatabase(config.database) as state, evidence_scope(state.connection, config):
        with compute.compute_scope(configured(rebuild_workers=4, rebuild_queue_bytes=65536)) as pool:
            assert merges._compatible(first, second, config) is None
            assert pool.submitted == 8  # Only the bounded demanded prefix.
            assert merges._compatible(first, second, config) is None
            assert pool.submitted == 8


def test_mutable_or_mixed_dimension_references_keep_scalar_failure_order(tmp_path):
    first, second = references()
    config = setup(tmp_path)
    first[0].samples = [a.copy() for a in first[0].samples]
    with compute.compute_scope(configured(rebuild_workers=4, rebuild_queue_bytes=65536)) as pool:
        expected = merges._compatible(first, second, config)
        assert expected is not None and pool.submitted == 0
        second[1].samples = (immutable(np.ones(4, dtype='float32')/2),)
        with pytest.raises(ValueError, match='mixed embedding dimensions'):
            merges._compatible(first, second, config)
        assert pool.submitted == 0
