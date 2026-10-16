import numpy as np
import pytest

from stash_ai.recognition.clustering import Group, _similarities
from stash_ai.recognition.scoring import comparisons, score_scope, _blocks


def immutable(value):
    return np.frombuffer(np.asarray(value, dtype='<f4').tobytes(), dtype='<f4')


def groups(dimensions):
    rng = np.random.default_rng(2701)
    query = rng.normal(size=dimensions).astype('float32')
    query /= np.linalg.norm(query)
    result = []
    for index in range(64):
        samples = []
        for _ in range(4):
            value = (query + rng.normal(scale=.01, size=dimensions) if index < 3
                     else rng.normal(size=dimensions)).astype('float32')
            value /= np.linalg.norm(value)
            samples.append(immutable(value))
        result.append(Group(str(index), samples, True))
    return result, immutable(query)


@pytest.mark.parametrize('dimensions', [128, 512, 1024])
@pytest.mark.parametrize('threshold,cohesion', [(.0, .0), (.65, .6), (.99, .9)])
def test_native_screen_never_loses_exact_candidates_and_keeps_scalar_scores(dimensions, threshold, cohesion):
    references, query = groups(dimensions)
    expected = {group.identifier: _similarities(group, query) for group in references}
    with score_scope():
        for _ in range(2):
            actual = {group.identifier: (mean, low) for group, mean, low in comparisons(
                references, query, threshold, cohesion)}
            for identifier, (mean, low) in expected.items():
                if mean >= threshold and low >= cohesion:
                    assert actual[identifier] == (mean, low)
            assert all(value == expected[identifier] for identifier, value in actual.items())
        assert _blocks.get()['entries']


def test_exact_threshold_boundaries_and_replaced_representatives_are_not_screened():
    references, query = groups(128)
    mean, low = _similarities(references[0], query)
    with score_scope():
        for threshold in (mean-1e-12, mean, mean+1e-12):
            result = {group.identifier: (m, l) for group, m, l in comparisons(references, query, threshold, low)}
            if threshold <= mean:
                assert result['0'] == (mean, low)
        references[0].samples[1] = immutable(-query)
        expected = _similarities(references[0], query)
        result = {group.identifier: (m, l) for group, m, l in comparisons(references, query)}
        assert result['0'] == expected


def test_mutable_inputs_are_not_cached_and_tiny_cache_does_not_change_results():
    references, query = groups(128)
    references[0].samples = [value.copy() for value in references[0].samples]
    with score_scope():
        _blocks.get()['limit'] = 1
        list(comparisons(references, query))
        references[0].samples[0][:] = -query
        actual = {g.identifier: (m, l) for g, m, l in comparisons(references, query)}
        assert actual['0'] == _similarities(references[0], query)
        assert not _blocks.get()['entries']
