import copy
from dataclasses import replace

import numpy as np
import pytest

from stash_ai.recognition.merge_solver_measurements import reference_stats
from stash_ai.recognition.merge_solver_options import compact
from stash_ai.recognition.merge_goal_index import prepared
from stash_ai.recognition.config import plain
from stash_ai.recognition.merge_solver import solve
from stash_ai.recognition.merge_goals import objective
from stash_ai.recognition.fragmentation_report import _open_readonly
from benchmarks.benchmark_merge_solver import fixture, CASES
from tests.test_recognition_merge_identity_policy import member, goals, assignments, status


def immutable(values):
    return np.frombuffer(np.asarray(values, dtype='<f4').tobytes(), dtype='<f4')


@pytest.mark.parametrize('budget', [512, 4096, 1048576])
def test_block_statistics_match_dense_reference_and_tie_order(budget):
    rng = np.random.default_rng(42)
    a, b = rng.normal(size=(23, 8)), rng.normal(size=(31, 8))
    expected = a @ b.T
    result = reference_stats(a, b, max_bytes=budget)
    assert result[:3] == pytest.approx((expected.min(), expected.mean(), expected.max()), abs=1e-14)
    assert result[3] == tuple(np.unravel_index(expected.argmin(), expected.shape))
    assert reference_stats(np.ones((23, 8)), np.ones((31, 8)), max_bytes=budget)[3] == (0, 0)


def test_immutable_measurements_reuse_cache_but_mutable_content_does_not(tmp_path, monkeypatch):
    from stash_ai.recognition.run_cache import scope
    config, _ = fixture(tmp_path, 'face')
    a, b = [immutable([1., 0.])], [immutable([.6, .8])]
    with scope(config):
        expected = reference_stats(a, b)
        with monkeypatch.context() as patch:
            patch.setattr(np, 'asarray', lambda *a, **k: pytest.fail('recomputed immutable measurement'))
            assert reference_stats(a, b) == expected
        mutable = b[0].copy()
        first = reference_stats(a, [mutable])
        mutable *= -1
        assert reference_stats(a, [mutable])[0] == -first[0]


def test_compiled_goal_votes_match_uncached_including_mutated_prediction_maps():
    goal = goals([member('a', core=4, soft=3), member('b', core=3)], [member('c')])
    original = copy.deepcopy(goal)
    predicted = assignments(goal)
    changes = [dict(predicted), dict.fromkeys(predicted, 'one'),
               {oid: None if i % 3 else 'one' for i, oid in enumerate(predicted)}]
    expected = [status(goal, p) for p in changes]
    with prepared(goal):
        for p, result in zip(changes, expected):
            predicted.clear(); predicted.update(p)
            assert status(goal, predicted) == result
    assert goal == original


def test_option_pool_bounds_history_and_preserves_fresh_and_live_boundaries(tmp_path):
    config, _ = fixture(tmp_path, 'face')
    key = 'recognition.match_threshold'
    old = {key: [i/10000 for i in range(10000)]}
    fresh = {key: [.431237, .431239, .967123]}
    result = compact(config, old, fresh, parents=[{key:.654321}], limit=64)[key]
    assert len(result) <= 64
    assert set(fresh[key]) <= set(result)
    assert {.654321, .6543, .6544, 0., .9999, .99} <= set(result)


def test_no_compatible_scans_skip_measurements(tmp_path, monkeypatch):
    from tests.test_recognition_merge_solver_inputs import supersede
    config, goal = fixture(tmp_path, 'query')
    for scene in ('1', '2'):
        supersede(config, scene, reason='policy')
    monkeypatch.setattr('stash_ai.recognition.merge_solver.measured_options',
                        lambda *a, **k: pytest.fail('measured unavailable inputs'))
    with _open_readonly(config.database) as db:
        assert solve(config, goal, db, maximum=3)['status'] == 'no_compatible_scans'


@pytest.mark.parametrize('case', CASES)
def test_full_replay_matches_dense_uncached_and_uncompiled_path(tmp_path, monkeypatch, case):
    config, goal = fixture(tmp_path, case)
    with _open_readonly(config.database) as db:
        optimized = solve(config, goal, db, maximum=8)
    def dense(a, b, **kwargs):
        matrix = np.asarray(a, dtype='float64') @ np.asarray(b, dtype='float64').T
        return float(matrix.min()), float(matrix.mean()), float(matrix.max()), tuple(np.unravel_index(matrix.argmin(), matrix.shape))
    monkeypatch.setattr('stash_ai.recognition.merge_solver_measurements.reference_stats', dense)
    monkeypatch.setattr('stash_ai.recognition.merge_goal_index.indexed_votes', lambda *a: None)
    data = plain(config.data)
    data['execution'].update(rebuild_cache_bytes=0, rebuild_matrix_cache_bytes=0)
    with _open_readonly(config.database) as db:
        ordinary = solve(replace(config, data=data), goal, db, maximum=8)
    assert optimized['selected_trial'] == ordinary['selected_trial']
    assert optimized['best_result']['changes'] == ordinary['best_result']['changes']
    assert [objective(t['goal_result']) for t in optimized['simulation']['trials'].values()] == [
        objective(t['goal_result']) for t in ordinary['simulation']['trials'].values()]
