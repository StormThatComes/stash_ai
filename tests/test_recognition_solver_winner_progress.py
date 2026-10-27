"""Regression: a winner at trial 4 must not go unexplored for 50 trials."""
from collections import Counter
from itertools import islice

import numpy as np

from tests.test_recognition_merge_best_effort import goal_sets
from tests.test_recognition_merge_solver_hybrid import prepared
from stash_ai.recognition.fragmentation_report import _open_readonly
from stash_ai.recognition.merge_goals import assess, evaluate, objective, resolve
from stash_ai.recognition.merge_report_policy import patched, proposal
from stash_ai.recognition.merge_solver_bidirectional import (
    BidirectionalSearch, expansion_candidates, relaxed_seeds, repair_candidates)
from stash_ai.recognition.merge_solver_search import BeamSearch, PREFIX


def test_alternating_lanes_expand_trial_four_without_waiting_for_warmup():
    def factory(parent, changes, family, mode):
        for i in range(1000):
            yield 'probe', {'x': i}

    search = BeamSearch(['face'], factory, 500)
    search.observe('baseline', {}, (0, 0, 0), 'baseline')
    normal = []
    for number in range(1, 51):
        if number % 2:
            family, phase, parent, _, changes = search.next()
            normal.append((number, phase, parent))
        else:
            family, phase, parent, changes = 'face', 'bidirectional', 'baseline', {'x': -number}
        search.completed(phase, parent=parent)
        search.observe(f'trial-{number}', changes, (1 if number == 4 else 0, 0, -number),
                       str(number), family=family)
    children = [row for row in normal if row[2] == 'trial-4']
    assert children[0] == (5, 'adaptive', 'trial-4')
    assert len(children) >= 8  # Previously zero; first adaptive trial was 27.
    counts = Counter(row[1] for row in normal)
    assert counts['adaptive'] > counts['baseline_exploration'] >= 5
    assert any(phase == 'adaptive' and parent != 'trial-4' for _, phase, parent in normal)


def test_external_lane_does_not_advance_normal_parent_rotation():
    def factory(parent, changes, family, mode):
        for i in range(100):
            yield 'probe', {'x': i}

    search = BeamSearch(['face'], factory, 500)
    search.observe('baseline', {}, (0, 0, 0), 'baseline')
    for name, score in [('best', 3), ('other-a', 2), ('other-b', 1)]:
        search.observe(name, {'x': score}, (score, 0, 0), name)
    parents = []
    for _ in range(20):
        _, phase, parent, _, _ = search.next()
        if phase == 'adaptive':
            parents.append(parent)
        search.completed(phase, parent=parent)
        search.completed('bidirectional', parent='other-a')
    assert parents.count('best') >= len(parents)//2
    assert {'other-a', 'other-b'} <= set(parents)


def test_disjoint_frontier_expands_beyond_its_own_safe_endpoint(tmp_path):
    config, _ = prepared(tmp_path, [[1., 0, 0, 0]], [[1., 0]])
    goals = goal_sets([['a'], ['b'], ['c']], [['d']])
    observations = dict.fromkeys('abcd', 'scene')
    status = assess(goals, observations, {}, dict(zip('abcd', 'aabd')))
    source = dict(proposals={}, options={})
    current = dict(next(relaxed_seeds(config, {})), **{'clustering.max_anchor_pose_deviation': .4})
    lane = BidirectionalSearch(config, goals, 500, source)
    lane.observe('trial-4', current, status, objective(status), 'partial')
    lane.seeds.clear()
    _, phase, parent, strategy, candidate = lane.next(6, {'trial-4': source})
    assert (phase, parent, strategy) == ('bidirectional', 'trial-4', 'frontier_expansion')
    assert candidate[PREFIX+'face_floor'] < current[PREFIX+'face_floor']
    assert candidate[PREFIX+'person_threshold'] < current[PREFIX+'person_threshold']
    assert candidate['clustering.max_anchor_pose_deviation'] == .4
    patched(config, candidate)
    # The old repair-only path can propose nothing beyond this same endpoint.
    assert all(proposal(config, p) == proposal(config, current)
               for _, p in repair_candidates(config, current, [source], current))


def test_expansion_does_not_reset_improved_gates_to_baseline_hints(tmp_path):
    config, _ = prepared(tmp_path, [[1., 0, 0, 0]], [[1., 0]])
    current = {PREFIX+'face_floor': .2, PREFIX+'person_threshold': .3,
               'clustering.max_anchor_pose_deviation': .4}
    source = dict(proposals={'hybrid': [{PREFIX+'face_floor': .75, PREFIX+'person_threshold': .25}],
                             'face': [{'clustering.max_anchor_pose_deviation': .3}]})
    candidates = [p for _, p in expansion_candidates(config, current, source)]
    assert any(p[PREFIX+'person_threshold'] == .25 for p in candidates)
    for candidate in candidates:
        assert candidate[PREFIX+'face_floor'] <= .2
        assert candidate[PREFIX+'person_threshold'] <= .3
        assert candidate['clustering.max_anchor_pose_deviation'] == .4
        patched(config, candidate)


def test_outward_joint_move_improves_real_partial_replay_without_mixing_sets(tmp_path):
    def vectors(weak):
        gram = np.full((4, 4), .05)
        gram[:3, :3] = weak
        gram[0, 1] = gram[1, 0] = .75
        np.fill_diagonal(gram, 1.)
        return np.linalg.cholesky(gram).tolist()

    config, people = prepared(tmp_path, vectors(.16), vectors(.25))
    current = next(relaxed_seeds(config, {}))
    before = config.database.read_bytes()
    variants = {'partial': patched(config, current)}
    variants.update({f'outward-{i}': patched(config, p)
                     for i, (_, p) in enumerate(islice(expansion_candidates(config, current, {}), 4))})
    with _open_readonly(config.database) as db:
        goals = resolve(db, config, [people[:3], people[3:]], by_performer=False)
        trials = evaluate(config, goals, db, variants)['trials']
    initial = trials['partial']['goal_result']
    assert initial['disjoint'] and not initial['fully_solved']
    assert initial['groups'][0]['complete_components'] == [2, 1]
    solved = trials['outward-3']['goal_result']
    assert solved['fully_solved'] and solved['disjoint']
    assert solved['groups'][0]['complete_components'] == [3]
    assert objective(solved) > objective(initial)
    assert config.database.read_bytes() == before
