"""Cross a joint-gate plateau, retain an overmerge, and repair it with real replay."""
import json

import numpy as np
import pytest

from tests.test_recognition_merge_solver_hybrid import prepared
from tests.test_recognition_merge_best_effort import goal_sets
from stash_ai.recognition.config import load
from stash_ai.recognition.fragmentation_report import _open_readonly
from stash_ai.recognition.merge_goals import assess, evaluate, objective, resolve
from stash_ai.recognition.merge_report_policy import patched
from stash_ai.recognition.merge_solver import solve
from stash_ai.recognition.merge_solver_bidirectional import BidirectionalSearch, relaxed_seeds, repair_candidates
from stash_ai.recognition.merge_solver_checkpoint import winner_file
from stash_ai.recognition.merge_solver_search import BeamSearch, PREFIX


def plateau(tmp_path, monkeypatch, size=4):
    def vectors(within, between):
        gram = np.full((size, size), between)
        gram[:size-2, :size-2] = within
        gram[size-2:, size-2:] = within
        np.fill_diagonal(gram, 1.)
        return np.linalg.cholesky(gram).tolist()

    config, people = prepared(tmp_path, vectors(.65, .4), vectors(.75, .5))
    with _open_readonly(config.database) as db:
        goals = resolve(db, config, [people[:-2], people[-2:]], by_performer=False)
    # Model sparse nomination feedback: measured values remain available but
    # no ready-made joint patch is nominated. Production decisions are intact.
    from stash_ai.recognition import merge_solver
    from stash_ai.recognition.merge_solver_feedback import QueryFeedback
    original, finish = merge_solver.measured_options, QueryFeedback.finish

    def sparse(*args, **kwargs):
        result = original(*args, **kwargs)
        kwargs['hybrid']['proposals'] = []
        kwargs['hybrid']['pairs'] = []
        return result

    def sparse_queries(self, current):
        result = finish(self, current)
        result['proposals'] = []
        return result

    monkeypatch.setattr(merge_solver, 'measured_options', sparse)
    # The independent face fallback is another ready-made measured route; this
    # fixture intentionally suppresses such routes to exercise frontier repair.
    monkeypatch.setattr('stash_ai.recognition.merge_solver_feedback.face_fallbacks', lambda *a, **k: [])
    monkeypatch.setattr(QueryFeedback, 'finish', sparse_queries)
    return config, goals


def test_eighteen_performers_escape_plateau_via_overmerge_and_repair(tmp_path, monkeypatch):
    from stash_ai.recognition.merge_goal_output import export
    config, goals = plateau(tmp_path, monkeypatch, size=20)
    before = config.database.read_bytes()
    with _open_readonly(config.database) as db:
        result = dict(goals, **solve(config, goals, db, targeted_trials=0, feasibility_trials=0, maximum=5))
    trials = result['simulation']['trials']
    assert list(trials) == ['baseline', 'trial-1', 'trial-2', 'trial-3', 'trial-4']
    assert trials['trial-1']['goal_result']['groups'][0]['largest_complete_performers'] == 1
    overmerge = trials['trial-2']
    assert overmerge['search_strategy'] == 'relaxed_seed'
    assert not overmerge['goal_result']['disjoint']
    assert overmerge['goal_result']['groups'][0]['largest_complete_performers'] == 18
    assert overmerge['best_trial'] != 'trial-2'
    repaired = trials['trial-4']
    assert repaired['search_phase'] == 'bidirectional'
    assert repaired['search_strategy'] == 'boundary_bridge' and repaired['search_parent'] == 'trial-2'
    assert repaired['goal_result']['fully_solved']
    assert [g['complete_components'] for g in repaired['goal_result']['groups']] == [[18], [2]]
    assert all(trials[n]['search_phase'] != 'bidirectional' for n in ('trial-1', 'trial-3'))
    assert result['selected_trial'] == 'trial-4'
    export(config, result, tmp_path/'result.json', tmp_path/'winner.yml')
    with _open_readonly(config.database) as db:
        assert evaluate(load(tmp_path/'winner.yml'), goals, db)['trials']['baseline']['goal_result']['fully_solved']
    with monkeypatch.context() as local:
        local.setattr(BidirectionalSearch, 'next', lambda *args, **kwargs: None)
        with _open_readonly(config.database) as db:
            normal_only = solve(config, goals, db, targeted_trials=0, feasibility_trials=0, maximum=5)
        assert all(t['goal_result']['groups'][0]['largest_complete_performers'] == 1
                   for t in normal_only['simulation']['trials'].values())
    assert before == config.database.read_bytes()
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('mode', ['full', 'diff'])
def test_resume_repairs_completed_overmerge_and_ctrl_c_keeps_safe_yaml(tmp_path, monkeypatch, mode):
    import stash_ai.recognition.replay_simulation as replay
    config, goals = plateau(tmp_path, monkeypatch)
    checkpoint, destination = tmp_path/'journal.json', tmp_path/'best.yml'

    def interrupt(name, stage, item):
        if name == 'trial-3' and stage == 'start':
            raise KeyboardInterrupt

    with _open_readonly(config.database) as db:
        first = solve(config, goals, db, targeted_trials=0, feasibility_trials=0, maximum=6, checkpoint=checkpoint,
            winner_writer=winner_file(destination), progress=interrupt, yaml_mode=mode)
    assert first['interrupted'] and first['selected_trial'] != 'trial-2'
    assert not first['simulation']['trials']['trial-2']['goal_result']['disjoint']
    assert f"best completed trial {first['selected_trial']} " in destination.read_text()
    assert 'best completed trial trial-2 ' not in destination.read_text()
    original, calls = replay.rebuild_selected, []

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(replay, 'rebuild_selected', counting)
    with _open_readonly(config.database) as db:
        resumed = solve(config, goals, db, targeted_trials=0, feasibility_trials=0, maximum=6, checkpoint=checkpoint, resume=True,
            winner_writer=winner_file(destination, resume=True), yaml_mode=mode)
    assert resumed['search']['restored_trials'] == 3 and len(calls) == 3
    assert resumed['best_result']['fully_solved']
    assert resumed['simulation']['trials']['trial-4']['search_parent'] == 'trial-2'
    assert f"best completed trial {resumed['selected_trial']} " in destination.read_text()
    calls.clear()
    with _open_readonly(config.database) as db:
        repeated = solve(config, goals, db, targeted_trials=0, feasibility_trials=0, maximum=6, checkpoint=checkpoint, resume=True,
            winner_writer=winner_file(destination, resume=True), yaml_mode=mode)
    assert not calls and repeated['simulation']['trials'] == resumed['simulation']['trials']


def test_overmerge_remains_expandable_after_normal_beam_prunes_it(tmp_path):
    config, _ = prepared(tmp_path, [[1., 0, 0, 0]], [[1., 0]])
    goals = goal_sets([['a'], ['b']], [['c'], ['d']])
    observations = dict.fromkeys('abcd', 'scene')
    safe = assess(goals, observations, {}, dict(zip('abcd', 'abcd')))
    mixed = assess(goals, observations, {}, dict.fromkeys('abcd', 'one'))
    feedback = dict(proposals={}, query={})
    lane = BidirectionalSearch(config, goals, 500, feedback)
    normal = BeamSearch(['hybrid'], lambda *a: iter(()), 500)
    for name, changes, status, outcome in [
        ('baseline', {}, safe, 'separate'), ('mixed', {PREFIX+'face_floor': .2}, mixed, 'mixed'),
        *[(f'safe-{i}', {PREFIX+'face_floor': .81+i/100}, safe, f'separate-{i}') for i in range(7)]]:
        rank = objective(status) + (0, 0)
        lane.observe(name, changes, status, rank, outcome)
        normal.observe(name, changes, rank, outcome)
    assert 'mixed' not in normal.beam and 'mixed' in lane.frontier
    assert len(lane.frontier) <= lane.width and lane.budget <= 64
    assert objective(safe) > objective(mixed)  # Export rank is unchanged.


@pytest.mark.parametrize('hybrid', [True, False])
def test_seeds_and_bridges_validate_without_changing_retained_contracts(tmp_path, hybrid):
    from stash_ai.recognition.models import namespace
    from stash_ai.recognition.scanning import scan_identity
    from stash_ai.recognition.person_signal import same_contract
    config, _ = prepared(tmp_path, [[1., 0, 0, 0]], [[1., 0]])
    config.data['sapiensid']['enabled'] = hybrid
    feedback = dict(proposals={'face': [{'clustering.max_anchor_pose_deviation': .4}]})
    seeds = list(relaxed_seeds(config, feedback))
    assert len(seeds) == (3 if hybrid else 1)
    for changes in seeds:
        patches = [changes] + [p for _, p in repair_candidates(config, changes, [feedback], {})]
        for candidate in patches:
            variant = patched(config, candidate)
            assert same_contract(config, variant)
            assert scan_identity(config, namespace(config)) == scan_identity(variant, namespace(config))
            assert all(type(v) is int for k, v in candidate.items() if k.endswith('support'))
            if not hybrid:
                assert not any(k.startswith(PREFIX) for k in candidate)


def test_repairs_use_negative_witnesses_while_retaining_positive_gates(tmp_path):
    config, _ = prepared(tmp_path, [[1., 0, 0, 0]], [[1., 0]])
    face, body = PREFIX+'face_floor', PREFIX+'person_threshold'
    source = dict(query=dict(positive_witnesses=[{face: .65, body: .75}],
                            negative_witnesses=[{face: .4, body: .5}]))
    strategy, patch = next(repair_candidates(config, {face: .2, body: .35}, [source], {}))
    assert strategy == 'separation_repair'
    assert patch[face] <= .65 and patch[body] <= .75
    assert patch[face] > .4 or patch[body] > .5


def test_lane_stops_at_its_budget_and_single_sets_can_explore(tmp_path):
    config, _ = prepared(tmp_path, [[1., 0, 0, 0]], [[1., 0]])
    goals = goal_sets([['a'], ['b']], [['c']])
    status = assess(goals, dict.fromkeys('abc', 'scene'), {}, dict(zip('abc', 'abc')))
    source = dict(proposals={}, query={})
    lane = BidirectionalSearch(config, goals, 12, source)
    lane.observe('baseline', {}, status, objective(status), 'baseline')
    feedback = {'baseline': source}
    for i in range(lane.budget):
        candidate = lane.next(2+2*i, feedback)
        assert candidate is not None
        _, phase, _, _, patch = candidate
        name = f'trial-{i}'
        feedback[name] = source
        lane.observe(name, patch, status, objective(status), name, phase=phase)
    assert lane.next(20, feedback, force=True) is None
    single = BidirectionalSearch(config, goal_sets([['a'], ['b']]), 500, source)
    assert single.enabled and single.next(2, feedback) is not None


@pytest.mark.parametrize('style', ['auto', 'off'])
def test_cli_explains_permissive_trial_and_keeps_json_clean(tmp_path, monkeypatch, capsys, style):
    from stash_ai.recognition.cli import main
    from tests.test_recognition_merge_trial_progress import arguments
    config, goals = plateau(tmp_path, monkeypatch)
    assert main(arguments(config, goals, tmp_path, style)) == 0
    output = capsys.readouterr()
    result = json.loads(output.out)
    assert result['simulation']['trials']['trial-2']['search_strategy'] == 'relaxed_seed'
    assert result['search']['executed_by_phase']['bidirectional'] == 1
    assert result['selected_trial'] != 'trial-2'
    if style == 'off':
        assert output.err == ''
    else:
        assert 'Search: phase=bidirectional strategy=relaxed_seed' in output.err
        delta = next(c for c in result['simulation']['trials']['trial-2']['changes_from_previous']
                     if c['parameter'] == 'sapiensid.matching.face_floor')
        assert f"sapiensid.matching.face_floor: {delta['previous']} -> {delta['current']}" in output.err
        assert 'trial-2: complete disjoint=False' in output.err
