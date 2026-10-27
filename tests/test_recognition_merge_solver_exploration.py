"""Small useful threshold changes must not depend on catalogue nominations."""
from itertools import islice
import math

import pytest

from tests.test_recognition_merge_solver_hybrid import prepared
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.fragmentation_report import _open_readonly
from stash_ai.recognition.merge_goals import resolve, evaluate
from stash_ai.recognition.merge_goal_output import export
from stash_ai.recognition.merge_report_policy import patched
from stash_ai.recognition.merge_solver import solve
from stash_ai.recognition.merge_solver_hybrid import analyse
from stash_ai.recognition.merge_solver_search import hybrid_neighbours, gate_values, spread, BeamSearch, PREFIX


def auxiliary_goals(tmp_path, sets=1):
    faces = [[1., 0, 0, 0], [.785, math.sqrt(1-.785**2), 0, 0],
             [0, 0, 1., 0], [0, 0, .785, math.sqrt(1-.785**2)]]
    bodies = [[1., 0, 0, 0]]*2 + [[0, 0, 1., 0]]*2
    config, people = prepared(tmp_path, faces[:2*sets], bodies[:2*sets],
        matching=dict(min_support=3, person_threshold=.9))
    with FaceDatabase(config.database) as state:
        for person in people[1::2]:
            state.connection.execute("UPDATE observations SET quality=json_set(quality,'$.pose_deviation',.4) "
                'WHERE id IN (SELECT observation FROM members WHERE person=?)', (person,))
        state.connection.commit()
    with _open_readonly(config.database) as db:
        goals = resolve(db, config, [people[i:i+2] for i in range(0, len(people), 2)], by_performer=False)
    return config, goals


@pytest.mark.parametrize('sets', [1, 2])
def test_face_floor_alone_recovers_auxiliary_members_without_pair_proposals(tmp_path, sets):
    from stash_ai.recognition.config import load
    config, goals = auxiliary_goals(tmp_path, sets)
    before = config.database.read_bytes()
    with _open_readonly(config.database) as db:
        hints = analyse(db, config, goals)
        assert len(hints['missing_identities']) == sets
        assert not hints['proposals']
        assert not any(p['relation'] == 'within_set' for p in hints['pairs'])
        baseline = evaluate(config, goals, db)['trials']['baseline']['goal_result']
        assert not baseline['fully_solved']
        assert sum(g['withheld'] for g in baseline['groups']) == 3*sets
        result = dict(goals, **solve(config, goals, db, maximum=2))
    assert result['status'] == 'fully_solved'
    assert result['selected_trial'] == 'trial-1'
    assert {c['parameter']: c['proposed'] for c in result['best_result']['changes']} == {PREFIX+'face_floor': .785}
    assert result['search']['tested_hybrid_values']['face_floor'] == [.785, .8]
    trial = result['simulation']['trials']['trial-1']
    assert trial['search_parent'] == 'baseline' and trial['search_phase'] == 'baseline_exploration'
    assert trial['goal_result']['disjoint'] and trial['goal_result']['coverage_preserved']
    export(config, result, tmp_path/'result.json', tmp_path/'result.yml')
    with _open_readonly(config.database) as db:
        assert evaluate(load(tmp_path/'result.yml'), goals, db)['trials']['baseline']['goal_result']['fully_solved']
    assert config.database.read_bytes() == before


def test_large_joint_list_cannot_delay_independent_gate_probes(tmp_path, monkeypatch):
    from stash_ai.recognition import merge_solver
    config, goals = auxiliary_goals(tmp_path)
    original = merge_solver.measured_options
    def noisy_hints(*args, **kwargs):
        options = original(*args, **kwargs)
        # A long list of support/body proposals does not address the face veto.
        kwargs['hybrid']['proposals'] = [{PREFIX+'min_support': 1,
            PREFIX+'person_threshold': .4 + i/1000} for i in range(100)]
        return options
    monkeypatch.setattr(merge_solver, 'measured_options', noisy_hints)
    with _open_readonly(config.database) as db:
        result = solve(config, goals, db, maximum=4)
    assert result['status'] == 'fully_solved'
    assert result['selected_trial'] == 'trial-1'
    assert {c['parameter']: c['proposed'] for c in result['best_result']['changes']} == {PREFIX+'face_floor': .785}


def test_gate_search_keeps_distant_boundaries_and_explores_both_directions(tmp_path):
    config, _ = prepared(tmp_path, [[1., 0, 0, 0]], [[1., 0]])
    measured = [.799999, .799998, .799997, .799996, .615, .4, .800001, .9]
    candidates = list(hybrid_neighbours(config, {}, {PREFIX+'face_floor': measured}, []))
    floors = {p[PREFIX+'face_floor'] for p in candidates if set(p) == {PREFIX+'face_floor'}}
    assert set(measured) <= floors
    assert {.78, .75, .7, .6, .79, .82, .85, .9, 1., .81} <= floors
    assert len(floors) == sum(set(p) == {PREFIX+'face_floor'} for p in candidates)
    # Near-duplicate boundary values cannot consume all early measured slots.
    values = list(islice(spread(measured, .8), 6))
    assert any(v <= .615 for v in values)
    for candidate in candidates:
        patched(config, candidate)  # Production validation, including integer support.


def test_root_probes_continue_even_when_every_adaptive_child_improves():
    def factory(parent, changes, family, refine):
        for value in range(100):
            yield 'probe', {'value': value, 'depth': changes.get('depth', 0)+1}
    frontier = BeamSearch(['face'], factory, 500)
    frontier.observe('baseline', {}, (0, 0, 0), 'baseline')
    baseline_values, parents = [], []
    for trial in range(25):
        family, phase, parent, strategy, changes = frontier.next()
        if phase == 'baseline_exploration':
            baseline_values.append(changes['value'])
        else:
            parents.append(parent)
        frontier.completed(phase)
        frontier.observe(f'trial-{trial}', changes, (trial+1, 0, 0), str(trial), family=family)
        assert len(frontier.beam) <= 5
    assert baseline_values == list(range(16))
    assert parents and all(p != 'baseline' for p in parents)


def test_local_probes_do_not_invent_missing_body_evidence(tmp_path):
    config, goals = auxiliary_goals(tmp_path)
    with FaceDatabase(config.database) as state:
        state.connection.execute('DELETE FROM person_evidence')
        state.connection.commit()
    before = config.database.read_bytes()
    with _open_readonly(config.database) as db:
        result = solve(config, goals, db, maximum=2)
    assert not result['best_result']['fully_solved']
    assert result['selected_trial'] == 'baseline'
    assert result['search']['tested_hybrid_values']['face_floor'] == [.78, .8]
    assert config.database.read_bytes() == before


@pytest.mark.parametrize('interrupted_trial', ['baseline', 'trial-1'])
def test_tested_values_exclude_unfinished_probes(tmp_path, interrupted_trial):
    config, goals = auxiliary_goals(tmp_path)
    def progress(name, stage, item):
        if name == interrupted_trial and stage == 'start':
            raise KeyboardInterrupt
    with _open_readonly(config.database) as db:
        result = solve(config, goals, db, maximum=4, progress=progress)
    assert result['interrupted']
    assert result['search']['executed_by_phase'] == dict(baseline_exploration=0, adaptive=0, bidirectional=0, refinement=0, restart=0, robustness=0)
    if interrupted_trial == 'baseline':
        assert result['search']['tested_hybrid_values'] == {}
    else:
        assert result['search']['tested_hybrid_values']['face_floor'] == [.8]


@pytest.mark.parametrize('gate,current', [('face_floor', 0.), ('face_floor', 1.),
    ('person_threshold', .005), ('ambiguity_margin', 0.), ('min_support', 1)])
def test_edge_configuration_probes_are_valid_unique_and_nonempty(gate, current):
    values = list(gate_values(PREFIX+gate, current, []))
    assert values and len(values) == len(set(values)) and current not in values
    if gate == 'min_support':
        assert all(type(value) is int and value >= 1 for value in values)
    else:
        assert all(math.isfinite(value) and 0 <= value <= 1 for value in values)
