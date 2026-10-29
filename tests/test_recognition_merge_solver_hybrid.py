"""Hybrid search must improve the actual replay, not just frozen pair scores."""
from dataclasses import replace
import json
import math

import numpy as np
import pytest

from tests.test_recognition_merge_report import seeded
from stash_ai.recognition.config import plain, load
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.fragmentation_report import _open_readonly
from stash_ai.recognition.person_signal import save, contract
from stash_ai.recognition.merge_goals import resolve, evaluate
from stash_ai.recognition.merge_solver import solve, neighbours, rebuilt_goals
from stash_ai.recognition.merge_solver_hybrid import analyse, mode_boundary, PREFIX


def prepared(tmp_path, faces, bodies, *, count=3, matching=None):
    config, people = seeded(tmp_path, vectors=faces, count=count, recognition=dict(match_threshold=.99))
    data = plain(config.data)
    data['sapiensid']['enabled'] = True
    data['sapiensid']['matching'].update(face_floor=.8, person_threshold=.99, min_support=4)
    data['sapiensid']['matching'].update(matching or {})
    config = replace(config, data=data)
    with FaceDatabase(config.database) as state:
        for person, body in zip(people, bodies):
            for row in state.connection.execute('SELECT observation FROM members WHERE person=?', (person,)).fetchall():
                if body is not None:
                    save(state.connection, row[0], contract(config), {}, np.pad(body, (0, 4096-len(body))))
        state.connection.commit()
    return config, people


def test_joint_hybrid_trial_clears_four_gates_and_export_replays(tmp_path, monkeypatch):
    faces = [[1., 0, 0, 0], [.7, math.sqrt(.51), 0, 0], [.9, 0, math.sqrt(.19), 0]]
    bodies = [[math.cos(a), math.sin(a)] for a in (0, .3, -.35)]
    config, people = prepared(tmp_path, faces, bodies)
    monkeypatch.setattr('stash_ai.recognition.models.load_backend', lambda *a, **k: pytest.fail('offline inference'))
    monkeypatch.setattr('stash_ai.recognition.sapiensid_assets.install', lambda *a, **k: pytest.fail('offline download'))
    before = config.database.read_bytes()
    with _open_readonly(config.database) as db:
        goals = resolve(db, config, [people[:2]], by_performer=False)
        result = dict(goals, **solve(config, goals, db, targeted_trials=0, feasibility_trials=0, maximum=2))
    assert result['status'] == 'fully_solved'
    assert not result['simulation']['trials']['baseline']['goal_result']['fully_solved']
    changed = {c['parameter']: c['proposed'] for c in result['recommendation']['changes']}
    assert set(changed) == {PREFIX+k for k in ('face_floor', 'person_threshold', 'min_support', 'ambiguity_margin')}
    assert changed[PREFIX+'min_support'] == 3
    assert 0 < changed[PREFIX+'ambiguity_margin'] < .02
    trial = result['simulation']['trials']['trial-1']
    assert trial['search_family'] == 'hybrid'
    assert trial['goal_result']['groups'][0]['outsider_observations'] == 0
    assert result['search']['executed_by_family'] == dict(hybrid=1, face=0)
    assert any(r['rival'] == people[2] for r in result['measured_hybrid_evidence']['nearest_competitors'])
    from stash_ai.recognition.merge_goal_output import export
    export(config, result, tmp_path/'result.json', tmp_path/'result.yml')
    exported = load(tmp_path/'result.yml')
    with _open_readonly(config.database) as db:
        assert evaluate(exported, goals, db)['trials']['baseline']['goal_result']['fully_solved']
    assert config.database.read_bytes() == before
    json.dumps(result, allow_nan=False)


def test_two_sets_share_hybrid_patch_without_cross_set_merge(tmp_path):
    faces = [[1., 0, 0, 0], [.7, math.sqrt(.51), 0, 0], [0, 0, 1., 0], [0, 0, .7, math.sqrt(.51)]]
    bodies = [[1., 0, 0, 0], [.96, .28, 0, 0], [0, 0, 1., 0], [0, 0, .96, .28]]
    config, people = prepared(tmp_path, faces, bodies)
    with _open_readonly(config.database) as db:
        goals = resolve(db, config, [people[:2], people[2:]], by_performer=False)
        result = solve(config, goals, db, targeted_trials=0, feasibility_trials=0, maximum=3)
    assert result['status'] == 'fully_solved'
    trial = result['simulation']['trials'][result['selected_trial']]
    assert trial['goal_result']['disjoint']
    assert [g['fragments'] for g in trial['goal_result']['groups']] == [1, 1]
    assert result['search']['executed_by_family'] == dict(hybrid=1, face=1)
    proof = result['measured_hybrid_evidence']
    assert sum(p['relation'] == 'within_set' for p in proof['pairs']) == 2
    assert sum(p['relation'] == 'between_sets' for p in proof['pairs']) == 4
    final = trial['hybrid_solver_evidence']
    assert {tuple(p['sets']) for p in final['identities']} == {(0,), (1,)}
    assert all(p['relation'] == 'between_sets' for p in final['pairs'])


@pytest.mark.parametrize('reason', ['missing', 'wrong_contract'])
def test_missing_or_incompatible_body_features_are_not_invented(tmp_path, reason):
    bodies = [None, None] if reason == 'missing' else [[1., 0], [1., 0]]
    config, people = prepared(tmp_path, [[1., 0, 0, 0], [.7, math.sqrt(.51), 0, 0]], bodies)
    if reason == 'wrong_contract':
        config.data['sapiensid']['model']['variant'] = 'wb4m'
    with _open_readonly(config.database) as db:
        goals = resolve(db, config, [people], by_performer=False)
        result = analyse(db, config, goals)
    assert not result['proposals']
    assert not result['options'][PREFIX+'min_support']
    assert not result['options'][PREFIX+'person_threshold']
    assert all(i['independent_support'] == 0 for i in result['identities'])
    assert 'missing_person_evidence' in result['pairs'][0]['blockers']
    from stash_ai.recognition.merge_goal_output import explain
    report = dict(simulation=dict(trials=dict(baseline=dict(hybrid_solver_evidence=result,
        goal_result=dict(groups=[], disjoint=True)))))
    explain(report)
    assert len(report['explanation']['hybrid']['missing_evidence']) == 2
    assert 'missing_person_evidence' in report['explanation']['hybrid']['pair_blockers'][0]['blockers']


def test_support_counts_independent_cues_not_duplicate_embeddings(tmp_path):
    config, people = prepared(tmp_path, [[1., 0, 0, 0], [.7, math.sqrt(.51), 0, 0]], [[1., 0], [1., 0]])
    with FaceDatabase(config.database) as state:
        db = state.connection
        db.execute('UPDATE observations SET seconds=0')
        db.commit()
        goals = resolve(db, config, [people], by_performer=False)
        result = analyse(db, config, goals)
    assert result['pairs'][0]['support'] == [1, 1]
    assert result['proposals'][0][PREFIX+'min_support'] == 1
    assert 'insufficient_independent_support' in result['pairs'][0]['blockers']


def test_modes_keep_supported_looks_and_original_core_guards(monkeypatch):
    from types import SimpleNamespace
    from stash_ai.recognition.config import defaults
    from stash_ai.recognition.hybrid_evidence import Identity
    # Raw cross-look minimum is zero, but both looks have a matching mode at .96.
    a = np.array([[1., 0, 0, 0], [0, 0, 1., 0]])
    b = np.array([[.96, .28, 0, 0], [0, 0, .96, .28]])
    modes_a, modes_b = tuple(a[i:i+1] for i in range(2)), tuple(b[i:i+1] for i in range(2))
    assert mode_boundary(modes_a, modes_b) == pytest.approx(.96)
    faces = np.array([[1., 0.]])
    bank = [Identity('a', ('a',), faces, a, (1, 2, 3), body_modes=modes_a),
            Identity('b', ('b',), faces, b, (1, 2, 3), body_modes=modes_b)]
    monkeypatch.setattr('stash_ai.recognition.merge_solver_hybrid.catalogue', lambda *a: bank)
    monkeypatch.setattr('stash_ai.recognition.global_views._cooccurs', lambda *a: False)
    config = SimpleNamespace(data=defaults())
    config.data['sapiensid']['enabled'] = True
    config.data['sapiensid']['matching']['person_threshold'] = .99
    goals = dict(scope='s', namespace='n', groups=[dict(members=[dict(identity_id=p) for p in ('a', 'b')])])
    result = analyse(None, config, goals)
    assert result['proposals'][0][PREFIX+'person_threshold'] == pytest.approx(.96)
    assert result['pairs'][0]['person']['minimum'] == 0
    assert result['pairs'][0]['mode_boundary_evidence']['person']['strategy'] == 'all_supported_modes_covered'
    bank[0] = replace(bank[0], guard_persons=-np.ones((1, 4))/2, guard_observations=('old',))
    guarded = analyse(None, config, goals)
    assert not guarded['proposals']
    bank[0] = replace(bank[0], guard_persons=None, guard_missing=('old',))
    missing = analyse(None, config, goals)
    assert not missing['proposals']
    assert missing['identities'][0]['missing_original_core_observations'] == ['old']


def test_real_mode_replay_does_not_lower_threshold_to_cross_look_minimum(tmp_path):
    config, people = prepared(tmp_path, [[1., 0, 0, 0], [.7, math.sqrt(.51), 0, 0]],
        [[1., 0, 0, 0], [.96, .28, 0, 0]], count=6, matching=dict(min_support=3))
    with FaceDatabase(config.database) as state:
        for person, body in zip(people, ([0, 0, 1., 0], [0, 0, .96, .28])):
            rows = state.connection.execute('SELECT o.id FROM observations o JOIN members m ON m.observation=o.id '
                                            'WHERE m.person=? AND o.cue>=3', (person,)).fetchall()
            for row in rows:
                save(state.connection, row[0], contract(config), {}, np.pad(body, (0, 4092)))
        state.connection.commit()
    before = config.database.read_bytes()
    with _open_readonly(config.database) as db:
        goals = resolve(db, config, [people], by_performer=False)
        result = solve(config, goals, db, targeted_trials=0, feasibility_trials=0, maximum=2)
    assert result['status'] == 'fully_solved'
    proof = result['measured_hybrid_evidence']['pairs'][0]
    assert proof['person']['minimum'] == 0
    assert proof['mode_boundary'] == pytest.approx(.96)
    assert all(i['body_modes'] == 2 for i in result['measured_hybrid_evidence']['identities'])
    proposed = {c['parameter']: c['proposed'] for c in result['recommendation']['changes']}
    assert proposed[PREFIX+'person_threshold'] > .95
    assert config.database.read_bytes() == before


def test_rebuilt_goals_preserve_splits_and_cross_set_conflicts(tmp_path):
    config, people = seeded(tmp_path)
    with FaceDatabase(config.database) as state:
        db = state.connection
        goals = resolve(db, config, [[people[0]], [people[1]]], by_performer=False, policy='strict')
        observation = goals['groups'][1]['members'][0]['observations'][0]['id']
        db.execute('UPDATE members SET person=? WHERE observation=?', (people[0], observation))
        mapped = rebuilt_goals(db, goals)
        assert [len(g['members']) for g in mapped['groups']] == [1, 2]
        assert mapped['groups'][0]['members'][0]['identity_id'] == people[0]
        assert {m['identity_id'] for m in mapped['groups'][1]['members']} == set(people)


def test_hybrid_parameters_are_interleaved_before_large_face_grid(tmp_path):
    from itertools import islice
    config, _ = seeded(tmp_path)
    options = {'recognition.match_threshold': [i/1000 for i in range(1000)],
        PREFIX+'face_floor': [.2], PREFIX+'person_threshold': [.7],
        PREFIX+'ambiguity_margin': [.01], PREFIX+'min_support': [1]}
    proposals = list(islice(neighbours(config, {}, options), 8))
    assert all(any(k.startswith(PREFIX) for k in p) for p in proposals[::2])
    assert any(PREFIX+'min_support' in p for p in proposals)


def test_diagnostics_stream_only_incident_pairs_and_enforce_proof_budgets(tmp_path):
    faces = [[math.cos(a), math.sin(a), 0, 0] for a in (0, .3, .6, .9)]
    bodies = [[math.cos(a), math.sin(a)] for a in (0, .3, .6, .9)]
    config, people = prepared(tmp_path, faces, bodies)
    before = config.database.read_bytes()
    with _open_readonly(config.database) as db:
        goals = resolve(db, config, [people[:2]], by_performer=False)
        result = analyse(db, config, goals)
        assert result['comparisons'] == 5  # No comparison of the two outsiders.
        assert len(result['pairs']) == 1
        assert len(result['nearest_competitors']) <= 2
        config.data['sapiensid']['execution']['max_candidate_pairs'] = 1
        with pytest.raises(ValueError, match='max_candidate_pairs'):
            analyse(db, config, goals)
        config.data['sapiensid']['execution']['max_candidate_pairs'] = 100000
        config.data['sapiensid']['execution']['plan_bytes'] = 1
        with pytest.raises(ValueError, match='plan_bytes'):
            analyse(db, config, goals)
    assert config.database.read_bytes() == before
