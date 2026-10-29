"""Negative scene labels must constrain real replay without changing matching."""
import json
import math

import pytest

from stash_ai.recognition.cli import parser, main
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.fragmentation_report import _open_readonly
from stash_ai.recognition.merge_goals import resolve, assess, evaluate, objective, unmet_conditions
from stash_ai.recognition.merge_solver import solve
from stash_ai.recognition.merge_report_policy import patched
from tests.test_recognition_merge_report import seeded


def fixture(tmp_path):
    return seeded(tmp_path, vectors=[[math.cos(math.radians(a)), math.sin(math.radians(a)), 0, 0]
        for a in (0, 10, 25)], recognition=dict(ambiguity_margin=0.), clustering=dict(view_ambiguity_margin=0.))


def test_parser_normalizes_repeatable_scene_exclusions():
    args = parser().parse_args(['report-merge', '--performer-set', '900,901',
        '--exclude-performer-scene', '00900:003', '--exclude-performer-scene', '901:4'])
    assert args.exclude_performer_scene == [('900', '3'), ('901', '4')]


@pytest.mark.parametrize('value', ['900', '900:3:4', '900:0', '-1:3', '900:abc', '900:３'])
def test_parser_rejects_malformed_exclusions(value):
    with pytest.raises(SystemExit):
        parser().parse_args(['report-merge', '--performer-set', '900', '--exclude-performer-scene', value])


@pytest.mark.parametrize('policy', ['identity', 'strict'])
def test_explicit_negative_replaces_wrong_positive_scene_not_retained_data(tmp_path, policy):
    c, people = fixture(tmp_path)
    with FaceDatabase(c.database) as state:
        state.connection.execute('UPDATE members SET person=? WHERE person=?', (people[0], people[2]))
        state.connection.commit()
    before = c.database.read_bytes()
    with _open_readonly(c.database) as db:
        goals = resolve(db, c, [['900', '901']], policy=policy, scene_exclusions=[('900', '3'), ('901', '3')])
        assert len(goals['scene_exclusions']) == 1
        assert goals['scene_exclusions'][0]['performer_ids'] == ['900', '901']
        group = goals['groups'][0]
        assert {r['scene'] for m in group['members'] for r in m['observations']} == {'1', '2'}
        assert len(group['members'][0]['excluded_observations']) == 3
        if policy == 'identity':
            assert set(group['identity_evidence']['scenes']) == {'1', '2'}
        assert len(goals['scene_exclusions'][0]['observation_ids']) == 3
    assert c.database.read_bytes() == before


def test_aliases_empty_fragments_and_invalid_or_unverifiable_goals(tmp_path):
    c, people = fixture(tmp_path)
    with FaceDatabase(c.database) as state:
        state.connection.execute('INSERT INTO identity_merges VALUES (?,?,?,?,?)',
            (people[2], people[0], 123., '{}', 'confirmed'))
        state.connection.commit()
    with _open_readonly(c.database) as db:
        goals = resolve(db, c, [['900', '901']], scene_exclusions=[('902', '3')])
        assert goals['scene_exclusions'][0]['resolved_identity_ids'] == [people[0]]
        # A member whose only scene was explicitly labelled wrong is no longer
        # a positive preservation obligation, including under strict policy.
        goals = resolve(db, c, [['900', '901']], policy='strict', scene_exclusions=[('900', '1')])
        assert len(goals['groups'][0]['members']) == 1
        assert goals['groups'][0]['excluded_members'][0]['identity_id'] == people[0]
        with pytest.raises(ValueError, match='without positive evidence'):
            resolve(db, c, [['900']], scene_exclusions=[('900', '1')])
        with pytest.raises(ValueError, match='must belong'):
            resolve(db, c, [['900']], scene_exclusions=[('901', '2')])
        with pytest.raises(ValueError, match='missing'):
            resolve(db, c, [['900']], scene_exclusions=[('900', '999')])
    with FaceDatabase(c.database) as state:
        state.connection.execute("UPDATE scans SET status='partial' WHERE scene='3'")
        state.connection.commit()
    with _open_readonly(c.database) as db:
        with pytest.raises(ValueError):
            resolve(db, c, [['900']], scene_exclusions=[('900', '3')])


@pytest.mark.parametrize('policy', ['identity', 'strict'])
def test_constraints_follow_splits_body_handoffs_and_block_vacuous_success(tmp_path, policy):
    c, people = fixture(tmp_path)
    with _open_readonly(c.database) as db:
        goals = resolve(db, c, [['900', '901']], policy=policy, scene_exclusions=[('900', '3')])
    item = goals['scene_exclusions'][0]
    positive = item['target_observation_ids']
    negative = item['observation_ids']
    obs = dict.fromkeys(positive, '1') | dict.fromkeys(negative, '3')
    good = dict.fromkeys(positive, 'new-merged-person') | dict.fromkeys(negative, 'other')
    passed = assess(goals, obs, {}, good)
    assert passed['fully_solved']
    for bad in (dict(good, **{negative[0]: 'new-merged-person'}),
                dict(good, **{positive[0]: 'split-person', negative[0]: 'split-person'})):
        failed = assess(goals, obs, {}, bad)
        assert not failed['admissible'] and not failed['fully_solved']
        assert 'excluded_performer_in_scene' in {r['reason'] for r in unmet_conditions(failed)}
        assert objective(passed) > objective(failed)
    # A split can reuse the old UUID for the rejected fragment. Constraints,
    # like merge goals, identify the person by its positive evidence.
    renamed = dict.fromkeys(positive, people[1]) | dict.fromkeys(negative, people[0])
    assert assess(goals, obs, {}, renamed)['fully_solved']
    # Alternative body addresses must not add extra positive votes, but must
    # expose a forbidden body-only assignment even if all linked faces are gone.
    item['target_observation_ids'].append('body-positive')
    body_obs = dict(obs, **{'body-positive': '1', 'body-negative': '3'})
    body_predicted = dict.fromkeys(positive, None) | {'body-positive': 'body-person', 'body-negative': 'body-person'}
    failed = assess(goals, body_obs, {}, body_predicted)
    assert failed['scene_exclusions'][0]['matching_observations'] == ['body-negative']
    assert not failed['fully_solved']
    absent = assess(goals, obs, {}, {})
    assert not absent['scene_exclusions_satisfied']
    assert 'excluded_performer_unresolved' in {r['reason'] for r in unmet_conditions(absent)}
    missing = assess(goals, {k: v for k, v in obs.items() if k not in negative}, {}, good)
    assert not missing['scene_exclusions_satisfied']
    assert 'excluded_scene_not_evaluated' in {r['reason'] for r in unmet_conditions(missing)}
    many = dict.fromkeys((f'extra-{i}' for i in range(100)), '3')
    overflow = assess(goals, obs | many, {}, good | dict.fromkeys(many, 'new-merged-person'))
    detail = overflow['scene_exclusions'][0]
    assert detail['matching_count'] == 100 and len(detail['matching_observations']) == 32
    assert detail['observation_examples_truncated']
    assert not overflow['fully_solved']


@pytest.mark.parametrize('policy', ['identity', 'strict'])
@pytest.mark.parametrize('already_assigned', [False, True])
def test_solver_repairs_forbidden_face_scene_preserving_positive_merge(tmp_path, policy, already_assigned):
    c, people = fixture(tmp_path)
    if already_assigned:
        with FaceDatabase(c.database) as state:
            state.connection.execute('UPDATE members SET person=? WHERE person=?', (people[0], people[2]))
            state.connection.commit()
    before = c.database.read_bytes()
    with _open_readonly(c.database) as db:
        goals = resolve(db, c, [['900', '901']], policy=policy, scene_exclusions=[('900', '3')])
        result = solve(c, goals, db, maximum=12)
        baseline = result['simulation']['trials']['baseline']['goal_result']
        assert not baseline['scene_exclusions_satisfied']
        assert result['status'] == 'fully_solved'
        changes = {r['parameter']: r['proposed'] for r in result['recommendation']['changes']}
        check = evaluate(patched(c, changes), goals, db)['trials']['baseline']['goal_result']
        assert check['fully_solved'] and check['scene_exclusions_satisfied']
        assert check['groups'][0]['assigned'] == 6
    assert c.database.read_bytes() == before


def test_cli_two_performer_simulation_reports_exclusions(tmp_path, capsys):
    c, _ = fixture(tmp_path)
    output = tmp_path/'excluded.json'
    assert main(['report-merge', '--face-config', str(c.source_path), '--faces-db', str(c.database),
        '--performer-id', '900', '--performer-id', '901', '--exclude-performer-scene', '900:3',
        '--simulate', '--output', str(output)]) == 0
    result = json.loads(output.read_text())
    assert result['scene_exclusions'][0]['scene'] == '3'
    assert result['explanation']['unresolved'][0]['reason'] == 'excluded_performer_in_scene'
    assert result['provenance']['goals_sha256']


def test_solver_repairs_independent_body_scene_and_exports_replayable_yaml(tmp_path):
    import numpy as np
    from tests.test_recognition_body_integration import seeded as body_seeded
    from stash_ai.recognition.merge_goal_output import export
    from stash_ai.recognition.config import load
    c, _, ids, _ = body_seeded(tmp_path)
    c.data['sapiensid']['global_matching']['ambiguity_margin'] = 0.
    with FaceDatabase(c.database) as state:
        for observations, angle in zip(ids, (0, 10, 25)):
            vector = np.zeros(4096, dtype='<f4')
            vector[:2] = [math.cos(math.radians(angle)), math.sin(math.radians(angle))]
            for oid in observations:
                state.connection.execute('UPDATE body_observations SET embedding=? WHERE id=?', (vector.tobytes(), oid))
        state.connection.commit()
    before = c.database.read_bytes()
    with _open_readonly(c.database) as db:
        goals = resolve(db, c, [['900', '901']], scene_exclusions=[('900', '3')])
        result = dict(goals, **solve(c, goals, db, maximum=16))
        assert not result['simulation']['trials']['baseline']['goal_result']['scene_exclusions_satisfied']
        assert result['status'] == 'fully_solved'
        assert any(r['parameter'].startswith('sapiensid.global_matching.') for r in result['recommendation']['changes'])
    export(c, result, tmp_path/'report.json', tmp_path/'winner.yml')
    winner = load(tmp_path/'winner.yml')
    with _open_readonly(c.database) as db:
        check = evaluate(winner, goals, db)['trials']['baseline']['goal_result']
        assert check['fully_solved'] and check['groups'][0]['assigned'] == 6
    assert c.database.read_bytes() == before


def test_linked_body_addresses_are_frozen_without_positive_double_counting(tmp_path):
    import uuid
    import numpy as np
    from tests.test_recognition_body_integration import seeded as body_seeded
    from stash_ai.recognition.database import encoded
    c, selected, body_ids, people = body_seeded(tmp_path)
    faces = []
    with FaceDatabase(c.database) as state:
        db = state.connection
        for i, body in enumerate(body_ids[0]):
            face = str(uuid.uuid4()); faces.append(face)
            db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                (face, selected[0]['scan'], i, float(i), '', '[]', '[]',
                 encoded(dict(pixels=100, sharpness=100., pose_deviation=0.)), None, 4,
                 np.asarray([1, 0, 0, 0], dtype='<f4').tobytes(), None))
            db.execute('INSERT INTO members VALUES (?,?)', (face, people[0]))
            db.execute('UPDATE body_observations SET face=? WHERE id=?', (face, body))
        db.commit()
    with _open_readonly(c.database) as db:
        goals = resolve(db, c, [['900']], scene_exclusions=[('900', '3')])
        assert len(goals['groups'][0]['members'][0]['observations']) == 3
        item = goals['scene_exclusions'][0]
        assert set(item['target_observation_ids']) == set(faces + body_ids[0])
        observations = dict.fromkeys(faces + body_ids[0], '1') | dict.fromkeys(body_ids[2], '3')
        predicted = dict.fromkeys(body_ids[0] + body_ids[2], 'body-only-destination')
        check = assess(goals, observations, {}, predicted)
        assert check['scene_exclusions'][0]['matching_observations'] == sorted(body_ids[2])


def test_negative_footprint_and_checkpoint_cannot_ignore_changed_exclusions(tmp_path):
    from stash_ai.recognition.merge_goal_index import GoalIndex
    c, _ = fixture(tmp_path)
    with _open_readonly(c.database) as db:
        goals = resolve(db, c, [['900', '901']], scene_exclusions=[('900', '3')])
        index = GoalIndex(goals)
        assert set(goals['scene_exclusions'][0]['observation_ids']) <= index.outcome_ids
        journal = tmp_path/'checkpoint.json'
        original = solve(c, goals, db, maximum=1, checkpoint=journal)
        resumed = solve(c, goals, db, maximum=1, checkpoint=journal, resume=True)
        assert resumed['search']['restored_trials'] == 1
        assert resumed['best_result'] == original['best_result']
        changed = resolve(db, c, [['900', '901']])
        with pytest.raises(ValueError, match='incompatible'):
            solve(c, changed, db, maximum=1, checkpoint=journal, resume=True)


def test_negative_hint_comparisons_respect_budget(tmp_path):
    from stash_ai.recognition.merge_exclusion_feedback import analyse
    from stash_ai.recognition.merge_report_policy import patched
    c, _ = fixture(tmp_path)
    c = patched(c, {'sapiensid.execution.max_candidate_pairs': 2})
    with _open_readonly(c.database) as db:
        goals = resolve(db, c, [['900', '901']], scene_exclusions=[('900', '3')])
        feedback = analyse(db, c, goals)
    assert feedback['comparisons'] <= 2 and feedback['truncated']
