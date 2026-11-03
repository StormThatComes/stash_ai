"""Sample negatives constrain production replays without deleting evidence."""
import json
import math
import uuid

import numpy as np
import pytest

from stash_ai.recognition.cli import main, parser
from stash_ai.recognition.config import load
from stash_ai.recognition.database import FaceDatabase, encoded
from stash_ai.recognition.fragmentation_report import _open_readonly
from stash_ai.recognition.merge_goal_index import GoalIndex
from stash_ai.recognition.merge_goal_output import export, summary
from stash_ai.recognition.merge_goals import resolve, assess, evaluate, objective, unmet_conditions
from stash_ai.recognition.merge_solver import solve
from tests.test_recognition_scene_exclusions import fixture


def face_fixture(tmp_path, *, already_assigned=True):
    c, people = fixture(tmp_path)
    with FaceDatabase(c.database) as state:
        db = state.connection
        ids = [list(r[0] for r in db.execute('SELECT observation FROM members WHERE person=? ORDER BY observation',
                (person,))) for person in people]
        scan = db.execute('SELECT scan FROM observations WHERE id=?', (ids[0][0],)).fetchone()[0]
        negative = ids[2][0]
        db.execute('UPDATE observations SET scan=?,cue=3,seconds=3 WHERE id=?', (scan, negative))
        db.execute('UPDATE scans SET total=4,cursor=4 WHERE id=?', (scan,))
        if already_assigned:
            db.execute('UPDATE members SET person=? WHERE observation=?', (people[0], negative))
        db.commit()
    return c, people, ids, negative


def linked_fixture(tmp_path):
    from tests.test_recognition_body_integration import seeded
    c, selected, bodies, people = seeded(tmp_path)
    faces = []
    with FaceDatabase(c.database) as state:
        db = state.connection
        for cue, body in enumerate(bodies[0]):
            face = str(uuid.uuid4())
            faces.append(face)
            db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                (face, selected[0]['scan'], cue, float(cue), '', '[]', '[]',
                 encoded(dict(pixels=100, sharpness=100., pose_deviation=0.)), None, 4,
                 np.asarray([1, 0, 0, 0], dtype='<f4').tobytes(), None))
            db.execute('INSERT INTO members VALUES (?,?)', (face, people[0]))
            db.execute('UPDATE body_observations SET face=? WHERE id=?', (face, body))
        db.commit()
    return c, faces, bodies, people


def test_cli_repeatable_colon_syntax():
    args = parser().parse_args(['report-merge', '--performer-set', '900',
        '--exclude-performer-observation', '00900:sample-a',
        '--exclude-performer-observation', '900:sample-b', '--exclude-performer-scene', '900:3'])
    assert args.exclude_performer_observation == [('900', 'sample-a'), ('900', 'sample-b')]
    assert args.exclude_performer_scene == [('900', '3')]


@pytest.mark.parametrize('value', ['900', '900:', ':sample', '900:a:b', '0:a', '-1:a', '９００:a', '900:a b'])
def test_cli_rejects_malformed_pairs(value):
    with pytest.raises(SystemExit):
        parser().parse_args(['report-merge', '--performer-set', '900', '--exclude-performer-observation', value])


@pytest.mark.parametrize('policy', ['identity', 'identity-physical', 'strict'])
def test_only_negative_sample_leaves_positive_goals_and_source_is_unchanged(tmp_path, policy):
    c, _, ids, negative = face_fixture(tmp_path)
    before = c.database.read_bytes()
    with _open_readonly(c.database) as db:
        goals = resolve(db, c, [['900', '901']], policy=policy,
                        observation_exclusions=[('900', negative), ('901', negative)])
        assert len(goals['observation_exclusions']) == 1
        item = goals['observation_exclusions'][0]
        assert item['performer_ids'] == ['900', '901']
        assert item['observation_ids'] == [negative]
        assert set(item['target_observation_ids']) == set(ids[0] + ids[1])
        group = goals['groups'][0]
        assert {r['id'] for m in group['members'] for r in m['observations']} == set(ids[0] + ids[1])
        assert group['members'][0]['excluded_observations'] == [dict(id=negative, scene='1')]
        if policy != 'strict':
            assert set(group['identity_evidence']['scenes']) == {'1', '2'}
        assert negative in GoalIndex(goals).outcome_ids
    assert c.database.read_bytes() == before


@pytest.mark.parametrize('policy', ['identity', 'identity-physical', 'strict'])
def test_constraints_follow_positive_evidence_and_report_final_assignments(tmp_path, policy):
    c, people, ids, negative = face_fixture(tmp_path)
    with _open_readonly(c.database) as db:
        goals = resolve(db, c, [['900', '901']], policy=policy, observation_exclusions=[('900', negative)])
    observations = dict.fromkeys(ids[0] + [negative], '1') | dict.fromkeys(ids[1], '2')
    good = dict.fromkeys(ids[0] + ids[1], 'new-person') | {negative: people[0]}
    passed = assess(goals, observations, {}, good)
    assert passed['fully_solved']  # The old UUID can be reused by the wrong fragment.
    detail = passed['observation_exclusions'][0]
    assert detail['assignments'] == [dict(observation_id=negative, status='assigned', identity=people[0])]
    for wrong in (good | {negative: 'new-person'}, good | {ids[0][0]: 'split', negative: 'split'}):
        failed = assess(goals, observations, {}, wrong)
        assert not failed['admissible'] and not failed['fully_solved']
        assert 'excluded_observation_assigned' in {r['reason'] for r in unmet_conditions(failed)}
        assert objective(passed) > objective(failed)
    unassigned = assess(goals, observations, {}, good | {negative: None})
    assert unassigned['fully_solved']
    assert unassigned['observation_exclusions'][0]['assignments'][0]['status'] == 'unassigned'
    missing = assess(goals, {k: v for k, v in observations.items() if k != negative}, {}, good)
    assert not missing['fully_solved']
    assert 'excluded_observation_not_evaluated' in {r['reason'] for r in unmet_conditions(missing)}
    assert missing['observation_exclusions'][0]['assignments'][0]['status'] == 'missing'
    lost = assess(goals, observations, {}, {negative: 'other'})
    assert not lost['observation_exclusions_satisfied']
    assert 'excluded_performer_unresolved' in {r['reason'] for r in unmet_conditions(lost)}


def test_aliases_empty_sets_and_invalid_inputs(tmp_path):
    c, people, ids, negative = face_fixture(tmp_path)
    with FaceDatabase(c.database) as state:
        state.connection.execute('INSERT INTO identity_merges VALUES (?,?,?,?,?)',
            (people[2], people[0], 123., '{}', 'confirmed'))
        state.connection.commit()
    with _open_readonly(c.database) as db:
        goals = resolve(db, c, [['900', '901']], observation_exclusions=[('902', negative)])
        assert goals['observation_exclusions'][0]['resolved_identity_ids'] == [people[0]]
        pairs = [('900', oid) for oid in ids[0] + [negative]]
        with pytest.raises(ValueError, match='without positive evidence'):
            resolve(db, c, [['900']], observation_exclusions=pairs)
        goals = resolve(db, c, [['900', '901']], policy='strict', observation_exclusions=pairs)
        assert goals['groups'][0]['excluded_members'][0]['identity_id'] == people[0]
        with pytest.raises(ValueError, match='must belong'):
            resolve(db, c, [['900']], observation_exclusions=[('901', negative)])
        with pytest.raises(ValueError, match='unknown'):
            resolve(db, c, [['900']], observation_exclusions=[('900', 'missing')])


@pytest.mark.parametrize('mutation, error', [
    ("scope='other'", 'scope'), ("namespace='other'", 'namespace'),
    ("status='partial'", 'incomplete_scan'), ("policy_sha='old'", 'scan_policy_mismatch')])
def test_wrong_scope_or_incompatible_scan_is_not_absence(tmp_path, mutation, error):
    c, _, ids, _ = face_fixture(tmp_path)
    negative = ids[2][1]
    with FaceDatabase(c.database) as state:
        state.connection.execute(f'UPDATE scans SET {mutation} WHERE scene=?', ('3',))
        state.connection.commit()
    with _open_readonly(c.database) as db:
        with pytest.raises(ValueError, match=error):
            resolve(db, c, [['900']], observation_exclusions=[('900', negative)])


def test_historical_observation_is_rejected_even_if_the_old_scan_is_compatible(tmp_path):
    c, _, ids, _ = face_fixture(tmp_path)
    negative = ids[2][1]
    with FaceDatabase(c.database) as state:
        state.connection.execute("INSERT INTO scans SELECT ?,scope,scene,namespace,'new-input',policy_sha,"
            'manifest,coverage,total,cursor,status FROM scans WHERE scene=?', (str(uuid.uuid4()), '3'))
        state.connection.commit()
    with _open_readonly(c.database) as db:
        with pytest.raises(ValueError, match='current scan'):
            resolve(db, c, [['900']], observation_exclusions=[('900', negative)])


@pytest.mark.parametrize('policy', ['identity', 'identity-physical', 'strict'])
def test_linked_face_and_body_are_one_negative_with_no_positive_backdoor(tmp_path, policy):
    c, faces, bodies, _ = linked_fixture(tmp_path)
    with _open_readonly(c.database) as db:
        goals = resolve(db, c, [['900']], policy=policy,
            observation_exclusions=[('900', faces[0]), ('900', bodies[0][0])])
        item = goals['observation_exclusions'][0]
        assert len(goals['observation_exclusions']) == 1
        assert set(item['requested_observation_ids']) == set(item['observation_ids']) == {faces[0], bodies[0][0]}
        assert set(item['target_observation_ids']) == set(faces[1:] + bodies[0][1:])
        assert {r['id'] for m in goals['groups'][0]['members'] for r in m['observations']} == set(faces[1:])
        assert not ({faces[0], bodies[0][0]} & set(goals.get('physical_alternatives', {})))
    observations = dict.fromkeys(faces + bodies[0], '1')
    # Negative face withheld, negative body still assigned: the goal must fail.
    bad = dict.fromkeys(bodies[0][1:] + [bodies[0][0]], 'body-target')
    failed = assess(goals, observations, {}, bad)
    assert not failed['observation_exclusions_satisfied']
    assert failed['observation_exclusions'][0]['matching_observations'] == [bodies[0][0]]
    # Selecting the body alone must also exclude its face.
    with _open_readonly(c.database) as db:
        body_goal = resolve(db, c, [['900']], policy=policy, observation_exclusions=[('900', bodies[0][0])])
    failed = assess(body_goal, observations, {}, dict.fromkeys(faces, 'face-target'))
    assert failed['observation_exclusions'][0]['matching_observations'] == [faces[0]]


@pytest.mark.parametrize('mismatch', ['disabled', 'contract', 'cross-cue'])
def test_incompatible_body_samples_are_rejected(tmp_path, mismatch):
    c, faces, bodies, _ = linked_fixture(tmp_path)
    if mismatch == 'disabled':
        c.data['sapiensid']['identities']['enabled'] = False
    else:
        with FaceDatabase(c.database) as state:
            if mismatch == 'contract':
                state.connection.execute("UPDATE body_observations SET contract='old' WHERE id=?", (bodies[0][0],))
            else:
                state.connection.execute('UPDATE body_observations SET face=? WHERE id=?', (faces[1], bodies[0][0]))
            state.connection.commit()
    with _open_readonly(c.database) as db:
        with pytest.raises(ValueError, match='compatible|incompatible'):
            resolve(db, c, [['900']], observation_exclusions=[('900', bodies[0][0])])


def test_scene_and_sample_constraints_coexist_and_rank_together(tmp_path):
    c, _, ids, negative = face_fixture(tmp_path)
    with _open_readonly(c.database) as db:
        goals = resolve(db, c, [['900', '901']], scene_exclusions=[('900', '3')],
                        observation_exclusions=[('900', negative)])
    observations = dict.fromkeys(ids[0] + [negative], '1') | dict.fromkeys(ids[1], '2') | dict.fromkeys(ids[2][1:], '3')
    good = dict.fromkeys(ids[0] + ids[1], 'target')
    passed = assess(goals, observations, {}, good)
    assert passed['fully_solved']
    both = assess(goals, observations, {}, good | {negative: 'target', ids[2][1]: 'target'})
    one = assess(goals, observations, {}, good | {negative: 'target'})
    assert not both['scene_exclusions_satisfied'] and not both['observation_exclusions_satisfied']
    assert objective(passed) > objective(one) > objective(both)


def test_same_cue_neighbor_is_not_part_of_the_excluded_physical_sample(tmp_path):
    c, faces, bodies, _ = linked_fixture(tmp_path)
    with FaceDatabase(c.database) as state:
        # A second person in the same cue has its own retained face association.
        state.connection.execute('UPDATE observations SET cue=0,seconds=0 WHERE id=?', (faces[1],))
        state.connection.execute('UPDATE body_observations SET cue=0,seconds=0 WHERE id=?', (bodies[0][1],))
        state.connection.commit()
    with _open_readonly(c.database) as db:
        goals = resolve(db, c, [['900']], observation_exclusions=[('900', faces[0])])
    item = goals['observation_exclusions'][0]
    assert set(item['observation_ids']) == {faces[0], bodies[0][0]}
    assert {faces[1], bodies[0][1]} <= set(item['target_observation_ids'])


@pytest.mark.parametrize('policy', ['identity', 'strict'])
@pytest.mark.parametrize('already_assigned', [False, True])
def test_solver_repairs_one_sample_in_a_positive_scene_and_exports_replayable_yaml(tmp_path, policy, already_assigned):
    c, _, _, negative = face_fixture(tmp_path, already_assigned=already_assigned)
    before = c.database.read_bytes()
    with _open_readonly(c.database) as db:
        goals = resolve(db, c, [['900', '901']], policy=policy, observation_exclusions=[('900', negative)])
        result = dict(goals, **solve(c, goals, db, maximum=16))
        assert not result['simulation']['trials']['baseline']['goal_result']['observation_exclusions_satisfied']
        assert result['status'] == 'fully_solved'
    export(c, result, tmp_path/'report.json', tmp_path/'winner.yml')
    assert 'performer/observation exclusions' in (tmp_path/'winner.yml').read_text()
    assert negative in summary(result)
    winner = load(tmp_path/'winner.yml')
    with _open_readonly(c.database) as db:
        status = evaluate(winner, goals, db)['trials']['baseline']['goal_result']
    assert status['fully_solved'] and status['observation_exclusions_satisfied']
    assert status['groups'][0]['assigned'] == 6
    assert c.database.read_bytes() == before


def test_solver_repairs_independent_body_observation(tmp_path):
    from tests.test_recognition_body_integration import seeded
    from stash_ai.recognition.merge_report_policy import patched
    c, _, bodies, _ = seeded(tmp_path)
    c.data['sapiensid']['global_matching']['ambiguity_margin'] = 0.
    with FaceDatabase(c.database) as state:
        for ids, angle in zip(bodies, (0, 10, 25)):
            vector = np.zeros(4096, dtype='<f4')
            vector[:2] = [math.cos(math.radians(angle)), math.sin(math.radians(angle))]
            for oid in ids:
                state.connection.execute('UPDATE body_observations SET embedding=? WHERE id=?', (vector.tobytes(), oid))
        state.connection.commit()
    before = c.database.read_bytes()
    with _open_readonly(c.database) as db:
        goals = resolve(db, c, [['900', '901']], observation_exclusions=[('900', bodies[2][0])])
        result = solve(c, goals, db, maximum=16)
        assert not result['simulation']['trials']['baseline']['goal_result']['observation_exclusions_satisfied']
        assert result['status'] == 'fully_solved'
        changes = {r['parameter']: r['proposed'] for r in result['recommendation']['changes']}
        checked = evaluate(patched(c, changes), goals, db)['trials']['baseline']['goal_result']
        assert checked['fully_solved'] and checked['groups'][0]['assigned'] == 6
    assert c.database.read_bytes() == before


def test_cli_two_performers_simulation_and_provenance(tmp_path):
    c, _, _, negative = face_fixture(tmp_path)
    argv = ['report-merge', '--face-config', str(c.source_path), '--faces-db', str(c.database),
        '--performer-id', '900', '--performer-id', '901', '--simulate']
    hashes = []
    for index, extra in enumerate(([], ['--exclude-performer-observation', '900:' + negative])):
        output = tmp_path/f'report-{index}.json'
        assert main(argv + extra + ['--output', str(output)]) == 0
        result = json.loads(output.read_text())
        if extra:
            assert result['observation_exclusions'][0]['observation_ids'] == [negative]
            assert 'excluded_observation_assigned' in {r['reason'] for r in result['explanation']['unresolved']}
            hashes.append(result['provenance']['goals_sha256'])
    # A different negative in the same scene changes the goal fingerprint.
    output = tmp_path/'another.json'
    with _open_readonly(c.database) as db:
        other = db.execute('SELECT id FROM observations WHERE id<>? LIMIT 1', (negative,)).fetchone()[0]
    assert main(argv + ['--exclude-performer-observation', '900:' + other, '--output', str(output)]) == 0
    assert json.loads(output.read_text())['provenance']['goals_sha256'] != hashes[0]


def test_resume_rejects_changed_negative_and_hints_respect_budget(tmp_path):
    from stash_ai.recognition.merge_exclusion_feedback import analyse
    from stash_ai.recognition.merge_report_policy import patched
    c, _, ids, negative = face_fixture(tmp_path)
    c = patched(c, {'sapiensid.execution.max_candidate_pairs': 2})
    with _open_readonly(c.database) as db:
        goals = resolve(db, c, [['900', '901']], observation_exclusions=[('900', negative)])
        hints = analyse(db, c, goals)
        assert hints['comparisons'] <= 2 and hints['truncated']
        assert all(r['requested_observation_ids'] == [negative] for r in hints['evidence'])
        path = tmp_path/'checkpoint.json'
        original = solve(c, goals, db, maximum=1, checkpoint=path)
        assert original['status'] == 'best_effort'
        assert not original['best_result']['fully_solved']
        assert 'excluded_observation_assigned' in {r['reason'] for r in original['best_result']['unmet_conditions']}
        resumed = solve(c, goals, db, maximum=1, checkpoint=path, resume=True)
        assert resumed['search']['restored_trials'] == 1
        assert resumed['best_result'] == original['best_result']
        changed = resolve(db, c, [['900', '901']], observation_exclusions=[('900', ids[2][1])])
        with pytest.raises(ValueError, match='incompatible'):
            solve(c, changed, db, maximum=1, checkpoint=path, resume=True)
