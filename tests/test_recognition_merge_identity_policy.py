"""Frozen identity evidence must beat perfect retention of noisy membership."""
import copy
import json

import pytest

from stash_ai.recognition.merge_goal_identity import POLICY, freeze
from stash_ai.recognition.merge_goal_sets import freeze_set
from stash_ai.recognition.merge_goals import assess, objective, resolve, evaluate, unmet_conditions
from stash_ai.recognition.fragmentation_report import _open_readonly
from stash_ai.recognition.database import FaceDatabase
from tests.test_recognition_merge_report import seeded
from tests.test_recognition_merge_best_effort import incomplete_source


def member(name, core=3, soft=0, *, scenes=None, pixels=None):
    rows = [dict(id=f'{name}-{i}', scan='scan-'+name, scene=(scenes[i] if scenes else name),
                 seconds=i, quality=json.dumps(dict(evidence_pixel_sha256=pixels[i])) if pixels else '{}',
                 goal_role='reference' if i < core else 'uncertain') for i in range(core+soft)]
    return dict(identity_id=name, performer_id=name, observations=rows, identity_evidence=freeze(rows))


def goals(*groups):
    return dict(goal_policy=dict(POLICY), groups=[dict(index=i, members=g,
        identity_evidence=freeze_set([r for m in g for r in m['observations']])) for i,g in enumerate(groups)])


def assignments(goal):
    return {r['id']:m['identity_id'] for g in goal['groups'] for m in g['members'] for r in m['observations']}


def status(goal, predicted, *, missing=()):
    observations = {k:'scene' for k in assignments(goal) if k not in missing}
    return assess(goal, observations, {}, predicted)


@pytest.mark.parametrize('auxiliary_destination', [None, 'merged-b'])
@pytest.mark.parametrize('merge_count', [2, 9, 18])
def test_reducing_fragmentation_of_18_identities_beats_preserving_one_noisy_auxiliary(auxiliary_destination, merge_count):
    goal = goals([member('a', core=1, soft=1)], [member(f'b{i}', core=1) for i in range(18)])
    before = assignments(goal)
    after = dict(before)
    for i in range(merge_count):
        after[f'b{i}-0'] = 'merged-b'
    after['a-1'] = auxiliary_destination
    baseline, improved = status(goal, before), status(goal, after)
    assert objective(improved) > objective(baseline)
    assert improved['admissible'] and improved['disjoint']
    assert improved['fully_solved'] == (merge_count == 18)
    assert improved['groups'][1]['consolidated_performers'] == merge_count
    assert improved['groups'][0]['largest_complete_performers'] == 0
    assert improved['groups'][0]['soft_lost_units'] == 1
    assert not improved['strict_fully_solved']
    assert improved['strict_disjoint'] == (auxiliary_destination is None)
    assert [v['reason'] for v in unmet_conditions(improved)] == ([] if merge_count == 18 else ['incomplete_identity_set'])
    # The previous objective really would select the fragmented baseline.
    strict = dict(goal, goal_policy=dict(name='strict'))
    assert objective(status(strict, before)) > objective(status(strict, after))


def test_confirmed_cross_set_merge_remains_a_conflict():
    goal = goals([member('a')], [member('b')])
    predicted = dict.fromkeys(assignments(goal), 'outsider')
    result = status(goal, predicted)
    assert not result['disjoint'] and not result['admissible']
    assert result['cross_set_evidence_pairs'] == 9
    assert any(v['reason'] == 'cross_set_merge' for v in unmet_conditions(result))


def test_soft_budget_excess_is_a_tiebreaker_after_set_consolidation():
    goal = goals([member('a', core=3, soft=100), member('b')])
    before = assignments(goal)
    after = dict.fromkeys(before, 'merged')
    for i in range(3,93):
        after[f'a-{i}'] = None
    result = status(goal, after)
    assert result['admissible'] and result['fully_solved']
    assert result['groups'][0]['soft_budget_excess'] == 70
    assert objective(result) > objective(status(goal, before))
    assert objective(status(goal, dict.fromkeys(before, 'merged'))) > objective(result)
    assert not unmet_conditions(result)
    for i in range(3):
        after[f'b-{i}'] = None
    assert status(goal, after)['groups'][0]['represented_performers'] == 1
    assert not status(goal, after)['admissible']  # b's unique scene really disappeared.


def test_confirmed_scene_cannot_disappear_within_global_outlier_budget():
    goal = goals([member('a', core=10, scenes=['first']*8+['second']*2)])
    predicted = assignments(goal)
    predicted['a-8'] = predicted['a-9'] = None
    result = status(goal, predicted)
    evidence = result['groups'][0]['members'][0]['identity_evidence']
    assert evidence['core_assigned'] == evidence['core_required'] == 8
    assert not evidence['represented'] and not result['fully_solved']
    assert unmet_conditions(result)[0] == dict(set=0, reason='set_scene_not_preserved', scenes=['second'])


def test_reference_eligibility_is_not_a_perfect_label():
    from stash_ai.recognition.merge_goal_identity import destination_owners
    goal = goals([member('a')], [member('b')])
    predicted = assignments(goal)
    predicted['a-2'] = 'b'
    result = status(goal, predicted)
    assert result['fully_solved'] and not result['strict_disjoint']
    assert destination_owners(goal, predicted) == {'a': {0}, 'b': {1}}
    predicted['a-1'] = 'b'
    assert not status(goal, predicted)['disjoint']
    assert destination_owners(goal, predicted)['b'] == {0, 1}


def test_auxiliary_only_set_has_fallback_quorum_for_full_merge_and_cannot_vanish():
    goal = goals([member('a', core=0, soft=5)])
    predicted = assignments(goal)
    predicted['a-4'] = None
    result = status(goal, predicted)
    evidence = result['groups'][0]['members'][0]['identity_evidence']
    assert evidence['basis'] == 'fallback_membership'
    assert result['fully_solved']
    predicted['a-3'] = None
    assert status(goal, predicted)['admissible'] and not status(goal, predicted)['fully_solved']
    assert not status(goal, dict.fromkeys(predicted))['fully_solved']


def test_empty_performer_cannot_satisfy_identity_goal():
    goal = goals([member('empty', core=0)])
    assert not status(goal, {})['admissible']
    assert unmet_conditions(status(goal, {}))[0]['reason'] == 'empty_identity_set'


def test_duplicate_crops_do_not_inflate_votes_and_conflicting_copies_do_not_vote_twice():
    m = member('a', core=5, pixels=['a'*64]*4+['b'*64])
    assert len(m['identity_evidence']['core_units']) == 2
    assert freeze(list(reversed(m['observations']))) == m['identity_evidence']
    goal = goals([m])
    predicted = assignments(goal)
    assert status(goal, predicted)['fully_solved']
    predicted['a-4'] = None
    assert not status(goal, predicted)['fully_solved']  # Four copies cannot outvote the second sample.
    predicted['a-4'], predicted['a-0'] = 'a', 'other'
    evidence = status(goal, predicted)['groups'][0]['members'][0]['identity_evidence']
    assert evidence['core_assigned'] == 1


def test_frozen_roles_respect_production_authority_and_survive_trial_quality_changes(tmp_path):
    from stash_ai.recognition.merge_report_policy import patched
    config, people = seeded(tmp_path)
    with FaceDatabase(config.database) as state:
        db = state.connection
        rows = db.execute('SELECT observation FROM members WHERE person=? ORDER BY observation', (people[0],)).fetchall()
        auxiliary, contextual = rows[0][0], rows[1][0]
        db.execute("UPDATE observations SET quality=json_set(quality,'$.pose_deviation',.4) WHERE id=?", (auxiliary,))
        db.execute("UPDATE candidates SET payload=json_set(payload,'$.reason','hybrid_query') "
                   "WHERE json_extract(payload,'$.observation')=?", (contextual,))
        db.commit()
    with _open_readonly(config.database) as db:
        goal = resolve(db, config, [people], by_performer=False)
        evidence = goal['groups'][0]['members'][0]['identity_evidence']
        assert len(evidence['core_units']) == 1 and len(evidence['soft_units']) == 2
        snapshot = copy.deepcopy(goal)
        relaxed = patched(config, {'clustering.max_anchor_pose_deviation': .5})
        trial = evaluate(relaxed, goal, db)['trials']['baseline']['goal_result']
    assert goal == snapshot
    assert trial['groups'][0]['members'][0]['identity_evidence']['core_units'] == 1


def test_soft_correction_does_not_relabel_feedback_ownership(tmp_path):
    from stash_ai.recognition.merge_solver import rebuilt_goals
    from stash_ai.recognition.merge_solver_feedback import QueryFeedback
    config, people = seeded(tmp_path)
    with FaceDatabase(config.database) as state:
        db = state.connection
        soft = db.execute('SELECT observation FROM members WHERE person=? LIMIT 1', (people[0],)).fetchone()[0]
        db.execute("UPDATE observations SET quality=json_set(quality,'$.pose_deviation',.4) WHERE id=?", (soft,))
        goal = resolve(db, config, [[p] for p in people], by_performer=False)
        db.execute('UPDATE members SET person=? WHERE observation=?', (people[1], soft))
        rebuilt = rebuilt_goals(db, goal)
        assert [[m['identity_id'] for m in g['members']] for g in rebuilt['groups']] == [[p] for p in people]
        query = QueryFeedback(goal)
        assert soft not in query.sets
        assert sum(len(m['observations']) for g in query.goals['groups'] for m in g['members']) == 5


@pytest.mark.parametrize('mode', ['full', 'diff'])
def test_real_replay_selects_merge_despite_missing_auxiliary_and_keeps_winner_on_interrupt(tmp_path, mode):
    import uuid
    import yaml
    from stash_ai.recognition.config import load
    from stash_ai.recognition.merge_solver import solve
    from stash_ai.recognition.merge_solver_checkpoint import winner_file
    from stash_ai.recognition.merge_report_policy import patched
    from stash_ai.recognition.merge_goal_output import export, summary
    config, people = incomplete_source(tmp_path)
    with FaceDatabase(config.database) as state:
        db = state.connection
        db.execute("UPDATE observations SET quality=json_set(quality,'$.pose_deviation',.4) "
                                 "WHERE scan IN (SELECT id FROM scans WHERE scene='99')")
        # The missing historical auxiliary's scene is still represented by
        # another fragment of the same set in its new complete scan.
        scan = db.execute("SELECT id FROM scans WHERE scene='99' ORDER BY rowid DESC LIMIT 1").fetchone()[0]
        rows = db.execute('SELECT o.* FROM observations o JOIN members m ON m.observation=o.id '
                          'WHERE m.person=?', (people[1],)).fetchall()
        for row in rows:
            copied = list(row)
            copied[0], copied[1] = str(uuid.uuid4()), scan
            db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', copied)
            db.execute('INSERT INTO members VALUES (?,?)', (copied[0], people[1]))
        db.execute("UPDATE scans SET status='complete',cursor=?,total=? WHERE id=?", (len(rows), len(rows), scan))
        db.commit()
    before = config.database.read_bytes()
    checkpoint, output = tmp_path/'checkpoint.json', tmp_path/'winner.yml'
    def stop(name, stage, item):
        if stage == 'complete' and item['goal_result']['fully_solved']:
            assert name in output.read_text()
            raise KeyboardInterrupt
    with _open_readonly(config.database) as db:
        goal = resolve(db, config, [people], by_performer=False)
        result = dict(goal, **solve(config, goal, db, maximum=12, progress=stop, checkpoint=checkpoint,
                                   winner_writer=winner_file(output), yaml_mode=mode))
        assert result['interrupted'] and result['best_result']['fully_solved']
        assert result['selected_trial'] != 'baseline'
        selected = result['simulation']['trials'][result['selected_trial']]['goal_result']
        assert selected['groups'][0]['soft_lost_units'] == 1 and not selected['strict_fully_solved']
        text = output.read_text()
        assert 'Goal policy: identity v3' in text and 'soft units lost/reassigned: 1' in text
        assert selected['groups'][0]['covered_scenes'] == selected['groups'][0]['required_scenes'] == 3
        if mode == 'full':
            restored = load(output)
        else:
            patch = yaml.safe_load(text)
            changes = {f'{section}.{key}': value for section, fields in patch.items() for key,value in fields.items()}
            restored = patched(config, changes)
        assert objective(evaluate(restored, goal, db)['trials']['baseline']['goal_result']) == objective(selected)
        resumed = solve(config, goal, db, maximum=12, checkpoint=checkpoint, resume=True,
                        winner_writer=winner_file(output, resume=True), yaml_mode=mode)
        assert resumed['search']['restored_trials'] == len(result['simulation']['trials'])
        assert resumed['best_result']['fully_solved']
        previous_policy = copy.deepcopy(goal)
        previous_policy['goal_policy']['version'] = 1
        with pytest.raises(ValueError, match='incompatible'):
            solve(config, previous_policy, db, maximum=12, checkpoint=checkpoint, resume=True,
                  winner_writer=winner_file(output, resume=True), yaml_mode=mode)
        strict = resolve(db, config, [people], by_performer=False, policy='strict')
        with pytest.raises(ValueError, match='incompatible'):
            solve(config, strict, db, maximum=12, checkpoint=checkpoint, resume=True, yaml_mode=mode,
                  winner_writer=winner_file(output, resume=True))
    export(config, result, tmp_path/'result.json', tmp_path/'export.yml', yaml_mode=mode)
    assert 'consolidated=2/2' in summary(result) and 'largest_complete=1/2' in summary(result)
    assert 'strict_fully_solved=False' in summary(result)
    assert config.database.read_bytes() == before
