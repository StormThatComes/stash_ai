"""Face/body role counts describe final memberships without changing their score."""
import copy
import json
import sqlite3

import numpy as np
import pytest

from tests.test_recognition_merge_solver_hybrid import prepared
from tests.test_recognition_merge_solver_exploration import auxiliary_goals
from tests.test_recognition_merge_report import seeded
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.fragmentation_report import _open_readonly
from stash_ai.recognition.merge_goals import resolve, assess, objective, unmet_conditions, evaluate
from stash_ai.recognition.merge_goal_output import group_summary_lines, summary
from stash_ai.recognition.person_signal import save, contract


@pytest.mark.parametrize('policy', ['identity', 'strict'])
def test_counts_distinguish_membership_from_compatible_body_availability(tmp_path, policy):
    config, people = prepared(tmp_path, [[1.,0,0,0], [0,1.,0,0]], [None,None], count=8)
    with FaceDatabase(config.database) as state:
        db = state.connection
        ids = [r[0] for r in db.execute('SELECT o.id FROM observations o JOIN members m ON m.observation=o.id '
                                      'WHERE m.person=? ORDER BY o.cue', (people[0],))]
        for oid in ids[3:6]:
            db.execute("UPDATE observations SET quality=json_set(quality,'$.pose_deviation',.4) WHERE id=?", (oid,))
        db.execute("UPDATE candidates SET payload=json_set(payload,'$.reason','hybrid_identity') "
                   "WHERE json_extract(payload,'$.observation')=?", (ids[6],))
        db.execute("UPDATE observations SET rejected='invalid_alignment' WHERE id=?", (ids[7],))
        for i in (0,3,5,6):
            save(db, ids[i], contract(config), {}, np.pad([1.], (0,4095)))
        save(db, ids[1], contract(config), dict(reason='person_not_found'))
        save(db, ids[2], 'old-contract', {}, np.pad([1.], (0,4095)))
        # A stale ready row must not override an unavailable current-contract row.
        save(db, ids[1], 'old-contract', {}, np.pad([1.], (0,4095)))
        db.commit()
    before = config.database.read_bytes()
    with _open_readonly(config.database) as db:
        queries = []
        db.set_trace_callback(queries.append)
        goals = resolve(db, config, [['900'], ['901']], policy=policy)
    body_queries = [sql for sql in queries if 'FROM person_evidence' in sql]
    assert body_queries and all('embedding' not in sql and 'crop' not in sql for sql in body_queries)
    observations = {r['id']:'scene' for g in goals['groups'] for m in g['members'] for r in m['observations']}
    predicted = dict.fromkeys(observations, 'b')
    predicted.update(dict(zip(ids, ['a',None,None,'b','a',None,'fragment',None])))
    del observations[ids[2]]
    result = assess(goals, observations, {}, predicted)
    data = result['groups'][0]['sample_assignment']
    assert data['target_identity'] == 'a'
    assert data['requested_ids'] == ['900'] and data['performer_ids'] == ['900']
    assert data['face']['anchor'] == dict(total=3,assigned=1,to_target=1,other=0,withheld=1,missing=1,fragments=1,cross_set=0)
    assert data['face']['auxiliary'] == dict(total=3,assigned=2,to_target=1,other=1,withheld=1,missing=0,fragments=2,cross_set=1)
    assert data['face']['other']['total'] == 2
    anchor, auxiliary = data['sapiensid']['anchor'], data['sapiensid']['auxiliary']
    assert anchor['availability'] == dict(ready=1,unavailable=1,missing=0,incompatible=1,disabled=0)
    assert anchor['total'] == anchor['assigned'] == anchor['to_target'] == 1
    assert auxiliary['availability']['ready'] == 2 and auxiliary['availability']['missing'] == 1
    assert auxiliary['assigned'] == auxiliary['other'] == auxiliary['withheld'] == auxiliary['cross_set'] == 1
    assert auxiliary['to_target'] == 0  # Body must use the face/set target, not its own largest fragment.
    legacy = dict(goals)
    del legacy['sample_diagnostics']
    without = assess(legacy, observations, {}, predicted)
    assert objective(result) == objective(without) and unmet_conditions(result) == unmet_conditions(without)
    assert config.database.read_bytes() == before
    rendered = '\n'.join(group_summary_lines(result['groups'][0]))
    assert 'Face anchor: assigned=1/3 (33.3%)' in rendered
    assert 'SapiensID auxiliary: ready=2/3 assigned=1/2 (50.0%) target=0/2 other=1' in rendered
    assert 'unavailable=1 no_evidence=0 incompatible=1' in rendered
    assert 'Face other:' in rendered
    json.dumps(result, allow_nan=False)


def test_target_prefers_more_performers_over_sample_heavy_fragment_and_reports_ties(tmp_path):
    config, people = seeded(tmp_path, vectors=[[1.,0,0,0],[0,1.,0,0],[0,0,1.,0]], count=3)
    with _open_readonly(config.database) as db:
        goals = resolve(db, config, [people], by_performer=False)
    observations = {r['id']:'scene' for m in goals['groups'][0]['members'] for r in m['observations']}
    predicted = {r['id']:str(i) for i,m in enumerate(goals['groups'][0]['members']) for r in m['observations']}
    result = assess(goals, observations, {}, predicted)
    assert result['groups'][0]['sample_assignment']['target_ties'] == ['0','1','2']
    assert 'target_ties=3' in '\n'.join(group_summary_lines(result['groups'][0]))
    for m in goals['groups'][0]['members'][:2]:
        for r in m['observations']:
            predicted[r['id']] = 'shared'
    for i in range(20):
        row = dict(id=f'copy-{i}', sample_role='anchor', body_state='disabled')
        goals['groups'][0]['members'][2]['observations'].append(row)
        observations[row['id']], predicted[row['id']] = 'scene', 'large'
    result = assess(goals, observations, {}, predicted)
    assert result['groups'][0]['sample_assignment']['target_identity'] == 'shared'


def test_roles_stay_frozen_when_trial_promotes_auxiliary_and_ready_can_be_missing_from_replay(tmp_path):
    from stash_ai.recognition.merge_report_policy import patched
    config, goals = auxiliary_goals(tmp_path)
    frozen = copy.deepcopy(goals)
    with _open_readonly(config.database) as db:
        trial = evaluate(patched(config, {'clustering.max_anchor_pose_deviation':.4}), goals, db)
    data = trial['trials']['baseline']['goal_result']['groups'][0]['sample_assignment']
    assert data['face']['anchor']['total'] == data['face']['auxiliary']['total'] == 3
    assert goals == frozen
    missing = assess(goals, {}, {}, {})['groups'][0]['sample_assignment']
    assert missing['sapiensid']['anchor']['availability']['ready'] == 3
    assert missing['sapiensid']['anchor']['missing'] == 3
    assert missing['sapiensid']['anchor']['assigned'] == 0
    assert missing['target_identity'] is None


def test_missing_body_table_disabled_signal_and_zero_counts_are_explicit(tmp_path):
    from stash_ai.recognition.merge_goal_samples import freeze
    config, people = seeded(tmp_path)
    with _open_readonly(config.database) as db:
        goals = resolve(db, config, [['900','900'],['901']])
    result = assess(goals, {}, {}, {})
    data = result['groups'][0]['sample_assignment']
    assert data['requested_ids'] == ['900']
    rendered = '\n'.join(group_summary_lines(result['groups'][0]))
    assert 'Face auxiliary: assigned=0/0 (0.0%)' in rendered and 'SapiensID: disabled' in rendered
    rows = [dict(id='x',status='complete',rejected=None,reference_authority=1,quality='{}')]
    with sqlite3.connect(':memory:') as db:
        freeze(db, config, rows, dict(sapiensid_enabled=True,sapiensid_contract=contract(config)))
    assert rows[0]['body_state'] == 'missing'


def test_winner_role_counters_survive_checkpoint_resume_and_interruption(tmp_path):
    from stash_ai.recognition.merge_solver import solve
    from stash_ai.recognition.merge_solver_checkpoint import winner_file
    config, goals = auxiliary_goals(tmp_path, sets=2)
    checkpoint, output = tmp_path/'solver.json', tmp_path/'winner.yml'
    def stop(name, stage, item):
        if stage == 'complete' and name == 'trial-1':
            raise KeyboardInterrupt
    with _open_readonly(config.database) as db:
        first = dict(goals, **solve(config, goals, db, targeted_trials=0, feasibility_trials=0, maximum=4, checkpoint=checkpoint,
                                   winner_writer=winner_file(output), progress=stop))
        resumed = dict(goals, **solve(config, goals, db, targeted_trials=0, feasibility_trials=0, maximum=4, checkpoint=checkpoint, resume=True,
                                     winner_writer=winner_file(output, resume=True)))
    assert first['interrupted'] and first['selected_trial'] == 'trial-1'
    assert first['simulation']['trials']['trial-1']['goal_result'] == resumed['simulation']['trials']['trial-1']['goal_result']
    assert 'SapiensID auxiliary: ready=3/3 assigned=3/3 (100.0%) target=3/3' in summary(first)
    assert 'SapiensID auxiliary:' in summary(resumed)
