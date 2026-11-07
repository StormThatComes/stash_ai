import json
import sqlite3
from contextlib import closing

import pytest

from stash_ai.recognition.fragmentation_report import _open_readonly
from stash_ai.recognition.solver_validation import validate_winner
from tests.test_recognition_offline_evaluation import fixture


def test_holdout_replays_one_winner_and_removes_only_private_cues(tmp_path):
    cfg, labels = fixture(tmp_path)
    with closing(_open_readonly(cfg.database)) as db:
        scope, namespace = db.execute('SELECT scope,namespace FROM scans LIMIT 1').fetchone()
    manifest = tmp_path/'holdout.json'
    manifest.write_text(json.dumps(dict(database=str(cfg.database), scenes=['1','2'],
        observations={oid:{'truth':label,'slices':['back','entangled']} for oid,label in labels.items()},
        series=[['1',0,0]])))
    goal = dict(scope=scope,namespace=namespace,selected_trial='trial1',best_result={'changes':[]})
    with closing(sqlite3.connect(':memory:')) as training:
        training.execute('CREATE TABLE scans(scope,namespace,scene)')
        events = []
        result = validate_winner(cfg,training,goal,manifest,progress=events.append)
        assert [e.split(' · ')[0] for e in events] == ['Holdout baseline', 'Holdout winner',
            'Holdout incremental face_first','Holdout incremental rear_first','Holdout omission 1/1']
        assert result['incremental']['face_first']['order'] == list(reversed(result['incremental']['rear_first']['order']))
        assert {row['mode'] for row in result['omissions']}=={'empty','rejected'}
        from stash_ai.recognition.solver_explanation import holdout_summary
        text = holdout_summary(result)
        assert 'back (9)' in text and 'entangled (9)' in text and '1:0-0 lost=' in text
        assert result['metrics']['winner/joint']['back']['observations']==9
        assert result['omissions'][0]['scored']==8
        assert result['omissions'][0]['excluded_labels']==1
        with closing(_open_readonly(cfg.database)) as source:
            assert not source.execute("SELECT 1 FROM observations WHERE rejected='validation_omission'").fetchone()
        training.execute('INSERT INTO scans VALUES (?,?,?)',(scope,namespace,'2'))
        with pytest.raises(ValueError,match='training snapshot'):
            validate_winner(cfg,training,goal,manifest)


def test_back_slice_counts_joins_to_frontal_partner():
    from stash_ai.recognition.solver_validation import _metrics
    labels = {'back':'A','face':'A','partner':'B'}
    tagged = {k:{'slices':['back'] if k=='back' else []} for k in labels}
    scores = _metrics(labels, tagged, dict.fromkeys(labels,('1',0)),
                      {k:{'accepted_identity':'mixed'} for k in labels})
    assert scores['back']['pairs_touching_slice'] == dict(true_pairs=1,joined_pairs=2,
        false_merge_pairs=1,recall=1.,precision=.5)


def test_same_database_holdout_excludes_raw_inputs_before_solver_replay(tmp_path):
    from stash_ai.recognition.replay_simulation import prepared_history
    from stash_ai.recognition.database import FaceDatabase
    from types import SimpleNamespace
    cfg, labels = fixture(tmp_path)
    with closing(_open_readonly(cfg.database)) as source:
        scope,namespace = source.execute('SELECT scope,namespace FROM scans LIMIT 1').fetchone()
        installation = FaceDatabase.installation.fget(SimpleNamespace(connection=source))
        held = {r[0] for r in source.execute("SELECT o.id FROM observations o JOIN scans s ON s.id=o.scan WHERE s.scene='2'")}
        plan = tmp_path/'holdout.json'
        plan.write_text(json.dumps(dict(scenes=['2'],observations={
            oid:{'truth':labels[oid],'slices':['back']} for oid in held})))
        with prepared_history(source,cfg,scope,namespace,installation,excluded_scenes=['2']) as training:
            assert {r[0] for r in training.execute('SELECT scene FROM scans')} == {'1'}
            assert not held & {r[0] for r in training.execute('SELECT id FROM observations')}
            result = validate_winner(cfg,training,dict(scope=scope,namespace=namespace,
                                     best_result={'changes':[]}),plan)
            assert result['metrics']['winner/joint']['back']['observations']==len(held)


def test_provisional_partner_relations_cannot_score_as_production_membership():
    from stash_ai.recognition.solver_validation import _metrics, _partition_delta
    labels = {'face':'A', 'back':'A', 'back_projection':'A', 'partner':'B'}
    samples = {k:{'slices':['entangled']} for k in labels}
    places = {k:('1', 0 if k=='face' else 1) for k in labels}
    triples = [('face', k, 'partner') for k in ('back','back_projection')]
    decisions = {k:dict(accepted_identity=None, identity='x', status='provisional') for k in labels}
    values = _metrics(labels, samples, places, decisions, partner_sets=triples)
    assert values['all']['coverage'] == 0
    assert values['provisional_only']['assigned'] == 4
    joined = {k:dict(accepted_identity='x', identity='x', status='accepted') for k in labels}
    scores = _metrics(labels, samples, places, joined, partner_sets=triples)['partners']
    assert (scores['target_pairs'], scores['recovered'], scores['partner_pairs'], scores['false_joins']) == (1,1,2,2)
    assert _partition_delta(decisions, joined, labels) == dict(lost_pairs=0, joined_pairs=6)


def test_single_physical_target_cannot_recover_itself():
    from stash_ai.recognition.solver_validation import partner_metrics
    ids = ['face', 'body', 'partner']
    labels = dict(zip(ids,['A','A','B']))
    decisions = {k:{'accepted_identity':v} for k,v in labels.items()}
    result = partner_metrics([ids],labels,dict.fromkeys(ids,('scene',3)),decisions,())
    assert result['target_pairs']==result['recovered']==0
    assert result['partner_pairs']==1


def test_turn_and_partner_recovery_give_one_vote_per_cue_and_performer():
    from stash_ai.recognition.sequence_validation import sequence_metrics
    labels,locations,samples,decisions = {},{},{},{}
    for cue in range(3):
        for truth in ('A','B'):
            oid=f'{truth}-{cue}'
            labels[oid],locations[oid]=truth,('1',cue)
            samples[oid]={'slices':['back','entangled'] if cue==1 else []}
            decisions[oid]={'accepted_identity':truth}
    def measure():
        return sequence_metrics(labels,samples,locations,decisions)
    initial=measure()
    assert initial['turns']==2 and initial['macro_turn_recovery']==1
    assert initial['crossings']==initial['both_recovered']==1
    decisions['B-1']['accepted_identity']='A'
    mixed=measure()
    assert mixed['macro_turn_recovery']==.5 and mixed['macro_crossing_recovery']==0
    assert mixed['id_switches']==2
    decisions['B-1']['accepted_identity']=None
    assert measure()['both_recovered']==1
    decisions['A-2']['accepted_identity']='B'
    assert measure()['both_recovered']==0
