from dataclasses import replace
import json
import numpy as np
import pytest

from tests.test_recognition_hybrid_merge import prepared
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.person_signal import save,contract,coverage
from stash_ai.recognition.merge_report import build
from stash_ai.recognition.merge_report_policy import patched,proposal
from stash_ai.recognition.replay_simulation import replay
from stash_ai.recognition.threshold_tuning import tune


def test_report_replay_and_tuning_share_hybrid_production(tmp_path,monkeypatch):
    config,people=prepared(tmp_path)
    before=config.database.read_bytes()
    monkeypatch.setattr('stash_ai.recognition.models.load_backend',lambda *a:pytest.fail('offline inference'))
    report=build(config,*people,simulate=True)
    hybrid=next(r for r in report['recommendations'] if r['route']=='sapiensid_hybrid')
    assert hybrid['replay_verified'] and hybrid['changes']==[]
    assert report['simulation']['trials']['baseline']['hybrid']['ready']==6
    result=tune(config,{'sapiensid.matching.face_floor':[.8]},max_trials=2)
    assert result['trials']['trial-1']['versus_baseline']['split_pairs']==9
    assert result['recommendation'] is None
    assert config.database.read_bytes()==before
    assert 'private-crop' not in json.dumps(result)


def test_measured_hybrid_proposal_and_nested_yaml(tmp_path):
    config,people=prepared(tmp_path)
    config=patched(config,{'sapiensid.matching.face_floor':.8})
    report=build(config,*people,simulate=True)
    hybrid=next(r for r in report['recommendations'] if r['route']=='sapiensid_hybrid')
    assert hybrid['yaml_patch']['sapiensid']['matching']['face_floor']<=.7
    assert hybrid['replay_verified']
    assert proposal(config,{'sapiensid.matching.min_support':2})['yaml_patch']=={'sapiensid':{'matching':{'min_support':2}}}


def test_missing_body_evidence_is_visible_and_never_manufactured(tmp_path):
    config,people=prepared(tmp_path)
    with FaceDatabase(config.database) as state:
        db=state.connection
        identifier=db.execute('SELECT id FROM observations LIMIT 1').fetchone()[0]
        db.execute('DELETE FROM person_evidence WHERE observation=?',(identifier,));db.commit()
    report=build(config,*people)
    route=next(r for r in report['paths'] if r['route']=='sapiensid_hybrid')
    assert not route['recommendations'] and 'missing_person_evidence' in route['unresolved_reasons']
    result=replay(config,{})
    assert result['trials']['baseline']['hybrid']['missing']==1
    assert result['trials']['baseline']['versus_retained']['identities']==2


def test_evaluation_and_merge_goals_use_retained_body_features(tmp_path):
    from stash_ai.recognition.offline_evaluation import evaluate
    from stash_ai.recognition.merge_goals import resolve,evaluate as goals_evaluate,fingerprint
    from stash_ai.recognition.merge_solver import measured_options,neighbours
    config,people=prepared(tmp_path)
    with FaceDatabase(config.database) as state:
        db=state.connection
        labels={r[0]:'same' for r in db.execute('SELECT id FROM observations')}
        goals=resolve(db,config,[people],by_performer=False)
        result=goals_evaluate(config,goals,db)
        assert result['trials']['baseline']['goal_result']['fully_solved']
        options=measured_options(db,config,goals)
        assert 'sapiensid.matching.person_threshold' in options
        assert list(neighbours(config,{},options))
        old=fingerprint(db,config,'scope')
        db.execute("UPDATE person_evidence SET metadata='{}'");db.commit()
        assert fingerprint(db,config,'scope')!=old
    result=evaluate(config,labels,batch_sizes=(0,),repeat_runs=2)
    assert all(t['hybrid']['ready']==6 for t in result['trials'].values())
    assert all(t['reason_counts']['hybrid_identity']==3 for t in result['trials'].values())


def test_replay_refuses_changed_person_contract(tmp_path):
    from stash_ai.recognition.config import plain
    config,_=prepared(tmp_path)
    data=plain(config.data);data['sapiensid']['pose']['sha256']='2'*64
    with pytest.raises(ValueError,match='contracts'):
        replay(config,{'bad':replace(config,data=data)})
