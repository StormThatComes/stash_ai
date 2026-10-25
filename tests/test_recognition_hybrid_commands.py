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


def test_real_run_collection_merge_publication_repeat_and_maintenance(tmp_path,monkeypatch):
    from tests.test_recognition_rebuild import library,changed,two_people
    from tests.test_recognition_run import run
    from tests.test_recognition_person_collection import Person
    from stash_ai.recognition.config import plain
    from stash_ai.recognition.run import execute
    from stash_ai.recognition.sync_publications import execute as sync
    from stash_ai.recognition.cleanup_orphans import execute as cleanup
    from tests.test_recognition_model_cleanup import inventory,cleanup_args
    from tests.test_recognition_publication_sync import args as sync_args
    from types import SimpleNamespace
    config,client,services,backends=library(tmp_path,monkeypatch,two_people())
    config=changed(config,recognition={'match_threshold':.95})
    assert run(config,services,'1','2')==0
    originals={s:client.links[s][-1] for s in ('1','2')}
    assert originals['1']!=originals['2']
    data=plain(config.data);data['sapiensid']['enabled']=True
    config=replace(config,data=data)
    models=[]
    def factory(*args,**kwargs):
        model=Person();models.append(model);return model
    monkeypatch.setattr('stash_ai.recognition.sapiensid_backend.SapiensIDFaces',factory)
    loaded=len(backends)
    assert run(config,services,'1','2')==0
    assert len(backends)==loaded  # Collection reused all face embeddings.
    assert all(m.closed for m in models)
    assert client.links['1']==['77',originals['1']] and client.links['2']==[originals['1']]
    assert not any(w[0]=='merge' for w in client.writes)
    assert originals['2'] in client.profiles
    writes=list(client.writes);calls=sum(m.calls for m in models)
    assert run(config,services,'1','2')==0
    assert client.writes==writes and sum(m.calls for m in models)==calls
    # Maintenance must neither initialize the supplemental encoder nor collect media.
    monkeypatch.setattr('stash_ai.recognition.sapiensid_backend.SapiensIDFaces',lambda *a,**k:pytest.fail('maintenance loaded model'))
    assert execute(SimpleNamespace(command='portraits',scene_id=['1','2'],all_scenes=False,
                                  performer_id=None,progress_style='off'),config,services=services)==0
    assert sync(sync_args(apply=True),config,services=services)==0
    inventory(client)
    assert cleanup(cleanup_args(apply=True),config,services=services)==0
    assert not any(w[0]=='delete' for w in client.writes)


def test_identity_export_preserves_body_pixels_and_attachment_role(tmp_path):
    from io import BytesIO
    from PIL import Image
    from stash_ai.recognition.hybrid_merge import apply
    from stash_ai.recognition.identity_report import export
    config,people=prepared(tmp_path)
    with Image.new('RGB',(40,80),(10,20,30)) as image:
        buffer=BytesIO();image.save(buffer,format='PNG')
    with FaceDatabase(config.database) as state:
        db=state.connection
        db.execute('UPDATE person_evidence SET crop=?',(buffer.getvalue(),));db.commit()
        apply(state,[dict(scan=r['id'],scene=r['scene']) for r in db.execute('SELECT * FROM scans')],config)
    before=config.database.read_bytes()
    result=export(config,tmp_path/'export',identity_id=people[0])
    assert result['summary']['roles']['attachment']==3
    assert sum(s['current_reference_selected'] for s in result['samples'])==3
    for sample in result['samples']:
        body=sample['person_evidence']['image']
        assert body['status']=='exported'
        with Image.open(tmp_path/'export'/body['file']) as image:
            assert image.size==(40,80) and image.getpixel((0,0))==(10,20,30)
    assert config.database.read_bytes()==before
