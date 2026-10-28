import json
import uuid

import numpy as np
import pytest

from stash_ai.recognition.body_parameters import LOCAL,GLOBAL,PARAMETERS
from stash_ai.recognition.body_solver import analyse
from stash_ai.recognition.database import FaceDatabase,encoded
from stash_ai.recognition.fragmentation_report import _open_readonly
from stash_ai.recognition.merge_goals import resolve,evaluate
from stash_ai.recognition.merge_report_policy import patched,parameter
from stash_ai.recognition.merge_solver_domains import registry,normalize,dimensions
from stash_ai.recognition.merge_solver_search import hybrid_neighbours,refinement
from stash_ai.recognition.models import namespace
from tests.test_recognition_body_integration import seeded


def test_all_retained_body_parameters_reach_search_and_tpe_dimensions(tmp_path):
    c,_,_,_=seeded(tmp_path)
    c.data['sapiensid']['identities']['min_appearance_distance']=4
    with _open_readonly(c.database) as db:
        goals=resolve(db,c,[['900','901'],['902']])
        hints=analyse(db,c,goals)
    assert hints['options'].keys()==PARAMETERS
    proposals=list(hybrid_neighbours(c,{},hints['options'],hints['proposals']))
    assert PARAMETERS <= {k for p in proposals for k,v in p.items() if v!=parameter(c.data,k)}
    domains=registry(c)
    for p in proposals:
        for k,v in normalize(c,p).items():
            assert domains[k].low<=v<=domains[k].high
            assert type(v) is domains[k].kind
    selected=set().union(*(dimensions(c,hints['options'],{},i) for i in range(0,40,5)))
    assert selected==PARAMETERS
    from stash_ai.recognition.merge_solver_bidirectional import relaxed_seeds
    seeds=list(relaxed_seeds(c,dict(options=hints['options'])))
    assert PARAMETERS <= set().union(*(p.keys() for p in seeds))
    for key in ('min_appearance_distance','anchor_min_keypoints'):
        key='sapiensid.identities.'+key
        patches=list(refinement(c,{key:1},hints['options']))
        assert all(type(p[key]) is int for p in patches)


def test_disabled_routes_and_input_portrait_settings_are_not_solver_dimensions(tmp_path):
    c,_,_,_=seeded(tmp_path)
    c.data['sapiensid']['global_matching']['enabled']=False
    with _open_readonly(c.database) as db:
        hints=analyse(db,c,resolve(db,c,[['900']]))
    assert hints['options'].keys()==set(LOCAL)
    assert not set(GLOBAL)&registry(c).keys()
    assert not any('.collection.' in k or k.startswith('portraits.') for k in registry(c))
    c.data['sapiensid']['identities']['enabled']=False
    assert not PARAMETERS&registry(c).keys()


def test_tpe_samples_body_quality_and_deduplication_in_their_native_domains(tmp_path):
    pytest.importorskip('optuna')
    from stash_ai.recognition.merge_solver_sampler import create
    c,_,_,_=seeded(tmp_path)
    options={'sapiensid.identities.'+key:values for key,values in {
        'attach_min_support':[1,2,5], 'min_appearance_distance':[0,4,64],
        'anchor_min_keypoints':[1,10,17], 'anchor_min_sharpness':[0.,100.,200.],
    }.items()}
    history=[({},(0,))]
    patch=create('tpe',42).suggest(c,history,options,{},15)
    assert patch==create('tpe',42).suggest(c,history,options,{},15)
    assert patch.keys()==options.keys()
    domains=registry(c)
    for key,value in patch.items():
        assert type(value) is domains[key].kind
        assert domains[key].low<=value<=domains[key].high
    assert patch['sapiensid.identities.anchor_min_sharpness']>1


@pytest.mark.parametrize('gate,value',[
    ('anchor_min_keypoints',11),('anchor_min_sharpness',101.),('min_appearance_distance',4)])
def test_solver_repairs_body_gate_with_production_replay(tmp_path,monkeypatch,gate,value):
    from stash_ai.recognition.merge_solver import solve
    from stash_ai.recognition.body_observations import contract
    c,_,ids,_=seeded(tmp_path,vectors=[[1,0]])
    c.data['sapiensid']['global_matching']['enabled']=False
    if gate=='min_appearance_distance':
        with FaceDatabase(c.database) as state:
            for oid,hashed in zip(ids[0],['0','1','3']):
                q=json.loads(state.connection.execute('SELECT quality FROM body_observations WHERE id=?',(oid,)).fetchone()[0])
                q['quality']['appearance_hash']=hashed
                state.connection.execute('UPDATE body_observations SET quality=? WHERE id=?',(encoded(q),oid))
            state.connection.commit()
    original_contract=contract(c)
    c.data['sapiensid']['identities'][gate]=value
    assert contract(c)==original_contract
    monkeypatch.setattr('stash_ai.recognition.models.load_backend',lambda *a,**k:pytest.fail('offline inference'))
    monkeypatch.setattr('stash_ai.recognition.sapiensid_assets.install',lambda *a,**k:pytest.fail('offline download'))
    before=c.database.read_bytes()
    with _open_readonly(c.database) as db:
        goals=resolve(db,c,[['900']],policy='strict')
        baseline=evaluate(c,goals,db)['trials']['baseline']['goal_result']
        assert not baseline['fully_solved']
        result=solve(c,goals,db,maximum=24)
        assert result['status']=='fully_solved'
        changes={r['parameter']:r['proposed'] for r in result['recommendation']['changes']}
        key='sapiensid.identities.'+gate
        assert changes[key]<value
        assert evaluate(patched(c,changes),goals,db)['trials']['baseline']['goal_result']['fully_solved']
    assert c.database.read_bytes()==before


def test_body_goals_and_global_banks_count_same_timestamp_distinct_cues(tmp_path):
    from stash_ai.recognition.body_identity import rebuild
    from stash_ai.recognition.body_global import report
    c,selected,_,_=seeded(tmp_path)
    with FaceDatabase(c.database) as state:
        db=state.connection
        db.execute('UPDATE body_observations SET seconds=0');db.commit()
        rebuild(state,selected,c,{s['scan']:[] for s in selected})
        pairs=report(db,c,'scope',namespace(c))['pairs']
        assert pairs and all(pair['support']==[3,3] for pair in pairs)
        goals=resolve(db,c,[['900']])
        assert len(goals['groups'][0]['members'][0]['identity_evidence']['core_units'])==3


def test_solver_repairs_face_bank_attachment_support(tmp_path):
    from tests.test_recognition_body_identity import configuration,scene
    from stash_ai.recognition.rebuild import rebuild_selected
    from stash_ai.recognition.scanning import scan_identity
    from stash_ai.recognition.merge_solver import solve
    c=configuration(tmp_path)
    with FaceDatabase(c.database) as state:
        db=state.connection
        item,ids=scene(db,c,vectors=[[1,0]]*4)
        for i,oid in enumerate(ids[:3]):
            face=str(uuid.uuid5(uuid.UUID(item['scan']),'face-'+str(i)))
            db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                (face,item['scan'],i,float(i),'','[]','[]',encoded(dict(
                    pixels=100,sharpness=100.,pose_deviation=0.)),None,4,
                 np.asarray([1,0,0,0],dtype='<f4').tobytes(),None))
            db.execute('UPDATE body_observations SET face=? WHERE id=?',(face,oid))
        db.execute('UPDATE scans SET namespace=?,policy_sha=?',(namespace(c),scan_identity(c,namespace(c))))
        db.commit()
        rebuild_selected(state,[item],c)
        row=db.execute('SELECT person,status FROM body_members WHERE observation=?',(ids[-1],)).fetchone()
        assert row['status']=='accepted'
        db.execute('UPDATE people SET performer=? WHERE id=?',('900',row['person']));db.commit()
    c.data['sapiensid']['identities']['attach_min_support']=4
    before=c.database.read_bytes()
    with _open_readonly(c.database) as db:
        goals=resolve(db,c,[['900']],policy='strict')
        # Three face anchors and one faceless sample; linked bodies are not counted twice.
        observations=goals['groups'][0]['members'][0]['observations']
        assert len(observations)==4
        assert sum(r.get('evidence_kind')=='person' for r in observations)==1
        result=solve(c,goals,db,maximum=24)
        assert not result['simulation']['trials']['baseline']['goal_result']['fully_solved']
        assert result['status']=='fully_solved'
        changes={r['parameter']:r['proposed'] for r in result['recommendation']['changes']}
        assert changes['sapiensid.identities.attach_min_support']<=3
        assert evaluate(patched(c,changes),goals,db)['trials']['baseline']['goal_result']['fully_solved']
    assert c.database.read_bytes()==before
