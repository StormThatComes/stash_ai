import json
from dataclasses import replace

import numpy as np
import pytest

from stash_ai.recognition.body_identity import rebuild
from stash_ai.recognition.database import FaceDatabase,encoded
from stash_ai.recognition.person_anchor import pixels,pixel_gate,PIXEL_PARAMETER
from tests.test_recognition_body_identity import configuration,scene


@pytest.mark.parametrize('metadata,expected',[
    ({'quality':{'width':95,'height':300}},95),
    ({'quality':{'width':300,'height':96}},96),
    ({'person':{'bounds':[10,20,110,220]}},100),
    ({'person':{'bounds':[.5,0,81,240]}},80.5),
    ({'quality':{'width':80,'height':120},'person':{'bounds':[0,0,512,512]}},80),
    ({},None),({'person':None},None),({'quality':{'width':-1,'height':200}},None),
    ({'quality':{'width':True,'height':200}},None),
    ({'quality':{'width':float('nan'),'height':200}},None),
    ({'person':{'bounds':[0,0,float('inf'),200]}},None),
    ({'quality':{'width':80}},None),('invalid json',None),
])
def test_source_measurement_and_missing_dimensions(metadata,expected):
    assert pixels(metadata)==expected
    status=pixel_gate(metadata,dict(anchor_min_person_pixels=96))
    assert status['eligible']==(expected is not None and expected>=96)
    assert pixel_gate(metadata,dict(anchor_min_person_pixels=None))['eligible']


@pytest.mark.parametrize('value',[None,1,96,8192,0,-1,8193,96.0,True,'96'])
def test_configuration_accepts_only_null_or_positive_integer(tmp_path,value):
    from stash_ai.recognition.config import load
    p=tmp_path/'config.yml'
    p.write_text('version: 1\nsapiensid: '+json.dumps({'identities':{'anchor_min_person_pixels':value}}))
    if value is None or type(value) is int and 1<=value<=8192:
        assert load(p).data['sapiensid']['identities']['anchor_min_person_pixels']==value
    else:
        with pytest.raises(ValueError,match='anchor_min_person_pixels'):load(p)


def measured(db,ids,widths):
    for oid,width in zip(ids,widths):
        q=json.loads(db.execute('SELECT quality FROM body_observations WHERE id=?',(oid,)).fetchone()[0])
        q['quality'].update(width=width,height=240)
        db.execute('UPDATE body_observations SET quality=? WHERE id=?',(encoded(q),oid))
    db.commit()


def test_small_bodies_attach_but_cannot_create_or_support_identity(tmp_path):
    c=configuration(tmp_path);c.data['sapiensid']['identities']['anchor_min_person_pixels']=96
    with FaceDatabase(c.database) as state:
        db=state.connection;item,ids=scene(db,c,vectors=[[1,0]]*4)
        measured(db,ids,[96,100,120,95])
        rebuild(state,[item],c,{item['scan']:[]})
        rows={r['observation']:dict(r) for r in db.execute('SELECT * FROM body_members')}
        assert [rows[i]['role'] for i in ids]==['anchor','anchor','anchor','auxiliary']
        assert all(r['status']=='accepted' for r in rows.values())
        assert json.loads(rows[ids[-1]]['proof'])['anchor_pixel_gate']['reason']=='body_anchor_pixels'
        measured(db,ids,[96,100,95,95])
        rebuild(state,[item],c,{item['scan']:[]})
        assert not db.execute("SELECT 1 FROM body_members WHERE status='accepted'").fetchone()
        measured(db,ids,[95]*4)
        rebuild(state,[item],c,{item['scan']:[]})
        assert all(json.loads(r[0])['reason']=='body_anchor_pixels' for r in db.execute('SELECT proof FROM body_members'))


def test_global_banks_count_only_large_anchors_and_report_size(tmp_path):
    from stash_ai.recognition.body_global import banks
    from stash_ai.recognition.body_reports import export_members
    c=configuration(tmp_path);c.data['sapiensid']['identities']['anchor_min_person_pixels']=96
    with FaceDatabase(c.database) as state:
        db=state.connection;item,ids=scene(db,c,vectors=[[1,0]]*4)
        measured(db,ids,[96,100,120,95]);rebuild(state,[item],c,{item['scan']:[]})
        bank=next(iter(banks(db,c,'scope','ns').values()))
        assert bank['support']==3 and len(bank['refs'])==3
        person=db.execute('SELECT * FROM people LIMIT 1').fetchone()
        samples=export_members(db,c,person,tmp_path)
        small=next(r for r in samples if r['observation_id']==ids[-1])
        assert small['anchor_pixel_gate']==dict(pixels=95,minimum=96,eligible=False,reason='body_anchor_pixels')


def test_small_linked_bodies_do_not_supply_face_bank_support(tmp_path):
    import uuid
    from stash_ai.recognition.rebuild import rebuild_selected
    from stash_ai.recognition.scanning import scan_identity
    from stash_ai.recognition.models import namespace
    c=configuration(tmp_path);c.data['sapiensid']['identities']['anchor_min_person_pixels']=96
    with FaceDatabase(c.database) as state:
        db=state.connection;item,ids=scene(db,c,vectors=[[1,0]]*4)
        measured(db,ids,[96,95,95,64])
        for i,oid in enumerate(ids[:3]):
            face=str(uuid.uuid5(uuid.UUID(item['scan']),'face-'+str(i)))
            db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                (face,item['scan'],i,float(i),'','[]','[]',encoded(dict(
                    pixels=100,sharpness=100.,pose_deviation=0.)),None,4,
                 np.asarray([1,0,0,0],dtype='<f4').tobytes(),None))
            db.execute('UPDATE body_observations SET face=? WHERE id=?',(face,oid))
        db.execute('UPDATE scans SET namespace=?,policy_sha=?',(namespace(c),scan_identity(c,namespace(c))))
        db.commit();rebuild_selected(state,[item],c)
        faces=list(db.execute('SELECT observation,person FROM members'))
        assert len(faces)==3 and len({r['person'] for r in faces})==1
        assert db.execute('SELECT status FROM body_members WHERE observation=?',(ids[-1],)).fetchone()[0]=='withheld'
        assert db.execute("SELECT COUNT(*) FROM body_members WHERE role='anchor'").fetchone()[0]==1
        c.data['sapiensid']['identities']['anchor_min_person_pixels']=95
        rebuild_selected(state,[item],c)
        query=db.execute('SELECT person,status,role FROM body_members WHERE observation=?',(ids[-1],)).fetchone()
        assert tuple(query)==(faces[0]['person'],'accepted','auxiliary')
        assert [tuple(r) for r in db.execute('SELECT observation,person FROM members')]==[tuple(r) for r in faces]


def test_size_gate_changes_matching_policy_but_not_model_or_scan_contracts(tmp_path):
    from stash_ai.recognition.clustering import matching_policy
    from stash_ai.recognition.person_signal import contract,same_contract
    from stash_ai.recognition.body_observations import contract as body_contract
    from stash_ai.recognition.scanning import scan_identity
    from stash_ai.recognition.config import plain
    c=configuration(tmp_path)
    for enabled in (False,True):
        c.data['sapiensid']['identities']['enabled']=enabled
        changed=replace(c,data=plain(c.data));changed.data['sapiensid']['identities']['anchor_min_person_pixels']=96
        assert matching_policy(c)!=matching_policy(changed)
        assert contract(c)==contract(changed) and body_contract(c)==body_contract(changed)
        assert scan_identity(c,'ns')==scan_identity(changed,'ns') and same_contract(c,changed)


def test_face_anchor_body_bank_obeys_size_gate_without_demoting_faces(tmp_path):
    from tests.test_recognition_hybrid_merge import prepared
    from stash_ai.recognition.hybrid_references import body_rows
    from stash_ai.recognition.hybrid_evidence import catalogue
    c,people=prepared(tmp_path);c.data['sapiensid']['identities']['anchor_min_person_pixels']=96
    assert not c.data['sapiensid']['identities']['enabled']
    with FaceDatabase(c.database) as state:
        db=state.connection;ns=db.execute('SELECT namespace FROM people LIMIT 1').fetchone()[0]
        ids=[r[0] for r in db.execute('SELECT observation FROM members WHERE person=? ORDER BY observation',(people[0],))]
        for oid,width in zip(ids,[96,95,None]):
            metadata={'person':{'bounds':[0,0,width,240]}} if width is not None else {}
            db.execute('UPDATE person_evidence SET metadata=? WHERE observation=?',(encoded(metadata),oid))
        db.commit()
        rows=body_rows(db,c,'scope',ns,people[0])
        assert [r['id'] for r in rows]==ids[:1] and rows.support==1
        identity=next(i for i in catalogue(db,c,'scope',ns) if i.person==people[0])
        assert len(identity.faces)==3 and len(identity.persons)==1
        c.data['sapiensid']['identities']['anchor_min_person_pixels']=None
        assert len(body_rows(db,c,'scope',ns,people[0]))==3


def test_original_small_witnesses_keep_veto_power_but_not_query_authority(tmp_path):
    from tests.test_recognition_hybrid_merge import prepared
    from stash_ai.recognition.hybrid_merge import apply
    from stash_ai.recognition.hybrid_evidence import catalogue
    c,people=prepared(tmp_path)
    with FaceDatabase(c.database) as state:
        db=state.connection
        for owner,width in zip(people,[120,80]):
            db.execute('UPDATE person_evidence SET metadata=? WHERE observation IN (SELECT observation FROM members WHERE person=?)',
                (encoded(dict(person=dict(bounds=[0,0,width,240]))),owner))
        db.commit()
        selected=[dict(scan=r['id'],scene=r['scene']) for r in db.execute('SELECT * FROM scans')]
        apply(state,selected,c)
        c.data['sapiensid']['identities']['anchor_min_person_pixels']=96
        ns=db.execute('SELECT namespace FROM people LIMIT 1').fetchone()[0]
        identity=catalogue(db,c,'scope',ns)[0]
        assert len(identity.persons)==len(identity.independent)==3
        assert len(identity.guard_persons)==3 and not identity.guard_missing
        assert sorted(len(b) for _,_,b in identity.query_cores)==[0,3]


def test_tune_can_enable_and_disable_gate_without_rescanning(tmp_path,monkeypatch):
    from tests.test_recognition_body_integration import seeded
    from stash_ai.recognition.threshold_tuning import tune
    c,_,ids,_=seeded(tmp_path,vectors=[[1,0]])
    with FaceDatabase(c.database) as state:measured(state.connection,ids[0],[95]*3)
    monkeypatch.setattr('stash_ai.recognition.models.load_backend',lambda *a,**k:pytest.fail('inference'))
    before=c.database.read_bytes()
    result=tune(c,{PIXEL_PARAMETER:[96,None,95]},scope='scope')
    assert result['trials']['baseline']['versus_retained']['assigned']==3
    assert result['trials']['trial-1']['versus_retained']['assigned']==0
    assert result['trials']['trial-2']['versus_retained']['assigned']==3
    c.data['sapiensid']['identities']['anchor_min_person_pixels']=96
    result=tune(c,{PIXEL_PARAMETER:[None]},scope='scope')
    assert result['trials']['trial-1']['versus_retained']['assigned']==3
    assert c.database.read_bytes()==before


def test_solver_repairs_numeric_size_gate_using_retained_measurements(tmp_path,monkeypatch):
    from tests.test_recognition_body_integration import seeded
    from stash_ai.recognition.fragmentation_report import _open_readonly
    from stash_ai.recognition.merge_goals import resolve
    from stash_ai.recognition.merge_solver import solve
    from stash_ai.recognition.body_solver import analyse
    c,_,ids,_=seeded(tmp_path,vectors=[[1,0]])
    with FaceDatabase(c.database) as state:measured(state.connection,ids[0],[95]*3)
    c.data['sapiensid']['identities']['anchor_min_person_pixels']=96
    monkeypatch.setattr('stash_ai.recognition.models.load_backend',lambda *a,**k:pytest.fail('inference'))
    before=c.database.read_bytes()
    with _open_readonly(c.database) as db:
        goals=resolve(db,c,[['900']],policy='strict')
        hints=analyse(db,c,goals)
        assert {95,96}<=set(hints['options'][PIXEL_PARAMETER])
        result=solve(c,goals,db,maximum=8)
        assert result['status']=='fully_solved'
        patch={r['parameter']:r['proposed'] for r in result['recommendation']['changes']}
        assert 1<=patch[PIXEL_PARAMETER]<=95 and type(patch[PIXEL_PARAMETER]) is int
    assert c.database.read_bytes()==before


@pytest.mark.parametrize('feasibility_trials',[0,6],ids=['measured_body_gate','initial_face_probe'])
def test_solver_uses_legacy_person_bounds_when_independent_identities_are_disabled(tmp_path,monkeypatch,feasibility_trials):
    from tests.test_recognition_merge_solver_hybrid import prepared
    from stash_ai.recognition.fragmentation_report import _open_readonly
    from stash_ai.recognition.merge_goals import resolve,evaluate
    from stash_ai.recognition.merge_report_policy import patched
    from stash_ai.recognition.merge_solver import solve
    from stash_ai.recognition.body_solver import analyse
    from stash_ai.recognition.merge_solver_domains import registry
    c,people=prepared(tmp_path,[[1.,0,0,0],[.7,np.sqrt(.51),0,0]],[[1.,0],[1.,0]],
        matching=dict(face_floor=.4,person_threshold=.85,min_support=3))
    assert not c.data['sapiensid']['identities']['enabled']
    assert PIXEL_PARAMETER not in registry(c)
    c.data['sapiensid']['identities']['anchor_min_person_pixels']=96
    with FaceDatabase(c.database) as state:
        for owner,width in zip(people,[120,80.5]):
            state.connection.execute('UPDATE person_evidence SET metadata=? WHERE observation IN (SELECT observation FROM members WHERE person=?)',
                (encoded(dict(person=dict(bounds=[0,0,width,240]))),owner))
        state.connection.commit()
    monkeypatch.setattr('stash_ai.recognition.models.load_backend',lambda *a,**k:pytest.fail('inference'))
    before=c.database.read_bytes()
    with _open_readonly(c.database) as db:
        goals=resolve(db,c,[people],by_performer=False)
        hints=analyse(db,c,goals)
        assert hints['options'].keys()=={PIXEL_PARAMETER}
        assert {80,81,120,121}<=set(hints['options'][PIXEL_PARAMETER])
        result=solve(c,goals,db,maximum=4,targeted_trials=0,feasibility_trials=feasibility_trials)
        assert not result['simulation']['trials']['baseline']['goal_result']['fully_solved']
        assert result['status']=='fully_solved'
        patch={r['parameter']:r['proposed'] for r in result['recommendation']['changes']}
        if feasibility_trials==0:
            # Isolate measured legacy body admission from the initial face fallback.
            assert patch=={PIXEL_PARAMETER:80}
        else:
            trial=result['simulation']['trials']['trial-1']
            assert trial['search_phase']=='feasibility'
            assert trial['goal_result']['fully_solved']
            assert result['feasibility']['joint_witness_trial']=='trial-1'
            assert PIXEL_PARAMETER not in patch
            assert patch['recognition.match_threshold']<c.data['recognition']['match_threshold']
        winner=patched(c,patch)
        assert not winner.data['sapiensid']['identities']['enabled']
        assert evaluate(winner,goals,db)['trials']['baseline']['goal_result']['fully_solved']
    assert c.database.read_bytes()==before
