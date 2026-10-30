"""Global identity history, appearance rivals and configuration transitions."""
import json
import math
import uuid
import numpy as np
from stash_ai.recognition.database import FaceDatabase, encoded
from tests.test_recognition_body_identity import configuration, scene
from tests.test_recognition_merge_report import seeded
from tests.test_recognition_hybrid_recovery import enable


def disabled_alias(p):
    from stash_ai.recognition.body_identity import rebuild
    from stash_ai.recognition.body_observations import contract
    c=configuration(p);c.data['sapiensid']['global_matching']['enabled']=True
    with FaceDatabase(c.database) as state:
        db=state.connection;a,_=scene(db,c,'1');b,_=scene(db,c,'2');db.commit()
        rebuild(state,[a,b],c,{a['scan']:[],b['scan']:[]})
        before=[dict(r) for r in db.execute('SELECT source,target FROM identity_merges')]
        c.data['sapiensid']['identities']['enabled']=False;c.data['sapiensid']['global_matching']['enabled']=False
        rebuild(state,[a,b],c,{a['scan']:[],b['scan']:[]})
        return dict(before=before,after=[dict(r) for r in db.execute('SELECT source,target FROM identity_merges')],members=db.execute('SELECT COUNT(*) FROM body_members').fetchone()[0])


def add_bodies_to_faces(db,c,owners):
    from stash_ai.recognition.body_observations import contract
    out=[]
    for index,r in enumerate(db.execute('SELECT o.id,o.scan,o.cue,o.seconds,m.person FROM observations o JOIN members m ON m.observation=o.id').fetchall()):
        if r['person'] not in owners:continue
        v=np.zeros(4096,dtype='<f4');v[0]=1
        bid=str(uuid.uuid4());out.append(bid)
        q=encoded(dict(quality=dict(keypoints=10,sharpness=100,pixel_digest=bid)))
        db.execute('INSERT INTO body_observations VALUES (?,?,?,?,?,?,?,?,?,?,?)',(bid,r['scan'],r['cue'],r['seconds'],contract(c),r['id'],'{}',q,None,v.tobytes(),None))
        db.execute('INSERT OR IGNORE INTO body_cues VALUES (?,?,?)',(r['scan'],contract(c),r['cue']))
    return out


def cross_route_history(p):
    from stash_ai.recognition.body_identity import rebuild
    from stash_ai.recognition.hybrid_evidence import report,catalogue
    c,people=seeded(p,vectors=[[1,0,0,0],[.5,math.sqrt(.75),0,0],[.5,-math.sqrt(.75),0,0]])
    with FaceDatabase(c.database) as state:
        db=state.connection;c=enable(c,db)
        c.data['sapiensid']['identities']['enabled']=True
        c.data['sapiensid']['identities']['min_appearance_distance']=0
        c.data['sapiensid']['collection']['enabled']=True
        c.data['sapiensid']['global_matching']['enabled']=True
        from stash_ai.recognition.clustering import matching_policy
        db.execute('UPDATE candidates SET policy=?',(matching_policy(c),))
        ids=add_bodies_to_faces(db,c,people[:2]);db.commit()
        selected=[dict(scan=r['id'],scene=r['scene']) for r in db.execute("SELECT * FROM scans WHERE scene IN ('1','2')")]
        rebuild(state,selected,c,{r['scan']:[] for r in selected})
        aliases=[json.loads(r[0])['route'] for r in db.execute('SELECT evidence FROM identity_merges')]
        ns=db.execute('SELECT namespace FROM people LIMIT 1').fetchone()[0]
        result=report(db,c,'scope',ns)
        return dict(aliases=aliases,raw_bc=-.5,guards=[len(i.guard_observations) for i in catalogue(db,c,'scope',ns)],outcomes=[(d['outcome'],d['blockers']) for d in result['decisions']])


def global_mode_rival(p):
    from stash_ai.recognition.body_identity import rebuild
    from stash_ai.recognition.body_global import report
    from stash_ai.recognition.body_global_banks import banks
    from tests.test_recognition_assignment_guards import face
    c=configuration(p);c.data['sapiensid']['global_matching']['enabled']=False
    with FaceDatabase(c.database) as state:
        db=state.connection;selected=[]
        for s in ['1','2']:
            item,ids=scene(db,c,s);selected.append(item)
        rival,rids=scene(db,c,'3',vectors=[[1,0]]*3+[[0,1]]*3);selected.append(rival)
        owner=str(uuid.uuid4());db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',(owner,'scope','ns',0,None,'token'))
        for i,bid in enumerate(rids):
            fid=face(db,rival['scan'],i,owner)
            db.execute('UPDATE body_observations SET face=? WHERE id=?',(fid,bid))
        db.commit();rebuild(state,selected,c,{r['scan']:[] for r in selected})
        c.data['sapiensid']['global_matching']['enabled']=True
        from stash_ai.recognition.body_identity import policy
        db.execute('UPDATE body_members SET policy=?',(policy(c),));db.commit()
        result=report(db,c,'scope','ns')
        return dict(mature_rival=owner,rival_support=banks(db,c,'scope','ns')[owner]['support'],rival_mode_score=1,outcomes=[dict(people=d['people'],outcome=d['outcome'],nearest=d['nearest_rival']) for d in result['decisions']])


def stale_body_contract(p):
    from stash_ai.recognition.body_identity import rebuild,policy
    from stash_ai.recognition.body_global import report
    from stash_ai.recognition.body_global_banks import banks
    from stash_ai.recognition.body_observations import contract
    from tests.test_recognition_assignment_guards import face
    c=configuration(p)
    with FaceDatabase(c.database) as state:
        db=state.connection;selected=[]
        for s in ['1','2']:
            item,_=scene(db,c,s);selected.append(item)
        rival,rids=scene(db,c,'3',vectors=[[1,0]]*3+[[.8,.6]]*3);selected.append(rival)
        owner=str(uuid.uuid4());db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',(owner,'scope','ns',0,None,'token'))
        for i,bid in enumerate(rids):
            fid=face(db,rival['scan'],i,owner);db.execute('UPDATE body_observations SET face=? WHERE id=?',(fid,bid))
        db.commit();rebuild(state,selected,c,{r['scan']:[] for r in selected})
        c.data['sapiensid']['global_matching']['enabled']=True
        db.execute('UPDATE body_members SET policy=?',(policy(c),));db.commit()
        before=report(db,c,'scope','ns')
        c.data['sapiensid']['collection']['min_sharpness']+=1
        for item in selected[:2]:
            db.execute('UPDATE body_observations SET contract=? WHERE scan=?',(contract(c),item['scan']))
            db.execute('UPDATE body_cues SET contract=? WHERE scan=?',(contract(c),item['scan']))
            db.execute('UPDATE body_members SET policy=? WHERE observation IN (SELECT id FROM body_observations WHERE scan=?)',(policy(c),item['scan']))
        db.commit();after=report(db,c,'scope','ns')
        return dict(before=[(d['outcome'],d['blockers']) for d in before['decisions']],after=[(d['outcome'],d['blockers']) for d in after['decisions']],old_rival_disappeared=owner not in banks(db,c,'scope','ns'))

def test_body_merge_preserves_original_face_veto_in_hybrid(tmp_path):
    result=cross_route_history(tmp_path)
    assert result['aliases']==['body_global']
    assert sum(result['guards'])==3
    assert all(outcome!='merge' for outcome,_ in result['outcomes'])


def test_global_rival_keeps_each_confirmed_appearance(tmp_path):
    result=global_mode_rival(tmp_path)
    assert result['rival_support']==6
    assert result['outcomes'][0]['outcome']=='blocked'
    assert result['outcomes'][0]['nearest']==1.


def test_disabling_body_identities_retires_fully_selected_aliases(tmp_path):
    result=disabled_alias(tmp_path)
    assert len(result['before'])==1
    assert result['after']==[] and result['members']==0


def test_obsolete_contract_is_unknown_competition_not_absence(tmp_path):
    result=stale_body_contract(tmp_path)
    assert result['before']==[('blocked',['body_ambiguity'])]
    assert result['after']==[('blocked',['incompatible_body_reference_contract'])]


def test_partial_disable_preserves_alias_until_its_scenes_are_selected(tmp_path):
    from stash_ai.recognition.body_identity import rebuild
    from stash_ai.recognition.body_aliases import retire_disabled
    c=configuration(tmp_path);c.data['sapiensid']['global_matching']['enabled']=True
    with FaceDatabase(c.database) as state:
        db=state.connection;a,_=scene(db,c,'1');b,_=scene(db,c,'2');db.commit()
        rebuild(state,[a,b],c,{a['scan']:[],b['scan']:[]})
        c.data['sapiensid']['identities']['enabled']=False;c.data['sapiensid']['global_matching']['enabled']=False
        with db:retire_disabled(state,[a],c)
        assert db.execute('SELECT COUNT(*) FROM identity_merges').fetchone()[0]==1
        with db:retire_disabled(state,[a,b],c)
        assert db.execute('SELECT COUNT(*) FROM identity_merges').fetchone()[0]==0


def test_disable_replays_faces_after_retiring_body_alias_and_reuses_profiles(tmp_path):
    from stash_ai.recognition.rebuild import rebuild_selected
    from stash_ai.recognition.body_identity import rebuild
    from stash_ai.recognition.clustering import matching_policy
    c,people=seeded(tmp_path)
    with FaceDatabase(c.database) as state:
        db=state.connection;selected=[dict(scan=r['id'],scene=r['scene']) for r in db.execute('SELECT * FROM scans')]
        rebuild_selected(state,selected,c)
        original={r[0] for r in db.execute('SELECT DISTINCT person FROM members')}
        c=enable(c,db);c.data['sapiensid']['collection']['enabled']=True
        c.data['sapiensid']['identities'].update(enabled=True,min_appearance_distance=0)
        c.data['sapiensid']['global_matching']['enabled']=True
        for row in db.execute('SELECT id,payload FROM candidates').fetchall():
            oid=json.loads(row['payload'])['observation'];stamp=matching_policy(c)
            db.execute('UPDATE candidates SET id=?,policy=? WHERE id=?',(str(uuid.uuid5(uuid.UUID(oid),stamp)),stamp,row['id']))
        add_bodies_to_faces(db,c,original);db.commit()
        rebuild(state,selected,c,{r['scan']:[] for r in selected})
        assert db.execute('SELECT COUNT(*) FROM identity_merges').fetchone()[0]==1
        for section in (c.data['sapiensid'],c.data['sapiensid']['collection'],c.data['sapiensid']['identities'],c.data['sapiensid']['global_matching']):
            section['enabled']=False
        rebuild_selected(state,selected,c)
        assert db.execute('SELECT COUNT(*) FROM identity_merges').fetchone()[0]==0
        assert {r[0] for r in db.execute('SELECT DISTINCT person FROM members')}==original
        before=[tuple(r) for r in db.execute('SELECT * FROM members ORDER BY observation')]
        rebuild_selected(state,selected,c)
        assert [tuple(r) for r in db.execute('SELECT * FROM members ORDER BY observation')]==before


def test_parallel_global_appearance_rival_matches_scalar(tmp_path,monkeypatch):
    from stash_ai.recognition.compute import compute_scope
    from tests.test_recognition_compute import configured
    monkeypatch.setattr('stash_ai.recognition.compute.available_cpus',lambda:16)
    with compute_scope(configured(rebuild_workers=4)) as pool:
        result=global_mode_rival(tmp_path)
        assert result['outcomes'][0]['outcome']=='blocked'
        assert result['outcomes'][0]['nearest']==1.
        assert pool.submitted>0 and not pool.pending
