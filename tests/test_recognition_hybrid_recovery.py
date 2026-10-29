from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from tests.test_recognition_merge_report import seeded
from stash_ai.recognition.config import plain
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.person_signal import save, contract
from stash_ai.recognition.hybrid_evidence import catalogue, report
from stash_ai.recognition.clustering import matching_policy


ANCHOR_METADATA=dict(quality=dict(keypoints=17,sharpness=100),
                     person=dict(bounds=[0,0,512,512],score=1,keypoints=[[1,1]]*17))


def enable(config, db):
    config=replace(config,data=plain(config.data));config.data['sapiensid']['enabled']=True
    db.execute('UPDATE candidates SET policy=?',(matching_policy(config),))
    for r in db.execute('SELECT id FROM observations').fetchall():save(db,r[0],contract(config),ANCHOR_METADATA,np.ones(4096))
    db.commit();return config


def test_missing_face_body_is_unknown_when_independent_body_bank_suffices(tmp_path):
    config,people=seeded(tmp_path,count=4)
    with FaceDatabase(config.database) as state:
        db=state.connection;config=enable(config,db)
        missing=db.execute('SELECT observation FROM members WHERE person=? LIMIT 1',(people[0],)).fetchone()[0]
        db.execute('DELETE FROM person_evidence WHERE observation=?',(missing,));db.commit()
        ns=db.execute('SELECT namespace FROM people LIMIT 1').fetchone()[0]
        result=report(db,config,'scope',ns)
        assert result['decisions'][0]['outcome']=='merge'
        assert missing in sum(result['pairs'][0]['missing'],[])
        assert min(result['pairs'][0]['support'])==3
        assert missing not in sum(result['pairs'][0]['person_observations'],[])


def test_query_recovery_is_attachment_only_and_idempotent(tmp_path):
    from stash_ai.recognition.hybrid_merge import apply
    from stash_ai.recognition.hybrid_queries import recover
    from stash_ai.recognition.reference_roles import strict_sql
    config,people=seeded(tmp_path,vectors=[[1.,0,0,0],[.7,np.sqrt(.51),0,0],[.5,np.sqrt(.75),0,0]])
    with FaceDatabase(config.database) as state:
        db=state.connection;config=enable(config,db)
        db.execute('DELETE FROM members WHERE person=?',(people[2],))
        db.execute("UPDATE candidates SET person=NULL,status='withheld' WHERE person=?",(people[2],));db.commit()
        selected=[dict(scan=r['id'],scene=r['scene']) for r in db.execute('SELECT * FROM scans')]
        apply(state,selected,config)
        assert recover(state,selected,config)==3
        assert recover(state,selected,config)==0
        assert db.execute("SELECT COUNT(*) FROM candidates WHERE json_extract(payload,'$.reason')='hybrid_query'").fetchone()[0]==3
        assert db.execute('SELECT COUNT(*) FROM members m JOIN observations o ON o.id=m.observation WHERE '+strict_sql()).fetchone()[0]==3


def test_border_tolerance_does_not_resolve_overlapping_people_by_order():
    from stash_ai.recognition.sapiensid_backend import associate
    face=SimpleNamespace(box=(10,10,100,100))
    person=dict(bounds=[12,12,112,112],keypoints=[[50,50],[55,50],[60,50]])
    assert associate([face],[person])==[0]
    assert associate([face],[person,person])==[None]
    far=dict(person,bounds=[20,20,120,120])
    assert associate([face],[far])==[None]


def test_incomplete_hybrid_component_can_repair_stronger_complete_pair(monkeypatch):
    from stash_ai.recognition.hybrid_evidence import Identity, plan
    from stash_ai.recognition.config import defaults
    monkeypatch.setattr('stash_ai.recognition.global_views._cooccurs',lambda *a:False)
    bank=[]
    for name,angle in zip('abc',(0.,.1,.6)):
        body=np.array([[np.cos(angle),np.sin(angle)]],dtype='float32')
        bank.append(Identity(name,(name,),np.array([[1.,0.]]),body,tuple(range(3))))
    result=plan(None,SimpleNamespace(data=defaults()),bank)
    assert [d['people'] for d in result['decisions'] if d['outcome']=='merge']==[['a','b']]


def test_pending_scope_is_revalidated_and_drained_automatically(tmp_path):
    import json
    from tests.test_recognition_hybrid_merge import prepared
    from stash_ai.recognition.hybrid_merge import apply
    config,people=prepared(tmp_path)
    with FaceDatabase(config.database) as state:
        db=state.connection
        selected=[dict(scan=r['id'],scene=r['scene']) for r in db.execute('SELECT * FROM scans')]
        assert apply(state,selected[:1],config)['pending_consolidations']
        saved=json.loads(db.execute("SELECT value FROM metadata WHERE key LIKE 'hybrid-pending:%'").fetchone()[0])
        assert len(saved['pending'][0]['required_scans'])==2
        apply(state,selected,config)
        assert db.execute("SELECT COUNT(*) FROM metadata WHERE key LIKE 'hybrid-pending:%'").fetchone()[0]==0
        assert db.execute('SELECT COUNT(*) FROM identity_merges').fetchone()[0]==1


def test_strict_face_query_does_not_require_body_or_merge_support(tmp_path):
    from stash_ai.recognition.hybrid_queries import recover
    config,people=seeded(tmp_path,vectors=[[1.,0,0,0],[1.,0,0,0]],count=1)
    with FaceDatabase(config.database) as state:
        db=state.connection;config=enable(config,db)
        db.execute('DELETE FROM person_evidence')
        db.execute('DELETE FROM members WHERE person=?',(people[1],))
        db.execute("UPDATE candidates SET person=NULL,status='withheld' WHERE person=?",(people[1],));db.commit()
        selected=[dict(scan=r['id'],scene=r['scene']) for r in db.execute('SELECT * FROM scans')]
        assert recover(state,selected,config)==1
        row=db.execute("SELECT person,json_extract(payload,'$.hybrid_query.signal') FROM candidates WHERE json_extract(payload,'$.reason')='hybrid_query'").fetchone()
        assert tuple(row)==(people[0],'face')
