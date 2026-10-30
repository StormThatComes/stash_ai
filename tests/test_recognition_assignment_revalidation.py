"""Regression coverage for replayed aliases, complete body vetoes and rivals."""
import json
import math
import uuid
import numpy as np
import pytest
from stash_ai.recognition.database import FaceDatabase
from tests.test_recognition_global_identity_lifecycle import seeded, enable, add_bodies_to_faces
from tests.test_recognition_body_identity import configuration, scene
from tests.test_recognition_assignment_guards import face


def alias_retune(p,hot=True):
 p.mkdir(parents=True,exist_ok=True)
 from stash_ai.recognition.rebuild import rebuild_selected
 from stash_ai.recognition.body_identity import rebuild
 from stash_ai.recognition.clustering import matching_policy
 c,_=seeded(p)
 with FaceDatabase(c.database) as state:
  db=state.connection; selected=[dict(scan=r['id'],scene=r['scene']) for r in db.execute('SELECT * FROM scans')]
  rebuild_selected(state,selected,c)
  original={r[0] for r in db.execute('SELECT DISTINCT person FROM members')}
  c=enable(c,db);c.data['sapiensid']['collection']['enabled']=True
  c.data['sapiensid']['identities'].update(enabled=True,min_appearance_distance=0)
  c.data['sapiensid']['global_matching'].update(enabled=True,match_threshold=.9,cohesion_threshold=.9)
  c.data['sapiensid']['matching']['person_threshold']=.99
  for row in db.execute('SELECT id,payload FROM candidates').fetchall():
   oid=json.loads(row['payload'])['observation'];stamp=matching_policy(c)
   db.execute('UPDATE candidates SET id=?,policy=? WHERE id=?',(str(uuid.uuid5(uuid.UUID(oid),stamp)),stamp,row['id']))
  add_bodies_to_faces(db,c,original)
  owner=sorted(original)[1]
  ids=[r[0] for r in db.execute('SELECT observation FROM members WHERE person=?',(owner,))]
  v=np.zeros(4096,dtype='<f4');v[:2]=[.92,math.sqrt(1-.92**2)]
  v0=np.zeros(4096,dtype='<f4');v0[0]=1
  db.execute('UPDATE person_evidence SET embedding=?',(v0.tobytes(),))
  for oid in ids:
   db.execute('UPDATE body_observations SET embedding=? WHERE face=?',(v.tobytes(),oid))
   db.execute('UPDATE person_evidence SET embedding=? WHERE observation=?',(v.tobytes(),oid))
  db.commit()
  if hot:rebuild(state,selected,c,{r['scan']:[] for r in selected})
  before=dict(aliases=db.execute('SELECT COUNT(*) FROM identity_merges').fetchone()[0],owners=db.execute('SELECT COUNT(DISTINCT person) FROM members').fetchone()[0])
  c.data['sapiensid']['global_matching']['match_threshold']=.99
  rebuild_selected(state,selected,c)
  after=dict(aliases=db.execute('SELECT COUNT(*) FROM identity_merges').fetchone()[0],owners=db.execute('SELECT COUNT(DISTINCT person) FROM members').fetchone()[0],reasons=[tuple(r) for r in db.execute("SELECT json_extract(payload,'$.reason'),count(*) FROM candidates GROUP BY 1")])
  return dict(original=len(original),before=before,after=after)


def hard_rival(p,rival_score=1.,rival_face=-1):
 from stash_ai.recognition.body_identity import rebuild,policy
 from stash_ai.recognition.body_global import report
 c=configuration(p)
 with FaceDatabase(c.database) as state:
  db=state.connection;selected=[]; owners=[]
  for n,(v,f) in enumerate([([1,0],1),([.95,math.sqrt(1-.95**2)],1),([rival_score,-math.sqrt(1-rival_score**2)],rival_face)]):
   item,ids=scene(db,c,str(n+1),vectors=[v]*3);selected.append(item)
   owner=str(uuid.uuid4());owners.append(owner)
   db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',(owner,'scope','ns',0,None,owner))
   for i,bid in enumerate(ids):
    fid=face(db,item['scan'],i,owner,value=f);db.execute('UPDATE body_observations SET face=? WHERE id=?',(fid,bid))
  db.commit();rebuild(state,selected,c,{r['scan']:[] for r in selected})
  c.data['sapiensid']['global_matching'].update(enabled=True,match_threshold=.9,cohesion_threshold=.9,ambiguity_margin=.1)
  db.execute('UPDATE body_members SET policy=?',(policy(c),));db.commit()
  r=report(db,c,'scope','ns')
  names=dict(zip(owners,'ABC'))
  return dict(pairs=[dict(people=[names[x] for x in v['people']],score=v['score'],blockers=v['blockers']) for v in r['pairs']],decisions=r['decisions'])


def pixel_veto(p):
 from stash_ai.recognition.body_identity import scene_plan
 c=configuration(p);c.data['sapiensid']['identities']['min_support']=2
 with FaceDatabase(c.database) as state:
  db=state.connection;item,ids=scene(db,c,vectors=[[math.cos(a),math.sin(a)] for a in [0,.5,-.5]])
  for oid in ids[:2]:db.execute("UPDATE body_observations SET quality=json_set(quality,'$.quality.pixel_digest','shared') WHERE id=?",(oid,))
  _,_,out=scene_plan(db,item['scan'],c)
  return dict(actual_minimum=math.cos(1),required=c.data['sapiensid']['identities']['cohesion_threshold'],decisions=[out[i] for i in ids])


def stale_face_veto(p):
 from stash_ai.recognition.body_guards import FaceGuard
 from stash_ai.recognition.scan_heads import predicate
 c=configuration(p)
 with FaceDatabase(c.database) as state:
  db=state.connection;old,_=scene(db,c,'1');new,_=scene(db,c,'2')
  owner=str(uuid.uuid4());db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',(owner,'scope','ns',0,None,owner))
  prior=face(db,old['scan'],0,owner,-1);face(db,new['scan'],0,owner,1)
  db.execute("UPDATE scans SET scene='1',input_sha='new-input' WHERE id=?",(new['scan'],))
  query={'_face_vector':np.array([1,0],dtype='<f4')}
  before=FaceGuard(db,c).allows(query,[],owner)
  eligible=[r[0] for r in db.execute('SELECT o.id FROM observations o JOIN scans s ON s.id=o.scan WHERE '+predicate())]
  db.execute('DELETE FROM members WHERE observation=?',(prior,))
  return dict(old_is_current=prior in eligible,before=before,after=FaceGuard(db,c).allows(query,[],owner))


def test_retuning_enabled_body_aliases_restores_original_faces(tmp_path):
    hot=alias_retune(tmp_path/'hot')
    cold=alias_retune(tmp_path/'cold',hot=False)
    assert hot['before']==dict(aliases=1,owners=1)
    assert hot['after']['aliases']==cold['after']['aliases']==0
    assert hot['after']['owners']==cold['after']['owners']==2
    assert all(reason!='body_global_identity' for reason,_ in hot['after']['reasons'])


def test_pixel_deduplication_cannot_hide_a_different_pose_vector(tmp_path):
    result=pixel_veto(tmp_path)
    assert result['actual_minimum']<result['required']
    assert not any(r['status']=='accepted' for r in result['decisions'])
    assert result['decisions'][1]['person']!=result['decisions'][2]['person']


@pytest.mark.parametrize('score',[1.,.89])
def test_global_hard_face_veto_is_not_an_ambiguity_rival(tmp_path,score):
    result=hard_rival(tmp_path,rival_score=score)
    assert len(result['decisions'])==1
    assert result['decisions'][0]['outcome']=='merge'
    assert result['decisions'][0]['nearest_rival']==-1.


def test_global_compatible_weak_rival_still_blocks(tmp_path):
    result=hard_rival(tmp_path,rival_score=.89,rival_face=1)
    assert result['decisions'][0]['outcome']=='blocked'
    assert result['decisions'][0]['blockers']==['body_ambiguity']


def test_superseded_face_members_do_not_veto_body_queries(tmp_path):
    result=stale_face_veto(tmp_path)
    assert not result['old_is_current']
    assert result['before'] and result['after']


def test_partial_selection_keeps_body_alias_history(tmp_path):
    from stash_ai.recognition.body_identity import rebuild
    from stash_ai.recognition.body_aliases import prepare_replay
    c=configuration(tmp_path);c.data['sapiensid']['global_matching']['enabled']=True
    with FaceDatabase(c.database) as state:
        db=state.connection;a,_=scene(db,c,'1');b,_=scene(db,c,'2');db.commit()
        rebuild(state,[a,b],c,{a['scan']:[],b['scan']:[]})
        before=[tuple(r) for r in db.execute('SELECT * FROM identity_merges')]
        c.data['sapiensid']['global_matching']['match_threshold']=.99
        with db:prepare_replay(state,[a],c)
        assert before and [tuple(r) for r in db.execute('SELECT * FROM identity_merges')]==before
