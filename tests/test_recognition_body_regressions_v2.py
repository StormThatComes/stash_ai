import uuid,tempfile,json,time
from pathlib import Path
import numpy as np
from stash_ai.recognition.database import FaceDatabase,encoded
from stash_ai.recognition.hybrid_merge import signature,rebuild
from stash_ai.recognition.hybrid_evidence import report,catalogue
from stash_ai.recognition.clustering import matching_policy
from tests.test_recognition_hybrid_merge import prepared


def test_inherited_body_face_veto_invalidates_warm_stamp(tmp_path):
 c,people=prepared(tmp_path)
 with FaceDatabase(c.database) as state:
  db=state.connection
  selected=[dict(scan=r['id'],scene=r['scene']) for r in db.execute('select * from scans order by scene')]
  scan=selected[0]['scan']
  ns=db.execute('select namespace from scans where id=?',(scan,)).fetchone()[0]
  fid=str(uuid.uuid4())
  db.execute('insert into observations values (?,?,?,?,?,?,?,?,?,?,?,?)',(fid,scan,0,0.,'url','[]','[]','{}',None,4,np.array([-1.,0,0,0],dtype='<f4').tobytes(),None))
  db.execute('insert into members values (?,?)',(fid,people[0]))
  db.execute('insert into candidates values (?,?,?,?,?,?)',(str(uuid.uuid4()),scan,matching_policy(c),people[0],'accepted',encoded(dict(observation=fid,reason='auxiliary_identity'))))
  bid=str(uuid.uuid4());old=str(uuid.uuid4())
  db.execute('insert into people values (?,?,?,?,?,?)',(old,'scope',ns,3,None,'oldtoken'))
  db.execute('insert into body_observations values (?,?,?,?,?,?,?,?,?,?,?)',(bid,scan,0,0.,'old',fid,'{}','{}',None,bytes(16384),None))
  db.execute('insert into identity_merges values (?,?,?,?,?)',(old,people[0],time.time(),encoded(dict(route='body_global',cores={old:[bid]},scenes=['1'])),'local'))
  db.commit()
  assert not any(d['outcome']=='merge' for d in report(db,c,'scope',ns)['decisions'])
  calls=[]
  rebuild(state,selected,c,lambda:calls.append(1),lambda:None)
  a=signature(db,selected,c)
  db.execute('update body_observations set face=null where id=?',(bid,));db.commit()
  assert a!=signature(db,selected,c)
  assert any(d['outcome']=='merge' for d in report(db,c,'scope',ns)['decisions'])
  rebuild(state,selected,c,lambda:calls.append(1),lambda:None)
  assert len(calls)==2 and state.hybrid_rebuild_mode!='warm'
  assert db.execute("select count(*) from identity_merges where json_extract(evidence,'$.route')='sapiensid_hybrid'").fetchone()[0]==1


def test_global_merge_keeps_archived_unselected_scene(tmp_path):
    from tests.test_recognition_body_global import configured,scene
    from stash_ai.recognition.body_identity import rebuild,policy
    from stash_ai.recognition.rebuild import rebuild_selected
    config=configured(tmp_path)
    with FaceDatabase(config.database) as state:
        db=state.connection
        a,ai=scene(db,config,'1');b,bi=scene(db,config,'2');db.commit()
        rebuild(state,[a],config,{a['scan']:[]});rebuild(state,[b],config,{b['scan']:[]})
        source=db.execute('SELECT person FROM body_members WHERE observation=?',(bi[0],)).fetchone()[0]
        old,ids=scene(db,config,'3')
        db.executemany('INSERT INTO body_members VALUES (?,?,?,?,?,?)',[(i,source,'auxiliary','accepted',policy(config),'{}') for i in ids])
        db.execute("INSERT INTO scans SELECT ?,scope,scene,namespace,'new',policy_sha,manifest,coverage,0,0,'complete' FROM scans WHERE id=?",(str(uuid.uuid4()),old['scan']))
        db.commit()
        before=[tuple(r) for r in db.execute('SELECT * FROM body_members WHERE observation IN (SELECT id FROM body_observations WHERE scan=?) ORDER BY observation',(old['scan'],))]
        rebuild_selected(state,[a,b],config)
        after=[tuple(r) for r in db.execute('SELECT * FROM body_members WHERE observation IN (SELECT id FROM body_observations WHERE scan=?) ORDER BY observation',(old['scan'],))]
        assert after==before


def test_body_mode_mean_uses_independent_cues_but_minimum_keeps_all_witnesses():
    import pytest
    from types import SimpleNamespace
    from stash_ai.recognition.config import defaults
    from stash_ai.recognition.disk_body_modes import Records
    from stash_ai.recognition.hybrid_evidence import scores
    config=SimpleNamespace(data=defaults());config.data['sapiensid']['matching'].update(min_support=2,person_threshold=.1)
    spool=Records()
    try:
        for i,(cue,cosine) in enumerate(((0,1),(0,.6),(1,1))):
            v=np.zeros(4096,dtype='<f4');v[0]=cosine;v[1]=np.sqrt(1-cosine*cosine)
            spool.add(dict(id=str(i),evidence_kind='person',scan='one',cue=cue,seconds=float(cue),embedding=v.tobytes(),person_metadata=json.dumps({'quality':{'pixel_digest':str(i)}}),quality='{}'))
        modes,support=spool.modes(config)
        q=np.zeros((1,4096),dtype='<f4');q[0,0]=1
        assert support==2
        assert modes[0].weights(slice(None)).sum()==2
        assert scores(q,modes[0])==pytest.approx(dict(minimum=.6,mean=1))
    finally:spool.close()
