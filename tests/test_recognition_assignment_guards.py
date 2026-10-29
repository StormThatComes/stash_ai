import uuid, math
from pathlib import Path

import numpy as np
from stash_ai.recognition.database import FaceDatabase, encoded
from tests.test_recognition_body_identity import configuration, scene
from stash_ai.recognition.body_identity import scene_plan, rebuild

def face(db, scan, cue, who=None, value=1):
    oid=str(uuid.uuid4())
    feature=np.array([value,0],dtype='<f4')
    db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',(oid,scan,cue,float(cue),'','[]','[]',encoded(dict(pose_deviation=.1,pixels=100,sharpness=100)),None,2,feature.tobytes(),None))
    if who: db.execute('INSERT INTO members VALUES (?,?)',(oid,who))
    return oid



def test_body_aux_claims(tmp_path):
    t=tmp_path

    c=configuration(Path(t))
    with FaceDatabase(c.database) as st:
        db=st.connection
        item,ids=scene(db,c,vectors=[[1,0]]*5,times=[0,1,2,3,3])
        for oid in ids[-2:]: db.execute("UPDATE body_observations SET quality=json_set(quality,'$.quality.keypoints',4) WHERE id=?",(oid,))
        _,_,d=scene_plan(db,item['scan'],c)
        assert all(d[i]['status']=='withheld' for i in ids[-2:])


def test_face_veto(tmp_path):
    t=tmp_path

    c=configuration(Path(t))
    with FaceDatabase(c.database) as st:
        db=st.connection
        item,ids=scene(db,c,vectors=[[1,0]]*4)
        who=str(uuid.uuid4());db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',(who,'scope','ns',0,None,'token'))
        for n,oid in enumerate(ids):
            f=face(db,item['scan'],n,who if n<3 else None,1 if n<3 else -1)
            db.execute('UPDATE body_observations SET face=? WHERE id=?',(f,oid))
        _,_,d=scene_plan(db,item['scan'],c)
        assert d[ids[-1]]['status']=='withheld'


def test_global_face_presence(tmp_path):
    t=tmp_path

    c=configuration(Path(t))
    from stash_ai.recognition.body_identity import policy as body_policy
    from stash_ai.recognition.body_global import report
    with FaceDatabase(c.database) as st:
        db=st.connection;a,ai=scene(db,c,'1');b,bi=scene(db,c,'2');db.commit()
        result=rebuild(st,[a,b],c,{a['scan']:[],b['scan']:[]})
        pa,pb=result[a['scan']][0]['person'],result[b['scan']][0]['person']
        fa=face(db,b['scan'],0,pa);fb=face(db,b['scan'],0,pb)
        db.execute('UPDATE observations SET embedding=? WHERE id=?',(np.array([.7,np.sqrt(.51)],dtype='<f4').tobytes(),fb))
        c.data['sapiensid']['global_matching']['enabled']=True
        db.execute('UPDATE body_members SET policy=?',(body_policy(c),));db.commit()
        r=report(db,c,'scope','ns')
        assert 'person_copresence' in r['pairs'][0]['blockers']


def test_hybrid_aux_disabled(tmp_path):
    t=tmp_path

    from tests.test_recognition_merge_report import seeded
    from tests.test_recognition_hybrid_recovery import enable
    from stash_ai.recognition.hybrid_queries import recover
    from stash_ai.recognition.clustering import matching_policy
    c,people=seeded(Path(t),vectors=[[1.,0,0,0],[1.,0,0,0]],count=3,clustering=dict(auxiliary_matching=False))
    with FaceDatabase(c.database) as st:
        db=st.connection;c=enable(c,db)
        db.execute('DELETE FROM members WHERE person=?',(people[1],))
        db.execute("UPDATE observations SET quality=json_set(quality,'$.pose_deviation',.8) WHERE scan IN (SELECT scan FROM candidates WHERE person=?)",(people[1],))
        db.execute("UPDATE candidates SET person=NULL,status='withheld',payload=json_set(payload,'$.reason','auxiliary_disabled') WHERE person=?",(people[1],));db.commit()
        selected=[dict(scan=r['id'],scene=r['scene']) for r in db.execute('SELECT * FROM scans')]
        assert recover(st,selected,c)==0


def test_hybrid_simultaneous_claims(tmp_path):
    t=tmp_path

    from tests.test_recognition_merge_report import seeded
    from tests.test_recognition_hybrid_recovery import enable
    from stash_ai.recognition.hybrid_merge import apply
    from stash_ai.recognition.hybrid_queries import recover
    from stash_ai.recognition.clustering import matching_policy
    from stash_ai.recognition.scanning import scan_identity
    c,people=seeded(Path(t),vectors=[[1.,0,0,0],[.5,np.sqrt(.75),0,0]])
    with FaceDatabase(c.database) as st:
        db=st.connection;c=enable(c,db)
        selected=[dict(scan=r['id'],scene=r['scene']) for r in db.execute('SELECT * FROM scans')]
        merged=apply(st,selected,c)
        ns=db.execute('SELECT namespace FROM people LIMIT 1').fetchone()[0];sid=str(uuid.uuid4())
        db.execute('INSERT INTO scans VALUES (?,?,?,?,?,?,?,?,?,?,?)',(sid,'scope','3',ns,'input',scan_identity(c,ns),'{}','{}',1,1,'complete'))
        for n,angle in enumerate((-10,70)):
            oid=str(uuid.uuid4());a=math.radians(angle);v=np.array([math.cos(a),math.sin(a),0,0],dtype='<f4')
            db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',(oid,sid,0,0.,'','[]',encoded([n*100,0,90,90]),encoded(dict(pose_deviation=.1,pixels=100,sharpness=100)),None,4,v.tobytes(),None))
            db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?)',(str(uuid.uuid4()),sid,matching_policy(c),None,'withheld',encoded(dict(observation=oid,reason='ambiguous_identity'))))
        db.commit();selected.append(dict(scan=sid,scene='3'))
        assert recover(st,selected,c)==0


def test_missing_head_keypoints_cannot_associate_at_image_border():
    from types import SimpleNamespace
    from stash_ai.recognition.sapiensid_backend import associate
    assert associate([SimpleNamespace(box=(-1,-1,100,100))],
        [dict(bounds=[0,0,150,200],keypoints=[[-1,-1]]*17)])==[None]


def test_unassigned_faces_keep_global_body_contradiction_veto(tmp_path):
    c=configuration(tmp_path);c.data['sapiensid']['global_matching']['enabled']=True
    with FaceDatabase(c.database) as state:
        db=state.connection;selected=[]
        for name,direction in [('1',1),('2',-1)]:
            item,ids=scene(db,c,name);selected.append(item)
            for cue,oid in enumerate(ids):
                measured=face(db,item['scan'],cue,value=direction)
                db.execute('UPDATE body_observations SET face=? WHERE id=?',(measured,oid))
        db.commit();rebuild(state,selected,c,{i['scan']:[] for i in selected})
        assert db.execute('SELECT COUNT(*) FROM identity_merges').fetchone()[0]==0
        assert len({r[0] for r in db.execute('SELECT person FROM body_members')})==2
