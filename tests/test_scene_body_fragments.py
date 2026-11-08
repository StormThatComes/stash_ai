import json
import numpy as np

from stash_ai.recognition.body_identity import scene_plan, defaults
from stash_ai.recognition.body_perspectives import record_scene_roster
from stash_ai.recognition.database import FaceDatabase, encoded
from tests.test_recognition_body_identity import configuration, scene


def test_consistent_near_misses_remain_one_anonymous_fragment(tmp_path):
    cfg = configuration(tmp_path)
    with FaceDatabase(cfg.database) as state:
        db = state.connection
        item,ids = scene(db,cfg,vectors=[[1.,0.]]+[[.8,.6]]*3)
        db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',('A','scope','ns',1,None,'face-A'))
        quality = encoded(dict(pixels=100,sharpness=100,pose_deviation=0.,visible_face_fraction=1.))
        db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                   ('face-A',item['scan'],0,0.,'','[]','[]',quality,None,2,np.array([1.,0.],dtype='<f4').tobytes(),None))
        db.execute('INSERT INTO members VALUES (?,?)',('face-A','A'))
        db.execute('UPDATE body_observations SET face=? WHERE id=?',('face-A',ids[0]))
        decisions = scene_plan(db,item['scan'],cfg)[2]
        owners = {decisions[oid]['person'] for oid in ids[1:]}
        assert len(owners)==1 and not owners.intersection({None,'A'})
        assert all(decisions[oid]['status']=='provisional' for oid in ids[1:])


def test_roster_uses_body_hash_independence():
    rows = {i:[dict(id=str(i),scan='s',cue=i,seconds=i,
                    quality=json.dumps({'quality':dict(pixel_digest=str(i),appearance_hash=format(1<<i,'016x'))}))]
            for i in range(4)}
    decisions = {str(i):{} for i in rows}
    record_scene_roster(rows,dict.fromkeys(['0','1','2'],'A'),{i:i for i in rows},set(),decisions,3,settings=defaults())
    assert 'scene_roster' not in decisions['3']


def test_body_only_core_can_recover_borderline_cue_in_another_shot(tmp_path):
    cfg = configuration(tmp_path)
    with FaceDatabase(cfg.database) as state:
        db = state.connection
        item,ids = scene(db,cfg,vectors=[[1.,0.]]*3+[[.83,(1-.83**2)**.5]])
        for i,oid in enumerate(ids):
            db.execute("UPDATE body_observations SET quality=json_set(quality,'$.quality.view_shot',?) WHERE id=?",(i,oid))
        decisions = scene_plan(db,item['scan'],cfg)[2]
        assert decisions[ids[-1]]['reason']=='scene_body_attachment'
        assert decisions[ids[-1]]['person']==decisions[ids[0]]['person']
        assert decisions[ids[-1]]['role']=='auxiliary'


def test_scene_roster_does_not_access_unfrozen_external_face_banks(tmp_path):
    from stash_ai.recognition.body_identity import _load_scene, _plan
    cfg = configuration(tmp_path)
    with FaceDatabase(cfg.database) as state:
        db = state.connection
        item,ids = scene(db,cfg,vectors=[[1.,0.]]*3+[[.83,(1-.83**2)**.5]])
        for i,oid in enumerate(ids):
            db.execute("UPDATE body_observations SET quality=json_set(quality,'$.quality.view_shot',?) WHERE id=?",(i,oid))
        db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                   ('face',item['scan'],3,3.,'','[]','[]','{}',None,2,np.array([1.,0.],dtype='<f4').tobytes(),None))
        db.execute('UPDATE body_observations SET face=? WHERE id=?',('face',ids[-1]))
        data=_load_scene(db,item['scan'],cfg)
        serial=_plan(db,cfg,*data)[2]
        worker=_plan(None,cfg,*data,frozen_faces={})[2]
        assert serial==worker
