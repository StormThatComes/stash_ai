from types import SimpleNamespace

import numpy as np
from PIL import Image

from stash_ai.recognition import rotated_faces, temporal_prompts
from stash_ai.recognition.database import FaceDatabase, encoded
from stash_ai.recognition.models import checked_detection
from tests.test_recognition_body_identity import configuration, scene


def test_temporal_points_require_one_adjacent_consistent_torso():
    sig='48'*48
    b=(5,5,55,95)
    old=(('s',2,1,sig),[(b,{5:np.array([15,20]),6:np.array([40,21])})])
    current=(('s',3,1,sig),[(b,{5:np.array([16,20])})])
    def apply(frame):
        record={}
        ids=temporal_prompts.borrow(old,frame,0,np.array([5]),2,record)
        return list(ids),record
    assert apply(current)==([5,6],{'temporal_prompts':[(6,[41.,21.])],'temporal_source_cue':2})
    for context in [('s',4,1,sig),('t',3,1,sig),('s',3,2,sig),('s',3,1,'ff'*48)]:
        assert apply((context,current[1]))==([5],{})
    assert apply((current[0],current[1]*2))==([5],{})


def test_rotated_face_becomes_normal_observation_once(tmp_path):
    cfg=configuration(tmp_path)
    cfg.data['detection'].update(pose_guided_rotation=True,min_face_pixels=8)
    person=dict(bounds=[0,0,160,160],keypoints=[[76,75],[85,91],[85,59]]+[[-1,-1]]*14)
    box,turn=rotated_faces.head_crop(person,(160,160))
    assert turn
    # Project known original landmark centers into the counterclockwise retry crop.
    def rotated(x,y):
        return y-box[1],box[2]-1-x
    marks=[v for xy in [(85,59),(85,91),(76,75),(62,63),(62,87)] for v in rotated(*xy)]
    det=checked_detection([42-box[1],box[2]-112,66,65,*marks,.99])
    class Backend:
        dimensions=2
        detect=lambda self,image:[det]
        embed=lambda self,image,face:np.array([1.,0.])
        close=lambda self:None
    cue=SimpleNamespace(start=0.,rect=(0,0,160,160),image='https://example.test/sprite.jpg')
    source=SimpleNamespace(cues=[cue],frames_indices=lambda ids:((i,cue,Image.new('RGB',(160,160))) for i in ids))
    with FaceDatabase(cfg.database) as state:
        db=state.connection
        item,ids=scene(db,cfg,vectors=[[1,0]])
        db.execute('UPDATE body_observations SET geometry=?',(encoded(person),))
        rotated_faces.recover(db,item['scan'],source,cfg,Backend,None,lambda:None)
        row=db.execute('SELECT o.*,b.face FROM observations o JOIN body_observations b ON b.face=o.id').fetchone()
        assert row and row['rejected'] is None and row['embedding'] and row['id']==row['face']
        assert db.execute('SELECT count(*) FROM person_evidence WHERE observation=?',(row['id'],)).fetchone()[0]==1
        rotated_faces.recover(db,item['scan'],source,cfg,lambda:1/0,None,lambda:None)
        assert db.execute('SELECT count(*) FROM observations').fetchone()[0]==1


def test_only_active_guided_rotation_versions_face_scan(tmp_path):
    from stash_ai.recognition.scanning import scan_identity
    cfg=configuration(tmp_path)
    old=scan_identity(cfg,'ns')
    cfg.data['detection']['pose_guided_rotation']=True
    assert scan_identity(cfg,'ns')!=old
    cfg.data['sapiensid']['collection']['enabled']=False
    assert scan_identity(cfg,'ns')==old
