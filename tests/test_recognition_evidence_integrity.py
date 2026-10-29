import math
import json
import numpy as np
import pytest
from PIL import Image
from stash_ai.recognition.database import FaceDatabase,encoded
from stash_ai.recognition.models import checked_detection
from stash_ai.recognition.body_identity import scene_plan,rebuild
from tests.test_recognition_body_identity import configuration,scene


@pytest.mark.parametrize('angle',[0,45,90,135,180,225,270,315])
def test_face_admission_is_rotation_invariant(angle):
    from stash_ai.recognition.quality import assess,rejection
    from stash_ai.recognition.config import defaults
    base=np.array([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],[41.5493,92.3655],[70.7299,92.2041]])
    a=math.radians(angle);rotation=np.array([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]])
    points=(base-[56,72])@rotation.T+[150,150]
    policy=defaults()['detection'];policy.update(min_face_pixels=1,min_sharpness=0,max_pose_deviation=.2,require_complete_face=False)
    with Image.new('RGB',(300,300)) as image:
        q=assess(image,checked_detection([90,90,120,120,*points.flatten(),.99]))
        assert q['landmarks_inside']
        assert rejection(q,policy) is None


def test_duplicate_auxiliary_pixels_do_not_add_support(tmp_path):
    from tests.test_recognition_pose_policy import configured,FRONT,PROFILE,membership
    from tests.test_recognition_clustering import scan
    from stash_ai.recognition.rebuild import rebuild_selected
    c=configured(tmp_path,auxiliary_min_support=2)
    with FaceDatabase(c.database) as state:
        sid,_=scan(state,c,'1',[FRONT]*3+[PROFILE]*2,count=5)
        state.connection.execute("UPDATE observations SET quality=json_set(quality,'$.pose_deviation',.8,'$.evidence_pixel_sha256',?) WHERE cue>=3",('a'*64,))
        state.connection.commit()
        rebuild_selected(state,[dict(scene='1',scan=sid)],c)
        assert all(r['status']=='withheld' for r in membership(state,sid)[3:])


def test_duplicate_body_pixels_do_not_change_attachment_score(tmp_path):
    c=configuration(tmp_path)
    c.data['sapiensid']['identities'].update(match_threshold=.8,cohesion_threshold=.45)
    def v(angle):
        a=math.radians(angle);return [math.cos(a),math.sin(a)]
    with FaceDatabase(c.database) as state:
        db=state.connection
        item,ids=scene(db,c,vectors=[v(0),v(30),v(30),v(60)]+[v(30)]*10)
        db.execute("UPDATE body_observations SET quality=json_set(quality,'$.quality.keypoints',4) WHERE id=?",(ids[3],))
        for oid in ids[4:]:
            db.execute("UPDATE body_observations SET quality=json_set(quality,'$.quality.pixel_digest','1-1') WHERE id=?",(oid,))
        _,_,decisions=scene_plan(db,item['scan'],c)
        assert decisions[ids[3]]['status']=='withheld'


@pytest.mark.parametrize('duplicate',[False,True])
def test_provisional_bodies_need_independent_cross_scene_support(tmp_path,duplicate):
    c=configuration(tmp_path);c.data['sapiensid']['global_matching']['enabled']=True
    with FaceDatabase(c.database) as state:
        db=state.connection
        a,_=scene(db,c,'1',vectors=[[1,0]]*2);b,ids=scene(db,c,'2',vectors=[[1,0]]*2)
        if duplicate:
            for i,oid in enumerate(ids):
                db.execute("UPDATE body_observations SET quality=json_set(quality,'$.quality.pixel_digest',?) WHERE id=?",('1-'+str(i),oid))
        db.commit()
        result=rebuild(state,[a,b],c,{a['scan']:[],b['scan']:[]})
        assert bool(db.execute('SELECT 1 FROM identity_merges').fetchone()) is not duplicate
        if not duplicate:
            assert len({r[0] for r in db.execute('SELECT person FROM body_members')})==1
            assert set(r[0] for r in db.execute('SELECT status FROM body_members'))=={'accepted'}
            # Revalidation of the same originals produces the same identity.
            original=list(db.execute('SELECT observation,person,status FROM body_members ORDER BY observation'))
            rebuild(state,[a,b],c,{a['scan']:[],b['scan']:[]})
            assert original==list(db.execute('SELECT observation,person,status FROM body_members ORDER BY observation'))


def test_provisional_confirmation_keeps_support_beyond_reference_cap(tmp_path):
    from stash_ai.recognition.body_global_banks import banks
    c=configuration(tmp_path)
    c.data['sapiensid']['identities']['min_support']=9
    c.data['sapiensid']['global_matching'].update(enabled=True,max_references=3)
    with FaceDatabase(c.database) as state:
        db=state.connection
        a,_=scene(db,c,'1',vectors=[[1,0]]*5);b,_=scene(db,c,'2',vectors=[[1,0]]*5)
        db.commit();rebuild(state,[a,b],c,{a['scan']:[],b['scan']:[]})
        bank=banks(db,c,'scope','ns')
        assert len(bank)==1
        assert next(iter(bank.values()))['support']==10
