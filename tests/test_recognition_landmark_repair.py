"""Translated, rolled profiles must be repaired before recognition consumes them."""
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from stash_ai.recognition.config import defaults
from stash_ai.recognition.models import checked_detection
from stash_ai.recognition.quality import alignment_measurements, assess, rejection
from stash_ai.recognition.rotated_faces import head_crop, restore, detect_head
from stash_ai.recognition.scrfd import ScrfdDetector
from stash_ai.recognition.face_landmark_repair import quarter_turn, box_consistent


PROFILE = checked_detection([70,60,60,100,98,85,100,86,90,100,98,125,102,125,.99])


def project(face, box, turns):
    """Source-to-crop projection; inverse is exercised against real pixels below."""
    l,t,r,b = box
    def point(x,y,edge=False):
        offset = 0 if edge else 1
        return ((x-l,y-t),(y-t,r-offset-x),(r-offset-x,b-offset-y),
                (b-offset-y,x-l))[turns]
    x,y,w,h = face.box
    a,c = point(x,y,True),point(x+w,y+h,True)
    return checked_detection([min(a[0],c[0]),min(a[1],c[1]),abs(a[0]-c[0]),abs(a[1]-c[1]),
                              *[v for p in face.landmarks for v in point(*p)],face.score])


def shifted(face,dy):
    row=list(face.row)
    for i in range(5,14,2):row[i]+=dy
    return checked_detection(row)


def detector(faces, correct, *, rotation=True, threshold=0.):
    model=object.__new__(ScrfdDetector)
    data=defaults()
    data['sapiensid']['enabled']=True
    data['sapiensid']['collection']['enabled']=True
    data['detection'].update(pose_guided_rotation=rotation,alignment_retry_error=threshold)
    data['execution']['detector_batch_size']=2
    model.config=SimpleNamespace(data=data)
    calls=[]
    def raw(images):
        calls.append([np.array(pixels) for pixels in images])
        if len(calls)==1:
            return [faces for _ in images]
        return [correct for _ in images]
    model._raw_many=raw
    return model,calls


@pytest.mark.parametrize('turns',[1,2,3])
@pytest.mark.parametrize('dy',[-15,-70])
def test_existing_sideways_profile_is_repaired_even_when_shape_error_is_unchanged(turns,dy):
    expected=project(PROFILE,(0,0,300,300),turns)
    wrong=shifted(expected,dy)
    assert alignment_measurements(wrong)['alignment_error']==pytest.approx(
        alignment_measurements(expected)['alignment_error'],abs=1e-7)
    turn=quarter_turn(wrong)
    x,y,w,h=wrong.box
    crop=(int(x-.25*w),int(y-.25*h),int(x+1.25*w),int(y+1.25*h))
    answer=project(expected,crop,turn)
    model,calls=detector([wrong],[answer])
    pixels=np.arange(300*300*3,dtype=np.uint8).reshape(300,300,3)
    result=model.detect_many([pixels])
    np.testing.assert_allclose(result[0][0].row,expected.row,atol=1e-6)
    assert len(calls)==2
    np.testing.assert_array_equal(calls[1][0],np.rot90(pixels[crop[1]:crop[3],crop[0]:crop[2]],turn))
    assert model.config.data['detection']['alignment_retry_error']==0.


def test_outside_landmarks_trigger_local_retry_without_increased_shape_error():
    wrong=shifted(PROFILE,-70)
    x,y,w,h=wrong.box
    crop=(int(x-.25*w),int(y-.25*h),int(x+1.25*w),int(y+1.25*h))
    model,calls=detector([wrong],[project(PROFILE,crop,0)],rotation=False,threshold=.2)
    assert model.detect(np.zeros((300,300,3),np.uint8))[0].row==PROFILE.row
    assert len(calls)==2


@pytest.mark.parametrize('mode',['wrong_head','competing_face','ambiguous','weak','still_shifted'])
def test_repairs_cannot_replace_a_head_with_a_neighbour_or_bad_measurement(mode):
    good=project(PROFILE,(0,0,300,300),3)
    wrong=shifted(good,-15)
    turn=quarter_turn(wrong)
    x,y,w,h=wrong.box
    crop=(int(x-.25*w),int(y-.25*h),int(x+1.25*w),int(y+1.25*h))
    faces=[wrong]
    row=list(good.row)
    if mode=='wrong_head':
        row[0]+=50
        for i in range(4,14,2):row[i]+=50
    elif mode=='weak':row[-1]=.7
    elif mode=='still_shifted':row=list(shifted(good,-70).row)
    elif mode=='competing_face':faces.append(good)
    answers=[project(checked_detection(row),crop,turn)]
    if mode=='ambiguous':answers*=2
    model,_=detector(faces,answers)
    assert model.detect(np.zeros((300,300,3),np.uint8))[0].row==wrong.row


def test_upright_faces_and_disabled_retries_do_not_add_inference():
    for face,enabled in ((PROFILE,True),(project(PROFILE,(0,0,300,300),1),False)):
        model,calls=detector([face],[],rotation=enabled)
        assert model.detect(np.zeros((300,300,3),np.uint8))==[face]
        assert len(calls)==1


def test_landmark_location_gate_is_independent_of_optional_pose_filter():
    policy=defaults()['detection']
    policy.update(min_face_pixels=8,min_sharpness=0,max_pose_deviation=None)
    with Image.new('RGB',(300,300)) as image:
        valid=assess(image,PROFILE)
        displaced=assess(image,shifted(PROFILE,-70))
        from stash_ai.recognition.alignment_audit import _diagnostics
        diagnostics,_,_=_diagnostics(image,shifted(PROFILE,-70))
    assert rejection(valid,policy) is None
    assert not displaced['landmark_box_valid']
    assert not displaced['alignment_valid']
    assert not diagnostics['alignment_valid'] and not diagnostics['landmark_box_valid']
    assert rejection(displaced,policy)=='face_landmarks_invalid'
    # Small detector edge noise is distinct from global displacement.
    row=list(PROFILE.row);row[4]=69.
    assert box_consistent(checked_detection(row))


@pytest.mark.parametrize('turns',[0,1,2,3])
@pytest.mark.parametrize('crop',[(31,47,171,158),(0,0,140,111),(31,0,171,111)])
def test_quarter_turn_restore_matches_pixels_in_non_square_clipped_crops(turns,crop):
    with Image.new('RGB',(240,200)) as source:
        marks=[(61,73),(67,78),(83,90),(104,105),(116,119)]
        for i,p in enumerate(marks):source.putpixel(p,(i+1,0,0))
        with source.crop(crop) as tile:
            raster=np.rot90(np.asarray(tile),turns)
        valid=[p for p in marks if crop[0]<=p[0]<crop[2] and crop[1]<=p[1]<crop[3]]
        local=[]
        for p in valid:
            yy,xx=np.where(raster[:,:,0]==marks.index(p)+1)
            local.append((float(xx[0]),float(yy[0])))
        while len(local)<5:local.append(local[-1])
        face=checked_detection([0,0,raster.shape[1],raster.shape[0],*np.array(local).ravel(),.99])
        result=restore(face,crop,(0,True,180,False)[turns])
    np.testing.assert_allclose(result.landmarks[:len(valid)],valid,atol=0)
    assert result.box==(crop[0],crop[1],crop[2]-crop[0],crop[3]-crop[1])


def profile_person(face):
    return dict(bounds=[0,0,300,300],keypoints=[list(face.landmarks[2]),list(face.landmarks[1]),
                [-1,-1]]+[[-1,-1]]*14,keypoint_scores=[.99,.99]+[0]*15)


@pytest.mark.parametrize('turns',[1,3])
@pytest.mark.parametrize('far_eye',['missing','weak','borrowed'])
def test_missing_far_eye_tries_rotations_and_requires_original_pose_agreement(turns,far_eye):
    face=project(PROFILE,(0,0,300,300),turns)
    person=profile_person(face)
    if far_eye!='missing':
        person['keypoints'][2]=[face.landmarks[0][0]+40,face.landmarks[0][1]]
        person['keypoint_scores'][2]=.1 if far_eye=='weak' else .99
        if far_eye=='borrowed':person['pose_refinement']={'face_prompt':(2,person['keypoints'][2])}
    import copy
    original=copy.deepcopy(person)
    box,turn=head_crop(person,(300,300))
    assert turn is None
    calls=[]
    desired=quarter_turn(face)
    def detect(image):
        current=len(calls);calls.append(image.size)
        return [project(face,box,current)] if current==desired else []
    with Image.new('RGB',(300,300)) as image:
        result=detect_head(SimpleNamespace(detect=detect),image,box,turn,[person],0,[])
    assert len(calls)==4
    assert result is not None and result[0].row==face.row
    assert person==original


@pytest.mark.parametrize('source',['face','temporal','conflicted','recovered'])
def test_borrowed_or_conflicted_head_points_do_not_authorize_rotation(source):
    person=profile_person(PROFILE)
    if source=='face':person['pose_refinement']={'face_prompt':(0,[90,100])}
    elif source=='temporal':person['pose_refinement']={'temporal_prompts':[(0,[90,100])]}
    else:person['keypoint_states']=[source,'measured']+['unavailable']*15
    assert head_crop(person,(300,300)) is None


@pytest.mark.parametrize('mode',['contradictory_pose','contested_pose','existing_face','shifted_existing_face'])
def test_profile_recovery_refuses_inconsistent_or_competing_ownership(mode):
    face=project(PROFILE,(0,0,300,300),3)
    person=profile_person(face)
    box,turn=head_crop(person,(300,300))
    people=[person]
    faces=[]
    if mode=='contradictory_pose':person['keypoints'][0][0]+=30
    elif mode=='contested_pose':people.append(profile_person(face))
    else:faces=[shifted(face,-40) if mode=='shifted_existing_face' else face]
    calls=[]
    def detect(image):
        current=len(calls);calls.append(image.size)
        return [project(face,box,current)] if current==quarter_turn(face) else []
    with Image.new('RGB',(300,300)) as image:
        assert detect_head(SimpleNamespace(detect=detect),image,box,turn,people,0,faces) is None


def test_compressed_but_present_far_eye_does_not_disable_profile_recovery():
    person=profile_person(PROFILE)
    person['keypoints'][2]=list(PROFILE.landmarks[0])
    person['keypoint_scores'][2]=.99
    assert head_crop(person,(300,300))[1] is None


def test_one_eye_recovery_is_persisted_once_with_source_coordinates(tmp_path):
    import json
    from stash_ai.recognition import rotated_faces
    from stash_ai.recognition.database import FaceDatabase, encoded
    from tests.test_recognition_body_identity import configuration, scene
    cfg=configuration(tmp_path)
    cfg.data['detection'].update(pose_guided_rotation=True,min_face_pixels=8)
    face=project(PROFILE,(0,0,300,300),3)
    person=profile_person(face)
    box,turn=head_crop(person,(300,300))
    calls=[]
    class Backend:
        dimensions=2
        def detect(self,image):
            current=len(calls);calls.append(image.size)
            return [project(face,box,current)] if current==quarter_turn(face) else []
        def embed(self,image,detected):
            assert detected.row==face.row
            return np.array([1.,0.])
        def close(self):pass
    cue=SimpleNamespace(start=0.,rect=(0,0,300,300),image='https://example.test/sprite.jpg')
    source=SimpleNamespace(cues=[cue],frames_indices=lambda ids:(
        (i,cue,Image.new('RGB',(300,300))) for i in ids))
    with FaceDatabase(cfg.database) as state:
        db=state.connection
        item,ids=scene(db,cfg,vectors=[[1,0]])
        db.execute('UPDATE body_observations SET geometry=?',(encoded(person),))
        rotated_faces.recover(db,item['scan'],source,cfg,Backend,None,lambda:None)
        row=db.execute('SELECT o.* FROM observations o JOIN body_observations b ON b.face=o.id').fetchone()
        assert json.loads(row['geometry'])==list(face.row)
        assert row['embedding'] and row['rejected'] is None
        assert json.loads(row['quality'])['rotated_head_degrees']==90
        rotated_faces.recover(db,item['scan'],source,cfg,lambda:pytest.fail('repeated inference'),None,lambda:None)
        assert db.execute('SELECT count(*) FROM observations').fetchone()[0]==1
        assert db.execute('SELECT count(*) FROM person_evidence').fetchone()[0]==1
    assert len(calls)==4


def test_scan_embeds_and_persists_repaired_geometry_before_downstream_collection(tmp_path):
    import json
    from dataclasses import replace
    from stash_ai.recognition.config import plain
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.media import SpriteInput
    from stash_ai.recognition.scanning import scan_scene
    from tests.helpers.recognition import Backend, MediaClient
    from tests.test_recognition_detection_quality import settings
    cfg=settings(tmp_path,min_face_pixels=8,max_pose_deviation=None,min_visible_face_fraction=0)
    data=plain(cfg.data)
    data['detection']['pose_guided_rotation']=True
    data['sapiensid']['enabled']=True
    data['sapiensid']['collection']['enabled']=True
    cfg=replace(cfg,data=data)
    row=list(PROFILE.row)
    row[:14]=[v*.5 for v in row[:14]]
    expected=project(checked_detection(row),(0,0,160,160),3)
    wrong=shifted(expected,-15)
    x,y,w,h=wrong.box
    import math
    crop=(max(0,math.floor(x-.25*w)),max(0,math.floor(y-.25*h)),
          min(160,math.ceil(x+1.25*w)),min(160,math.ceil(y+1.25*h)))
    model,_=detector([wrong],[])
    model.config=cfg
    corrected=project(expected,crop,quarter_turn(wrong))
    model._raw_many=lambda images:[[wrong] if img.shape[:2]==(160,160) else [corrected] for img in images]
    backend=Backend();embedded=[];collected=[]
    backend.detect=lambda image:model.detect(np.asarray(image)[:,:,::-1])
    def embed(image,face):
        embedded.append(face.row)
        return np.array([1.,0,0,0])
    backend.embed=embed
    def on_cue(db,scan,index,image):
        collected.extend(json.loads(row[0]) for row in db.execute(
            'SELECT geometry FROM observations WHERE scan=? AND cue=?',(scan,index)))
    client=MediaClient(count=1)
    with FaceDatabase(cfg.database) as db, SpriteInput(client,client.scene,cfg) as source:
        scan,reused=scan_scene(db,'scope',client.scene,cfg,'ns',source,lambda:backend,on_cue=on_cue)
        record=db.connection.execute('SELECT geometry,quality,rejected,embedding FROM observations WHERE scan=?',(scan,)).fetchone()
    assert not reused and embedded==[expected.row]
    assert collected==[list(expected.row)]
    assert json.loads(record['geometry'])==list(expected.row)
    assert json.loads(record['quality'])['landmark_box_valid']
    assert record['rejected'] is None and record['embedding'] is not None


def test_face_cache_and_scan_contract_change_when_rotation_becomes_active(tmp_path):
    from stash_ai.recognition.inference_contracts import face
    from stash_ai.recognition.scanning import scan_identity
    cfg=SimpleNamespace(data=defaults())
    cfg.data['sapiensid']['enabled']=True
    cfg.data['sapiensid']['collection']['enabled']=True
    old_face=face(cfg);old_scan=scan_identity(cfg,'ns')
    cfg.data['detection']['pose_guided_rotation']=True
    assert face(cfg)!=old_face and scan_identity(cfg,'ns')!=old_scan
    cfg.data['sapiensid']['collection']['enabled']=False
    assert face(cfg)==old_face and scan_identity(cfg,'ns')==old_scan
