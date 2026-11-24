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
        return [correct(pixels) if callable(correct) else correct for pixels in images]
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
    pixels=np.arange(300*300*3,dtype=np.uint8).reshape(300,300,3)
    upright=np.rot90(pixels[crop[1]:crop[3],crop[0]:crop[2]],turn)
    model,calls=detector([wrong],lambda img:[answer] if np.array_equal(img,upright) else [])
    result=model.detect_many([pixels])
    np.testing.assert_allclose(result[0][0].row,expected.row,atol=1e-6)
    retries=[img for batch in calls[1:] for img in batch]
    assert len(retries)==(1 if box_consistent(wrong) else 4)
    assert sum(np.array_equal(img,upright) for img in retries)==1
    assert all(len(batch)<=2 for batch in calls[1:])
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
    # Distinguish actual crop orientations; returning one fixed result for
    # every orientation would describe conflicting detector measurements.
    pixels=np.arange(160*160*3,dtype=np.uint8).reshape(160,160,3)
    upright=np.rot90(pixels[crop[1]:crop[3],crop[0]:crop[2]],quarter_turn(wrong))
    model._raw_many=lambda images:[[wrong] if img.shape[:2]==(160,160) else
        [corrected] if np.array_equal(img,upright) else [] for img in images]
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
    from io import BytesIO
    with Image.fromarray(pixels[:,:,::-1]) as image:
        output=BytesIO();image.save(output,format='PNG')
        client.data['http://stash/sprite/1.jpg']=output.getvalue()
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


# Source coordinates from the supplied manifest, without media or scene metadata.
OBSERVED = checked_detection([
    -37.9178352355957,55.63352966308594,299.5679626464844,133.7397918701172,
    113.60991668701172,139.78175354003906,71.60289001464844,72.05097961425781,
    164.61428833007812,22.920503616333008,228.0191192626953,89.1190185546875,
    193.78591918945312,33.22037887573242,.8640366792678833])
OBSERVED_CROP = (0,22,337,223)


def synthetic_retry(turn):
    """A controlled model response, NOT inferred ground truth for the user's cue."""
    face=project(PROFILE,(0,0,300,300),(-turn)%4)
    points=(np.asarray(face.landmarks)-face.box[:2])/face.box[2:]*[260,134]+[0,55]
    return checked_detection([0,55,260,134,*points.ravel(),.85])


@pytest.mark.parametrize('turn',[0,1,2,3])
def test_out_of_box_axis_cannot_exclude_any_retry_orientation(turn):
    from stash_ai.recognition import landmark_repair_audit as audit
    assert quarter_turn(OBSERVED)==3
    assert not box_consistent(OBSERVED)
    expected=synthetic_retry(turn)
    answer=project(expected,OBSERVED_CROP,turn)
    pixels=np.arange(252*448*3,dtype=np.uint8).reshape(252,448,3)
    upright=np.rot90(pixels[22:223,0:337],turn)
    model,calls=detector([OBSERVED],lambda img:[answer] if np.array_equal(img,upright) else [])
    with audit.capture() as reports:
        result=model.detect(pixels)
    np.testing.assert_allclose(result[0].row,expected.row,atol=1e-6)
    trace=reports[0]['faces'][0]
    assert trace['original']==audit.geometry(OBSERVED)
    assert not trace['original_quality']['alignment_valid']
    assert trace['estimated_turn_degrees']==270
    assert trace['crop_xyxy']==list(OBSERVED_CROP)
    assert trace['tried_turn_degrees']==[0,90,180,270]
    assert trace['selected_turn_degrees']==turn*90 and trace['status']=='repaired'
    assert trace['result']==audit.geometry(result[0])
    candidate=trace['attempts'][turn]['candidates'][0]
    assert candidate['rejection'] is None
    assert candidate['crop_detection']==audit.geometry(answer)
    assert candidate['source_detection']==audit.geometry(result[0])
    assert [len(batch) for batch in calls]==[1,2,2]
    # Export must be numeric JSON; no image, tensor, or model object survives.
    import json
    json.dumps(reports,allow_nan=False)


def test_disagreeing_retry_orientations_preserve_original_and_explain_failure():
    from stash_ai.recognition import landmark_repair_audit as audit
    pixels=np.arange(252*448*3,dtype=np.uint8).reshape(252,448,3)
    choices={turn:project(synthetic_retry(turn),OBSERVED_CROP,turn) for turn in (1,3)}
    def responses(img):
        return [answer for turn,answer in choices.items()
                if np.array_equal(img,np.rot90(pixels[22:223,0:337],turn))]
    model,_=detector([OBSERVED],responses)
    with audit.capture() as reports:
        assert model.detect(pixels)==[OBSERVED]
    trace=reports[0]['faces'][0]
    assert trace['status']=='conflicting_orientations'
    assert trace['result']==trace['original']


@pytest.mark.parametrize('mode',['empty','weak','wrong_head','invalid_alignment','ambiguous'])
def test_failed_retry_exports_coordinates_and_rejection_without_png(mode):
    from stash_ai.recognition import landmark_repair_audit as audit
    pixels=np.arange(252*448*3,dtype=np.uint8).reshape(252,448,3)
    row=list(synthetic_retry(1).row)
    if mode=='weak':row[-1]=.7
    if mode=='wrong_head':row[0]+=100
    if mode=='invalid_alignment':
        for i in range(5,14,2):row[i]-=100
    answer=project(checked_detection(row),OBSERVED_CROP,1)
    upright=np.rot90(pixels[22:223,0:337],1)
    alternatives=[] if mode=='empty' else [answer]*(2 if mode=='ambiguous' else 1)
    model,_=detector([OBSERVED],lambda img:alternatives if np.array_equal(img,upright) else [])
    with audit.capture() as reports:
        assert model.detect(pixels)==[OBSERVED]
    trace=reports[0]['faces'][0]
    assert trace['status']=='no_eligible_candidate' and trace['result']==trace['original']
    attempt=trace['attempts'][1]
    if mode=='empty':
        assert attempt['candidates']==[]
    elif mode=='ambiguous':
        assert attempt['status']=='ambiguous_candidates'
        assert all(item['rejection'] is None for item in attempt['candidates'])
    else:
        candidate=attempt['candidates'][0]
        assert candidate['rejection']==dict(weak='lower_detection_score',
            wrong_head='different_face_box',invalid_alignment='invalid_alignment')[mode]
        assert candidate['crop_detection']==audit.geometry(answer)
        np.testing.assert_allclose(candidate['source_detection']['landmarks'],
                                   checked_detection(row).landmarks,atol=1e-6)


def test_retry_audit_is_scoped_and_resets_after_failure():
    from stash_ai.recognition import landmark_repair_audit as audit
    pixels=np.zeros((252,448,3),np.uint8)
    model,_=detector([OBSERVED],[])
    with audit.capture() as outer:
        model.detect(pixels)
        with pytest.raises(RuntimeError), audit.capture() as inner:
            model.detect(pixels)
            raise RuntimeError('failed inference')
        assert len(outer)==len(inner)==1
    before=len(outer)
    model.detect(pixels)
    assert len(outer)==before
    with audit.capture() as fresh:
        assert fresh==[]


def test_rotation_policy_version_invalidates_cached_geometry(monkeypatch):
    from stash_ai.recognition import rotated_faces
    from stash_ai.recognition.inference_contracts import face
    from stash_ai.recognition.scanning import scan_identity
    cfg=SimpleNamespace(data=defaults())
    cfg.data['sapiensid']['enabled']=True
    cfg.data['sapiensid']['collection']['enabled']=True
    cfg.data['detection']['pose_guided_rotation']=True
    current=(face(cfg),scan_identity(cfg,'ns'))
    monkeypatch.setattr(rotated_faces,'CONTRACT','rotated-head-v4-profile-repair')
    assert face(cfg)!=current[0] and scan_identity(cfg,'ns')!=current[1]
