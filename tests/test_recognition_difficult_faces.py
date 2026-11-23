import numpy as np
from PIL import Image
from stash_ai.recognition.models import checked_detection
from stash_ai.recognition.quality import assess
from stash_ai.recognition.rotated_faces import head_crop,restore,duplicate_face


def test_profile_axis_is_shared_with_pose_gate():
    face=checked_detection([10,10,80,80,48,40,49,42,49,51,45,65,53,65,.99])
    with Image.new('RGB',(100,100)) as image:quality=assess(image,face)
    assert quality['landmarks_inside'] and quality['pose_deviation']<.5


def test_upside_down_head_and_restore():
    crop=head_crop(dict(keypoints=[[50,40],[60,50],[40,50]]),(100,100))
    assert crop[1]==180
    face=checked_detection([10,20,30,40,20,30,30,30,25,40,20,50,30,50,.9])
    result=restore(face,[0,0,100,100],180)
    assert tuple(result.box)==(60,40,30,40)


def test_overlap_alone_is_not_a_duplicate_face():
    face=checked_detection([10,20,30,40,20,30,30,30,25,40,20,50,30,50,.9])
    row=np.asarray(face.row).copy();row[0]+=25;row[4:14:2]+=25
    assert not duplicate_face(face,checked_detection(row))
    assert duplicate_face(face,face)


def test_partial_head_requires_reliable_uncontested_points():
    person=dict(keypoints=[[-1,-1],[40,40],[50,40],[-1,-1],[-1,-1]],keypoint_scores=[0,.2,.9,0,0])
    assert head_crop(person,(100,100)) is None
    person['keypoint_scores'][1]=.9
    assert head_crop(person,(100,100)) is not None
    person['keypoint_states']=['unavailable','conflicted','measured','unavailable','unavailable']
    assert head_crop(person,(100,100)) is None


def test_calibrated_sharpness_units_and_profile_floor_are_explicit():
    from stash_ai.recognition.quality import sharpness_failed
    q=dict(sharpness=1.,sharpness_normalized=1.,face_profile=True)
    policy=dict(min_sharpness=2.,sharpness_metric='normalized',profile_min_sharpness=.5)
    assert not sharpness_failed(q,policy)
    assert sharpness_failed(dict(q,face_profile=False),policy)
    assert not sharpness_failed(q,dict(policy,sharpness_metric='raw'))
