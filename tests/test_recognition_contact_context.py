import numpy as np
from PIL import Image
from stash_ai.recognition.temporal_prompts import borrow
from stash_ai.recognition.models import checked_detection
from stash_ai.recognition.visibility import prepare_aligned
from benchmarks.benchmark_dance_contact import contour_pixels


def test_partial_box_overlap_allows_only_uncontested_rotated_torso_path():
    signature='21'*48
    angle=np.deg2rad(12)
    rotation=np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])
    before={5:np.array([25.,25]),6:np.array([52.,25]),11:np.array([28.,57])}
    center=np.array([40.,45])
    def move(point):
        return (np.asarray(point)-center) @ rotation * 1.05 + center
    rect=(10,10,70,80)
    corners=move([[10,10],[70,10],[10,80],[70,80]])
    new=tuple(np.r_[corners.min(axis=0),corners.max(axis=0)])
    current={k:move(before[k]) for k in (5,6)}
    old=(('a',3,0,signature),[(rect,before)])
    def check(peer,shot):
        record={}
        now=(('a',4,shot,signature),[(new,current),(peer,{})])
        borrow(old,now,0,np.array([5,6]),3,record)
        return record
    record=check((60,75,95,99),0)
    np.testing.assert_allclose(record['temporal_prompts'][0][1],move(before[11]))
    assert not check((20,50,44,70),0) and not check((60,75,95,99),4)


def test_subpixel_profile_axis_requires_nose_mouth_support():
    face=checked_detection([10,10,80,80,48,40,48.9,40,49,51,45,65,53,65,.99])
    with Image.new('RGB',(100,100)) as image:
        _,region,_,affine=prepare_aligned(image,face,0.)
        assert region.any() and affine[0]>0


def test_missing_person_still_contributes_contour_denominator():
    photo=dict(size=[20,20],people=[dict(box=[0,0,10,10],head=None)])
    result=contour_pixels(photo,[],[dict(person=0,limb='hand',points=[[1,1],[3,1],[3,3],[1,3]])])
    assert result==[dict(person=0,limb='hand',labelled=9,covered=0,claimed_by_partner=0)]
