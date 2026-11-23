"""Pose gates must ignore image roll, including changes to detector box extents."""
from contextlib import closing
from dataclasses import replace

import numpy as np
import pytest

from stash_ai.recognition.quality import POSE_CONTRACT, pose_measurements, rejection


FRONTAL = np.array([[-20,0],[20,0],[0,15],[-12,30],[12,30]], dtype=float)
NARROW = np.array([[-7,0],[7,0],[0,15],[-12,30],[-8,30]], dtype=float)
PROFILE = np.array([[48,40],[49,42],[49,51],[45,65],[53,65]], dtype=float)
BOUNDARY = np.array([[-6,0],[6,0],[4,15],[-12,30],[12,30]], dtype=float)
NOSE_BOUNDARY = np.array([[-1,0],[1,0],[2,3.6],[-5,30],[5,30]], dtype=float)
SPAN_BOUNDARY = np.array([[-.4,0],[.4,0],[0,1.5],[-1,3],[1,3]], dtype=float)


def rotate(points, degrees):
    angle = np.deg2rad(degrees)
    matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return points @ matrix.T, tuple(np.abs(matrix) @ [80,100])


@pytest.mark.parametrize('points', [FRONTAL, NARROW, PROFILE, BOUNDARY, NOSE_BOUNDARY, SPAN_BOUNDARY],
                         ids=['frontal','narrow','profile','eye-boundary','nose-boundary','span-boundary'])
def test_pose_is_constant_over_full_roll_and_rotated_boxes(points):
    expected = pose_measurements(points, (80,100))
    for degrees in range(-180,181,5):
        rotated, box = rotate(points,degrees)
        for size in (box, (20,200), (200,20), None):
            actual = pose_measurements(rotated+[300,200],size)
            assert actual['pose_deviation'] == pytest.approx(expected['pose_deviation'],abs=1e-10), degrees
            assert actual['pose_contract'] == POSE_CONTRACT


def test_borderline_box_can_no_longer_turn_roll_into_pose_rejection():
    from stash_ai.recognition.config import defaults
    from stash_ai.recognition.pose_policy import is_anchor
    from stash_ai.recognition.quality import portrait_components
    policy = dict(defaults()['detection'], min_face_pixels=0,min_sharpness=0,max_pose_deviation=.05)
    grouping = dict(defaults()['clustering'], max_anchor_pose_deviation=.05)
    for degrees in (0,30,45,90,180):
        points,size = rotate(NARROW,degrees)
        quality = dict(pose_measurements(points,size), pixels=80,sharpness=100,
                       complete=True,landmarks_inside=True,exposure=128.,clipped_fraction=0.)
        assert quality['pose_deviation'] == pytest.approx(0.,abs=1e-12)
        assert rejection(quality,policy) is None
        assert is_anchor(quality,grouping)
        assert portrait_components(quality,defaults()['portraits'])['frontal'] == pytest.approx(1.)


@pytest.mark.parametrize('nose,expected', [([8,15],.2),([0,5],1/3),([0,25],1/3),([8,5],1/3)])
def test_yaw_and_pitch_proxies_still_increase_deviation_at_every_roll(nose,expected):
    points = FRONTAL.copy();points[2] = nose
    for degrees in range(0,360,15):
        rotated,size = rotate(points,degrees)
        for scale in (.5,1.,3.):
            assert pose_measurements(rotated*scale,size)['pose_deviation'] == pytest.approx(expected,abs=1e-10)


def test_roll_diagnostic_is_separate_from_pose_gate():
    rotated,size = rotate(FRONTAL,90)
    quality = pose_measurements(rotated,size)
    assert quality['roll_deviation'] == pytest.approx(1.)
    assert quality['pose_deviation'] == pytest.approx(0.)


@pytest.mark.parametrize('points', [np.zeros((5,2)),np.full((5,2),np.nan),np.ones((4,2))])
def test_degenerate_landmarks_keep_conservative_pose_result(points):
    assert pose_measurements(points) == dict(pose_deviation=1.,roll_deviation=1.,pose_contract=POSE_CONTRACT)


def test_old_pixel_quality_cache_cannot_reuse_box_dependent_pose(tmp_path):
    from stash_ai.recognition.inference_cache import Store,using,many
    from stash_ai.recognition.quality import assess
    from tests.helpers.recognition import FACE,textured_image
    with closing(Store(tmp_path/'quality.sqlite',4*1024**2)) as store,using(store),textured_image() as image:
        old = dict(pose_deviation=.9,pose_contract='profile-frame-v3')
        many('face_quality',('face-quality-v9-model-input',False),[(image,tuple(FACE.row))],lambda values:[old])
        current = assess(image,FACE)
        assert current['pose_contract'] == POSE_CONTRACT and current['pose_deviation'] < .9


@pytest.mark.parametrize('visibility', [0.,.8])
def test_scan_and_portrait_policies_refresh_pose_measurements(tmp_path,visibility):
    from stash_ai.recognition.config import plain,digest,DEFAULT_FACEOCC_LOGIT_THRESHOLD
    from stash_ai.recognition.scanning import scan_identity
    from stash_ai.recognition.visibility import CONTRACT_ID
    from stash_ai.recognition.portraits import policy_identity,_policy_value
    from tests.test_recognition_clustering import setup
    config = setup(tmp_path);data=plain(config.data)
    data['detection']['min_visible_face_fraction']=visibility
    config=replace(config,data=data)
    detection=dict(data['detection'])
    for key in ('candidate_score_threshold','alignment_retry_error','pose_guided_rotation'):
        detection.pop(key,None)
    assert detection.pop('faceocc_logit_threshold') == DEFAULT_FACEOCC_LOGIT_THRESHOLD
    previous=digest(dict(namespace='ns',detection=detection,max_cues=data['execution']['max_cues'],
                         quality_format=10,visibility_contract=CONTRACT_ID if visibility else None))
    assert scan_identity(config,'ns') != previous
    assert policy_identity(config) != digest(dict(_policy_value(config),version=13))
