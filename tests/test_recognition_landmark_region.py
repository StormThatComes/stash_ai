"""Landmarks locate the face even when a detector box is loose or displaced."""
import numpy as np
from PIL import Image
import pytest

from stash_ai.recognition.face_region import fit, ellipse
from stash_ai.recognition.models import checked_detection
from stash_ai.recognition.visibility import prepare_aligned, VisibilityGeometryError
from tests.helpers.recognition import FACE
from tests.test_recognition_face_profile_geometry import tilted_face


def with_geometry(face, *, points=None, box=None):
    return checked_detection([*(face.box if box is None else box),
                              *np.asarray(face.landmarks if points is None else points).ravel(),face.score])


@pytest.mark.parametrize('profile', [-1,0,1])
@pytest.mark.parametrize('angle', [0,30,45,60,90,180])
def test_landmarks_translate_ellipse_inside_unchanged_box(profile, angle):
    face,_ = tilted_face(angle,profile)
    original = fit(face)
    shift = np.array([19.,-13.])
    moved = fit(with_geometry(face,points=np.asarray(face.landmarks)+shift))
    np.testing.assert_allclose(moved.center,np.asarray(original.center)+shift,atol=1e-10)
    np.testing.assert_allclose(moved.radii,original.radii,rtol=1e-10)
    np.testing.assert_allclose(moved.axis,original.axis,atol=1e-10)
    assert moved.view == original.view


@pytest.mark.parametrize('profile', [-1,0,1])
@pytest.mark.parametrize('box', [[40,30,240,250],[90,80,150,180],[130,100,90,120]])
def test_box_changes_neither_ellipse_nor_model_pixels(profile,box):
    face,_ = tilted_face(45,profile)
    other = with_geometry(face,box=box)
    assert fit(face) == fit(other)
    rng=np.random.default_rng(66)
    with Image.fromarray(rng.integers(0,256,(320,320,3),dtype='uint8')) as image:
        before = prepare_aligned(image,face,1.)
        after = prepare_aligned(image,other,1.)
    np.testing.assert_array_equal(before[0],after[0])
    np.testing.assert_array_equal(before[1],after[1])
    assert before[2:] == after[2:]


@pytest.mark.parametrize('profile', [-1,0,1])
@pytest.mark.parametrize('angle', [-135,-60,-45,30,45,60,135])
def test_fixed_box_does_not_interfere_with_roll_or_face_size(profile,angle):
    face,_ = tilted_face(0,profile)
    start = fit(face)
    c,s=np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))
    rotation=np.array([[c,-s],[s,c]])
    pivot=np.array([160.,160.])
    scale=.8
    points=(np.asarray(face.landmarks)-pivot) @ rotation.T*scale+pivot
    result=fit(with_geometry(face,points=points))
    np.testing.assert_allclose(result.axis,np.asarray(start.axis) @ rotation.T,atol=1e-9)
    np.testing.assert_allclose(result.radii,np.asarray(start.radii)*scale,rtol=1e-9)
    np.testing.assert_allclose(np.asarray(result.center)-.5,
                              (np.asarray(start.center)-.5-pivot) @ rotation.T*scale+pivot,atol=1e-9)


@pytest.mark.parametrize('index', range(5))
def test_one_bad_landmark_does_not_drag_whole_face_region(index):
    baseline=fit(FACE)
    points=np.asarray(FACE.landmarks).copy()
    points[index]+=[40,30]
    result=fit(with_geometry(FACE,points=points))
    assert result.ignored == ('left_eye','right_eye','nose','left_mouth','right_mouth')[index]
    assert np.linalg.norm(np.asarray(result.center)-baseline.center)<1.
    np.testing.assert_allclose(result.radii,baseline.radii,rtol=.03)
    assert np.dot(result.axis,baseline.axis)>.999


@pytest.mark.parametrize('profile', [-1,1])
def test_profile_width_is_not_the_tiny_eye_projection(profile):
    face,_=tilted_face(0,profile)
    result=fit(face)
    eyes=np.asarray(face.landmarks[:2])
    assert result.radii[0] > 10*np.linalg.norm(eyes[1]-eyes[0])
    # The eye/mouth column belongs near the profile edge; the cheek is behind it.
    eye_center=eyes.mean(axis=0)
    assert profile*(eye_center[0]-(result.center[0]-.5))>20
    assert result.view==('profile-left' if profile<0 else 'profile-right')


def test_incoherent_landmarks_report_unmeasurable_instead_of_optimizing_visibility():
    points=np.array([[20,20],[130,30],[50,120],[120,20],[20,130]])
    with pytest.raises(VisibilityGeometryError,match='consistent landmarks'):
        fit(with_geometry(FACE,points=points))


def test_model_crop_contains_entire_shifted_region():
    face,_=tilted_face(45)
    face=with_geometry(face,points=np.asarray(face.landmarks)+[50,0])
    with Image.new('RGB',(400,400)) as image:
        _,region,total,_=prepare_aligned(image,face,1.)
    assert region.sum()==total and total>0
    assert not any((region[0].any(),region[-1].any(),region[:,0].any(),region[:,-1].any()))


@pytest.mark.parametrize('profile', [-1,0,1])
def test_subpixel_jitter_does_not_move_profile_cheek_to_another_width(profile):
    face,_=tilted_face(45,profile)
    baseline=fit(face)
    for seed in range(20):
        points=np.asarray(face.landmarks)+np.random.default_rng(seed).normal(0,.3,(5,2))
        result=fit(with_geometry(face,points=points))
        assert np.linalg.norm(np.asarray(result.center)-baseline.center)<2.
        assert result.view==baseline.view


def test_legacy_comparison_cannot_use_a_truncated_denominator():
    from stash_ai.recognition.inspection_face_visibility import legacy_roi
    face,_=tilted_face(45)
    face=with_geometry(face,box=[20,20,280,280])
    with Image.new('RGB',(320,320)) as image:
        _,region,total,affine=prepare_aligned(image,face,1.)
        assert total==region.sum() and legacy_roi(face,image.size,affine) is None
