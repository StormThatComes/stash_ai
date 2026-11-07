"""Partial face boxes require anatomically corresponding measured head points."""
from copy import deepcopy
from threading import get_ident

from PIL import Image
import pytest

from stash_ai.recognition.detection_associations import explain_associations
from stash_ai.recognition.models import checked_detection
from stash_ai.recognition.sapiensid_backend import SapiensIDFaces, associate, association_rejection


FACE = checked_detection([20, 20, 60, 80, 35, 45, 65, 45, 50, 60, 40, 80, 60, 80, .99])


def person(bounds=(28, 0, 100, 140)):
    return dict(bounds=list(bounds), score=.99,
                keypoints=[[50., 60.], [65., 45.], [35., 45.]] + [[-1., -1.]] * 14)


@pytest.mark.parametrize('bounds', [[28, 0, 100, 140], [0, 0, 72, 140],
                                    [0, 30, 100, 140], [0, 0, 100, 88], [28, 30, 72, 88]])
def test_partial_edges_and_corners_match_without_modifying_geometry(bounds):
    pose = person(bounds)
    before = deepcopy(pose)
    assert associate([FACE], [pose]) == [0]
    assert explain_associations([FACE], [pose], [0])[0]['reason'] is None
    assert pose == before


@pytest.mark.parametrize('missing', [0, 1, 2])
def test_two_corresponding_landmarks_suffice_when_third_is_unavailable(missing):
    pose = person()
    pose['keypoints'][missing] = [-1., -1.]
    assert associate([FACE], [pose]) == [0]


def test_one_head_point_is_insufficient_only_for_partial_boxes():
    pose = person()
    pose['keypoints'][1:3] = [[-1., -1.]] * 2
    assert association_rejection(FACE, pose) == 'insufficient_head_landmarks'
    pose['bounds'][0] = 0
    assert associate([FACE], [pose]) == [0]  # existing contained-face behavior


def test_in_box_landmarks_of_another_face_cannot_rescue_partial_box():
    pose = person()
    pose['keypoints'][:3] = [[60, 70], [75, 55], [45, 55]]
    assert association_rejection(FACE, pose) == 'head_landmark_mismatch'
    assert associate([FACE], [pose]) == [None]


def test_eye_indices_are_anatomical_and_a_conflicting_third_point_vetoes():
    pose = person()
    pose['keypoints'][1:3] = list(reversed(pose['keypoints'][1:3]))
    assert association_rejection(FACE, pose) == 'head_landmark_mismatch'
    pose = person()
    pose['keypoints'][0][0] += 10  # the two perfect eyes do not hide a wrong nose
    assert association_rejection(FACE, pose) == 'head_landmark_mismatch'
    pose['keypoints'][0][0] -= 1  # exactly 15% of the 60px shorter side
    assert associate([FACE], [pose]) == [0]


def test_collapsed_head_points_do_not_count_as_independent_confirmation():
    face = checked_detection([20, 20, 60, 80, 50, 50, 50, 50, 50, 50, 40, 80, 60, 80, .99])
    pose = person()
    pose['keypoints'][:3] = [[50, 50]] * 3
    assert association_rejection(face, pose) == 'insufficient_head_landmarks'


@pytest.mark.parametrize('bounds', [[90, 0, 160, 140], [30, 44, 90, 70]])
def test_disjoint_or_small_overlap_is_not_rescued_even_by_matching_points(bounds):
    assert association_rejection(FACE, person(bounds)) == 'face_outside_person_box'


def test_multiple_eligible_poses_and_competing_faces_remain_unmatched():
    partial, contained = person(), person((0, 0, 100, 140))
    assert associate([FACE], [partial, contained]) == [None]
    assert associate([FACE, FACE], [partial]) == [None, None]
    assert associate([FACE], [contained, partial]) == [None]


def test_bad_partial_candidate_does_not_block_the_correct_person():
    bad, good = person(), person()
    bad['keypoints'][:3] = [[60, 70], [75, 55], [45, 55]]
    assert associate([FACE], [bad, good]) == [1]


def test_mixed_new_failures_are_reported_without_mislabeling_missing_points():
    few, wrong = person(), person()
    few['keypoints'][1:3] = [[-1., -1.]] * 2
    wrong['keypoints'][0][0] += 10
    details = explain_associations([FACE], [few, wrong], [None])[0]
    assert details['reason'] == 'mixed_association_failures'
    assert details['pair_rejections'] == [dict(person_index=0, reason='insufficient_head_landmarks'),
                                         dict(person_index=1, reason='head_landmark_mismatch')]


def test_rescued_face_uses_original_person_crop_and_omits_outside_landmarks():
    pose = person((40, 0, 100, 140))
    pose['keypoints'][2] = [-1., -1.]  # right eye lies outside the actual crop
    model = SapiensIDFaces.__new__(SapiensIDFaces)
    model.owner = get_ident()
    with Image.new('RGB', (160, 160)) as image:
        model._activate(image, [FACE], [pose])
        assert model.face_matches == [0]
        bounds, points = model.person_input(image, FACE)
    assert bounds == pose['bounds']
    assert points[17] == [-1., -1.]
    assert points[18:20] == [[25., 45.], [10., 60.]]


def test_joint_assignment_uses_landmark_evidence_and_abstains_on_tie():
    from types import SimpleNamespace
    faces = [SimpleNamespace(box=(x,20,60,80), landmarks=[
        (x+15,45), (x+45,45), (x+30,60)]) for x in (20,70)]
    first = person((0,0,155,150))
    first['keypoints'][:3] = [[50,60],[75,45],[35,45]]
    second = person((0,0,155,150))
    second['keypoints'][:3] = [[100,60],[115,45],[85,45]]
    assert associate(faces,[first,second]) == [0,1]
    assert associate(faces,[second,first]) == [1,0]
    assert associate([faces[0]], [first,deepcopy(first)]) == [None]


def test_contained_singleton_rejects_contradiction_and_accepts_profile():
    pose = person((0,0,100,140))
    pose['keypoints'][1] = [-1,-1]
    assert associate([FACE], [pose]) == [0]
    pose['keypoints'][0] = [70.,78.]
    assert association_rejection(FACE, pose) == 'head_landmark_mismatch'


def test_merged_box_retries_only_after_two_distinct_heads(monkeypatch):
    from types import SimpleNamespace
    from stash_ai.recognition.crowded_retry import retry_people
    import numpy as np
    calls = []
    class Tensor:
        def __init__(self, value): self.value = np.asarray(value)
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self.value
    heads = [FACE, checked_detection([v+80 if i in (0,4,6,8,10,12) else v for i,v in enumerate(FACE.row)])]
    poses = [person((0,0,100,140)), person((80,0,190,140))]
    poses[1]['keypoints'][:3] = [[130.,60.], [145.,45.], [115.,45.]]
    raw = SimpleNamespace(boxes=SimpleNamespace(xyxy=Tensor([[0,0,100,140],[80,0,190,140]]),conf=Tensor([.9,.9])))
    model = SimpleNamespace(config=SimpleNamespace(data={'sapiensid':{'pose':{'max_people_per_frame':4,
        'score_threshold':.5}}}), pose_refiner=SimpleNamespace(predict=lambda *a: ()), _people=lambda *a:poses,
        _predict_pose=lambda crop:calls.append(crop.size) or [raw])
    original = [dict(bounds=[0,0,190,140])]
    with Image.new('RGB', (200,150)) as image:
        assert retry_people(model,image,[FACE,FACE],original) is original and not calls
        assert retry_people(model,image,heads,original) == poses and len(calls)==1


def test_full_frame_merged_box_retries_distinct_tiles_containing_each_complete_face():
    from stash_ai.recognition.crowded_retry import _regions
    from types import SimpleNamespace
    with Image.new('RGB',(160,160)) as image:
        faces = [SimpleNamespace(box=(10,40,40,50)),SimpleNamespace(box=(100,40,36,50))]
        regions = _regions(image,faces,[(30,65),(118,65)],(0,0,160,160))
        assert len(regions)==2
        for face,(bounds,heads) in zip(faces,regions):
            x,y,w,h = face.box
            assert bounds[0]<=x<x+w<=bounds[2] and bounds[1]<=y<y+h<=bounds[3]
            assert len(heads)==1 and bounds[2]-bounds[0]<image.width
