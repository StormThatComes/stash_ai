import numpy as np
from PIL import Image
from stash_ai.recognition.models import checked_detection
from stash_ai.recognition.prompt_recovery import face_inputs
from tests.test_recognition_pose_refinement import BOX, prediction, pose_points, wrapper


def plan(valid, *, with_face=False, other=False):
    model = wrapper()
    model.settings['recover_prompt_points'] = True
    boxes = [BOX] * (2 if other else 1)
    pred = prediction([pose_points()] * len(boxes))
    pred[1][:] = 0
    pred[1][:, valid] = .95
    face = checked_detection([25, 10, 24, 18, 38, 18, 32, 18, 35, 20, 33, 24, 38, 24, .98])
    with Image.new('RGB', (160, 100)) as image, face_inputs([(image, [face] if with_face else [])]):
        return model._plan(boxes, pred, image)


def test_single_clean_seed_and_unowned_overlapping_seed():
    _, records, _, jobs = plan([0])
    assert records[0]['prompt_minimum'] == 1 and jobs[0][1].tolist() == [0]
    assert records[0]['prompt_rejections']['score'] == 8
    assert not plan([0], other=True)[3]


def test_supported_elbow_and_uniquely_matched_face_landmark():
    assert 7 in plan([0, 7, 9])[3][0][1]
    _, records, _, jobs = plan([1], with_face=True)
    assert records[0]['face_prompt'] == [0, [35., 20.]]
    assert set(jobs[0][1]) == {0, 1}
    assert not plan([1], with_face=True, other=True)[3]
