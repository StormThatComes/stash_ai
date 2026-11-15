import numpy as np
from tests.test_recognition_pose_refinement import BOX, pose_points, prediction, wrapper
from stash_ai.recognition.limb_ownership import fuse


def test_lost_knee_recovers_its_own_first_pass_score_and_probabilities():
    raw = prediction([pose_points()])
    refined = prediction([pose_points()])
    refined[1][0,13] = .01
    refined[2]['presence'][0,13] = .02
    mask = np.ones((100,160), bool)
    points, scores, probs, restored = fuse(0,0,raw,refined,mask,BOX,wrapper().settings)
    assert restored == [13]
    assert scores[13] == raw[1][0,13] and probs['presence'][13] == raw[2]['presence'][0,13]


def test_missing_elbow_requires_independent_wrist_and_rejects_partner():
    raw = prediction([pose_points(), pose_points() + [80,0]])
    refined = prediction([pose_points()])
    for values in (raw, refined):
        values[1][0,7] = 0
    mask = np.ones((100,160),bool)
    assert fuse(0,0,raw,refined,mask,BOX,wrapper().settings)[1][9] > 0
    raw[0][1,10] = raw[0][0,9]
    assert fuse(0,0,raw,refined,mask,BOX,wrapper().settings)[1][9] == 0


def test_invisible_seed_cannot_confirm_a_masked_pose():
    model = wrapper()
    raw = prediction([pose_points()])
    refined = prediction([pose_points()])
    refined[2]['visibility'][:,:3] = .01
    result = prediction([pose_points()])
    records = [dict(detector_bounds=BOX, separated=False)]
    model._apply_second_pass([(0, BOX, np.ones((100,160),bool), [0,1])],result,records,raw,refined)
    assert records[0]['reason'] == 'second_pass_core_drift'


def test_reliable_output_joint_can_be_below_the_sam_seed_floor():
    from stash_ai.recognition.pose_refinement import RefinedPMPose
    from stash_ai.recognition.pose_refinement_config import defaults
    model = RefinedPMPose(None,None,dict(defaults(), min_keypoint_score=.85),
                          joint_quality={'min_keypoint_score':.5, 'min_presence':.5})
    raw = prediction([pose_points()])
    refined = prediction([pose_points()])
    refined[1][0,9] = .70
    result = prediction([pose_points()])
    records = [dict(detector_bounds=BOX, separated=False)]
    model._apply_second_pass([(0,BOX,np.ones((100,160),bool),[0,1])],result,records,raw,refined)
    assert result[1][0,9] == refined[1][0,9]
