"""Regression cases for regional SAM prompts and measured adjacent-cue hints."""
import numpy as np
import pytest
from PIL import Image

from stash_ai.recognition.inspection_diagnostics import pose_landmarks
from stash_ai.recognition.pose_refinement import RefinedPMPose
from stash_ai.recognition.pose_refinement_config import defaults
from stash_ai.recognition.temporal_prompts import SUPPORT
from tests.test_recognition_pose_refinement import BOX, Pose, Segmenter, pose_points, prediction
from tests.test_recognition_profile3_batching import BatchedPose, BatchedSegmenter


def model(batched=False, **options):
    settings = dict(defaults(), recover_prompt_points=True, adaptive_decoder_retries=True)
    settings.update(options)
    return RefinedPMPose(BatchedPose() if batched else Pose(),
                         BatchedSegmenter() if batched else Segmenter(), settings)


@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('repair', [False, True])
def test_partial_mask_retries_hip_and_preserves_the_first_pose_on_exhaustion(batched, repair):
    pipeline = model(batched, min_prompt_coverage=0.5)
    prompts = []
    def segment(positive, negative=()):
        prompts.append(positive.copy())
        mask = np.zeros((100, 100), bool)
        mask[10:90 if repair and len(prompts) > 1 else 45, 20:60] = True
        return mask, .99
    pipeline.segmenter.predict = segment
    with Image.new('RGB', (100, 100)) as image:
        points, _, _, (record,) = pipeline.predict(image, [BOX])
    hips = pose_points()[[11, 12]].tolist()
    assert prompts[0][1].tolist() in hips
    assert prompts[1][0].tolist() in hips and not np.array_equal(prompts[1][0], prompts[0][1])
    assert 2 <= len(prompts) <= 3
    np.testing.assert_array_equal(points[0, [11,12]], hips)
    if repair:
        assert record['status'] == 'refined' and record['overlap_mask']
    else:
        assert record['reason'] == 'missing_hip_region' and 'overlap_mask' not in record
        assert all(kind == 'first' for kind, _ in pipeline.pose.calls)


def plan(pipeline, cue, pose, *, shot=0):
    with Image.new('RGB', (100,100)) as image:
        image._recognition_cue_signature = (cue, shot, '22' * 48)
        image._recognition_scan = 'scan'
        return pipeline._plan([BOX], pose, image)


def test_enough_face_points_do_not_block_temporal_hip_but_hints_never_chain():
    pipeline = model()
    first = prediction([pose_points()])
    plan(pipeline, 3, first)
    missing = prediction([pose_points()])
    missing[1][0, 11:13] = 0
    result, (record,), _, jobs = plan(pipeline, 4, missing)
    assert any(k in (11, 12) for k in jobs[0][1][:2])
    assert record['temporal_source_cue'] == 3
    assert record['temporal_prompts'] == [(11, pose_points()[11].tolist())]
    assert not result[1][0, 11:13].any()
    assert 'temporal_prompts' not in plan(pipeline, 5, missing)[1][0]
    plan(pipeline, 6, first)
    assert 'temporal_prompts' not in plan(pipeline, 7, missing, shot=1)[1][0]


@pytest.mark.parametrize('joint', list(SUPPORT))
def test_mobile_hints_use_current_local_prediction_and_require_anatomical_support(joint):
    pipeline = model(min_prompt_points=3)
    original = prediction([pose_points()])
    def candidate(current):
        plan(pipeline, 4, original)
        return plan(pipeline, 5, current)[1][0]
    weak = prediction([pose_points()])
    weak[1][0,joint] = .4
    weak[0][0,joint] += [1,1]
    record = candidate(weak)
    assert record['temporal_prompts'] == [(joint, weak[0][0,joint].tolist())]
    assert record['point_exclusions']['score'] & (1 << joint)
    weak[0][0,joint] += [16,0]
    assert 'temporal_prompts' not in candidate(weak)
    weak[0][0,joint] -= [16,0]
    weak[2]['presence'][0,SUPPORT[joint][0]] = 0.
    assert joint not in dict(candidate(weak).get('temporal_prompts', ()))


def test_unconfirmed_temporal_point_vetoes_otherwise_successful_second_pass():
    pipeline = model(min_prompt_points=2)
    raw = prediction([pose_points()])
    plan(pipeline, 3, raw)
    missing = prediction([pose_points()])
    missing[1][0,11:13] = 0
    result, records, _, _ = plan(pipeline, 4, missing)
    ids = np.array([0,5,11])
    records[0]['prompt_points'] = pose_points()[ids].tolist()
    mask = np.ones((100,100), bool)
    pipeline._apply_second_pass([(0,BOX,mask,ids)],result,records,missing,missing)
    assert records[0]['reason'] == 'second_pass_core_drift'
    assert 'overlap_mask' not in records[0]
    assert result[1][0,11] == 0


def test_inspector_explains_final_missing_point_and_reads_older_observations():
    pipeline = model()
    original = prediction([pose_points()])
    plan(pipeline, 1, original)
    weak = prediction([pose_points()])
    weak[1][0,7] = .4
    record = plan(pipeline, 2, weak)[1][0]
    final = pose_points()
    final[7] = (-1,-1)
    body = {'pose_refinement': record, 'keypoints': final.tolist()}
    geometry = {'shapes': [{'kind':'body', 'selected':True, 'geometry':body}]}
    rows = pose_landmarks(geometry)
    assert len(rows) == 17
    assert rows[7] == ('left_elbow', '(25, 48)', 'score', 'Candidate from cue 1 (25, 48)', 'Missing')
    body['pose_refinement'] = {}
    assert pose_landmarks(geometry) == []


def test_competing_person_at_the_limbs_parent_blocks_temporal_endpoint():
    from stash_ai.recognition.temporal_prompts import borrow
    xy = pose_points()
    old_joints = dict(enumerate(xy))
    current = dict(old_joints)
    del current[9]
    peer = ([23,46,28,51], {})  # Other box contains our measured elbow.
    def state(cue, joints, competing):
        people = [(BOX, joints)]
        if competing:
            people.append(peer)
        return (('same', cue, 3, '44'*48), people)
    for past_peer, current_peer in ((True, False), (False, True)):
        record = {}
        borrow(state(0, old_joints, past_peer), state(1, current, current_peer), 0,
               np.array([0,5,6,11,12]), 2, record, local={9: xy[9]})
        assert not record


def test_diagnostics_preserve_unused_local_points_when_sam_skips_an_isolated_person():
    pipeline = model(mode='overlap')
    record = plan(pipeline, 0, prediction([pose_points()]))[1][0]
    assert record['reason'] == 'no_overlap'
    assert record['first_pass_pixels'] == pose_points().tolist()
    assert 'prompt_indices' not in record
