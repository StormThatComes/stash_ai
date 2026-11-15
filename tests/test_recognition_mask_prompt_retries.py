"""SAM retry budgets, spatial evidence and ownership across inference paths."""
import numpy as np
from PIL import Image
import pytest

from stash_ai.recognition.pose_mask_capture import capture_masks
from stash_ai.recognition.pose_refinement import RefinedPMPose
from stash_ai.recognition.pose_refinement_config import defaults
from tests.test_recognition_pose_refinement import BOX, Segmenter, pose_points
from tests.test_recognition_profile3_batching import BatchedPose, BatchedSegmenter


def driver(batched, **settings):
    return RefinedPMPose(BatchedPose(), BatchedSegmenter() if batched else Segmenter(),
                         dict(defaults(), min_prompt_points=1, **settings))


@pytest.mark.parametrize('batched', [True, False])
def test_only_the_rejected_person_adds_a_spatially_distinct_point(batched):
    model = driver(batched, max_prompt_points=4)
    calls = []
    def segment(positive, negative=()):
        first = positive[0, 0] < 100
        calls.append((first, len(positive)))
        mask = np.zeros((100, 200), bool)
        left = 20 if first else 120
        mask[10:90, left:left+40] = True
        return mask, .1 if first and len(positive) == 1 else .95
    model.segmenter.predict = segment
    with Image.new('RGB', (200, 100)) as image:
        records = model.predict(image, [BOX, [120, 10, 160, 90]])[3]
    assert calls == [(True, 1), (False, 1), (True, 2)]
    first, other = records
    assert first['status'] == other['status'] == 'refined'
    assert first['prompt_indices'][0] == 0 and first['prompt_indices'][1] in (11, 12)
    assert [row['mask_attempts'] for row in records] == [2, 1]


@pytest.mark.parametrize('batched', [True, False])
@pytest.mark.parametrize('resolved', [True, False])
def test_conflict_retries_only_until_resolved_or_exhausted(batched, resolved):
    model = driver(batched, max_prompt_points=2)
    boxes = [[20,10,80,90], [64,10,125,90]]
    calls = []
    def segment(positive, negative=()):
        calls.append((len(positive),len(negative)))
        left = 20 if positive[0,0] < 65 else 67
        mask = np.zeros((100, 150), bool)
        mask[10:90, left:left+40] = True
        if len(positive) == 1 or not resolved:
            mask[65:90, 35:97] = True
        return mask, .99
    model.segmenter.predict = segment
    with Image.new('RGB', (150,100)) as image:
        joints, confidence, probabilities, records = model.predict(image, boxes)
    assert calls == [(1,1), (1,1), (2,1), (2,1)]
    assert all(r['mask_attempts'] == 2 for r in records)
    if resolved:
        assert all(r['status'] == 'refined' and 'overlap_mask' in r for r in records)
    else:
        for i, record in enumerate(records):
            assert record['mask_conflict'] and record['reason'] == 'shared_mask_pixels'
            assert record['status'] == 'rejected' and record['bounds'] == boxes[i]
            assert not {'encoder_mask','overlap_mask'}.intersection(record)
            np.testing.assert_array_equal(joints[i], pose_points() + [boxes[i][0]-20, 0])
        assert np.allclose(confidence, .95) and np.allclose(probabilities['visibility'], .95)
        assert model.pose.calls == [('first', 1)]


def test_final_rejection_replaces_earlier_audit_stages_and_obeys_maximum():
    model = driver(False, max_prompt_points=2)
    def segment(positive, negative=()):
        mask = np.zeros((100,160), bool)
        if len(positive) == 1:
            mask[85:90,30:40] = True  # Cleaned, but lacks its positive prompt.
        return mask, .95 if len(positive) == 1 else .1
    model.segmenter.predict = segment
    with Image.new('RGB', (160,100)) as image, capture_masks(image) as capture:
        record = model.predict(image, [BOX])[3][0]
        assert record['reason'] == 'low_mask_score' and record['mask_attempts'] == 2
        assert set(capture.stages(record)) == {'raw'}
        assert not capture.mask(record, 'raw').any() and capture.bytes == 2000


def test_each_neighbour_has_an_independent_exclusion_budget():
    model = driver(False, max_prompt_points=1)
    boxes = [[0,0,140,100]] * 3
    points = np.array([pose_points() + [dx,0] for dx in (0,40,80)])
    cores = [np.array([0,1,5,11])] * 3
    for limit in (0,1,2):
        model.settings['max_negative_prompt_points_per_neighbor'] = limit
        _, peers, negatives = model._prompts(boxes, points, cores, 0, [0])
        assert peers == [1,2] and len(negatives) == 2 * limit
        if limit:
            np.testing.assert_array_equal(negatives[::limit], points[1:,0])


def test_retry_preserves_both_initial_visibility_ranked_seeds():
    model = driver(False, max_prompt_points=3)
    model.settings['min_prompt_points'] = 2
    sent = []
    def segment(positive, negative=()):
        sent.append(positive.copy())
        return np.full((100,160), True), .1
    model.segmenter.predict = segment
    with Image.new('RGB',(160,100)) as image:
        record = model.predict(image, [BOX])[3][0]
    np.testing.assert_array_equal(sent[0], pose_points()[:2])
    np.testing.assert_array_equal(sent[1][:2], sent[0])
    assert record['prompt_indices'][2] in (11,12)
    assert record['mask_attempts'] == 2
