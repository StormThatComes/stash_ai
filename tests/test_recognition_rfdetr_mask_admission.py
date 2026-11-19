"""RF-DETR proposals obey pose admission before truncation and survive replay."""
from contextlib import contextmanager
import json
import sqlite3
from types import SimpleNamespace
import zlib

import numpy as np
from PIL import Image
import pytest

from stash_ai.recognition.config import defaults as config_defaults
from stash_ai.recognition.pose_mask_capture import capture_masks
from stash_ai.recognition.pose_masks import PoseMasks
from stash_ai.recognition.pose_refinement import RefinedPMPose, retain_quality_mask
from stash_ai.recognition.pose_refinement_config import defaults
from stash_ai.recognition.rfdetr_masks import RFDETRMasks
from tests.test_recognition_pose_refinement import BOX, pose_points
from tests.test_recognition_profile3_batching import BatchedPose


class Instances(RFDETRMasks):
    """Use measured instance proposals without running the detector network."""
    def __init__(self, proposals, settings):
        super().__init__(None, 'cpu', 'fp32', settings['min_mask_score'], settings=settings)
        self.proposals = proposals
        self.frames = 0

    @contextmanager
    def frame(self, image):
        self.frames += 1
        self.size = image.size
        self.instance_ids = [p[0] for p in self.proposals]
        self.masks = [np.packbits(p[1]) for p in self.proposals]
        self.scores = np.array([p[2] for p in self.proposals])
        try:
            yield self
        finally:
            self.clear()


def mask(left=20, right=60):
    result = np.zeros((100, 160), bool)
    result[10:90, left:right] = True
    return result


def run(proposals, settings, image, boxes, batched, *, second_pass=False):
    segmenter = Instances(proposals, settings)
    pose = BatchedPose()
    wrapper = RefinedPMPose if second_pass else PoseMasks
    owner = wrapper(pose, segmenter, settings, queue_bytes=134217728 if batched else 1)
    # Initial RF proposals must not be selected without the shared pose context.
    segmenter.predict = lambda *a, **kw: pytest.fail('Premature prompt-only selection')
    result = owner.predict_many([(image, boxes)])[0]
    assert segmenter.frames == 1 and segmenter.masks is None
    return result, pose


@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('coverage,reason', [(.6, None), (1., 'missing_prompt_support')])
def test_partial_prompt_coverage_uses_yaml_and_reports_real_score(batched, coverage, reason):
    pixels = mask(); pixels[18, 38] = False  # Two of the first three prompts remain.
    settings = dict(defaults(), backend='rfdetr_seg_large', min_prompt_points=3,
                    min_prompt_coverage=coverage)
    with Image.new('RGB', (160, 100)) as image:
        result, _ = run([(17, pixels, .95)], settings, image, [BOX], batched)
    record = result[3][0]
    assert record['reason'] == reason and record['mask_score'] == .95
    assert record['mask_instance'] == 17
    assert ('overlap_mask' in record) is (reason is None)


@pytest.mark.parametrize('batched', [False, True])
def test_foreign_speckle_is_cleaned_before_negative_prompt_rejection(batched):
    pixels = mask(); pixels[20, 85] = True  # The neighbour's negative nose prompt.
    settings = dict(defaults(), backend='rfdetr_seg_large')
    with Image.new('RGB', (160, 100)) as image, capture_masks(image) as capture:
        result, _ = run([(17, pixels, .95), (42, mask(70,110), .94)], settings,
                        image, [[20,10,90,90], [70,10,140,90]], batched, second_pass=True)
        a, b = result[3]
        assert a['status'] == b['status'] == 'refined'
        assert capture.mask(a, 'raw')[20, 85]
        assert not capture.mask(a, 'cleaned')[20, 85]
        assert a['mask_pixels'] == 3200 and a['mask_instance'] == 17


@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('adaptive', [False, True])
def test_fourth_instance_survives_earlier_expansion_failures(batched, adaptive):
    proposals = []
    for instance, stop, score in ((1,145,.99), (2,150,.98), (3,160,.97)):
        pixels = mask(); pixels[80:100,20:stop] = True
        proposals.append((instance, pixels, score))
    proposals.extend([(88, mask(), .96), (42, mask(70,110), .95)])
    settings = dict(defaults(), backend='rfdetr_seg_large', adaptive_decoder_retries=adaptive)
    with Image.new('RGB', (160, 100)) as image:
        result, pose = run(proposals, settings, image, [[20,10,90,90], [70,10,140,90]],
                           batched, second_pass=True)
    a, b = result[3]
    assert a['status'] == b['status'] == 'refined'
    assert a['mask_instance'] == 88 and a['mask_score'] == .96
    assert a['mask_attempts'] == 1 and a['mask_candidates_evaluated'] == 5
    assert a['mask_pixels'] == 3200
    assert any(call[0] == 'masked_batch' for call in pose.calls)


@pytest.mark.parametrize('batched', [False, True])
def test_connected_foreign_prompt_still_rejects_after_cleanup(batched):
    settings = dict(defaults(), backend='rfdetr_seg_large', max_foreign_coverage=1.,
                    max_side_expansion=2.)
    with Image.new('RGB', (160, 100)) as image:
        result, _ = run([(17, mask(20,110), .97)], settings, image,
                        [[20,10,90,90], [70,10,140,90]], batched)
    for record in result[3]:
        assert record['reason'] == 'negative_prompt_in_mask'
        assert record['mask_score'] == .97 and 'overlap_mask' not in record


@pytest.mark.parametrize('batched', [False, True])
def test_no_instances_has_its_own_rejection_reason(batched):
    with Image.new('RGB', (160, 100)) as image:
        result, _ = run([], dict(defaults(), backend='rfdetr_seg_large'), image, [BOX], batched)
    record = result[3][0]
    assert record['reason'] == 'no_person_instances'
    assert record['mask_candidates_evaluated'] == 0 and 'mask_instance' not in record


@pytest.mark.parametrize('batched', [False, True])
def test_joint_choice_keeps_the_selected_instance_id_and_audit(batched):
    shared = mask(); shared[65:90,35:97] = True
    settings = dict(defaults(), backend='rfdetr_seg_large')
    with Image.new('RGB', (160, 100)) as image, capture_masks(image) as capture:
        result, _ = run([(11, shared, .99), (88, mask(), .96), (42, mask(70,110), .95)],
                        settings, image, [[20,10,90,90], [70,10,140,90]], batched)
        a, b = result[3]
        assert a['status'] == b['status'] == 'masked'
        assert a['mask_instance'] == 88 and a['mask_variant'] == 1
        assert a['mask_score'] == .96 and a['mask_pixels'] == 3200
        history = capture.history(a)[0]
        assert history['paired_choice'] == 1 and history['variants'][1]['instance'] == 88


def test_standalone_selection_uses_coverage_and_never_unions_people():
    settings = dict(defaults(), min_prompt_coverage=.6)
    proposals = [(17, mask(), .95), (42, mask(70,110), .94)]
    segmenter = Instances(proposals, settings)
    with Image.new('RGB', (160,100)) as image, segmenter.frame(image):
        chosen, score = segmenter.predict([[35,20],[32,18],[85,20]])
        np.testing.assert_array_equal(chosen, mask())
        assert score == .95
        # Half the prompts on each person does not satisfy the configured .6.
        assert not segmenter.predict([[35,20],[85,20]])[0].any()


def test_encoder_preserves_query_id_when_confidence_sorts_proposals():
    torch = pytest.importorskip('torch')
    class Model:
        def __call__(self, pixels):
            assert torch.is_inference_mode_enabled() and pixels.shape == (1,3,504,504)
            logits = torch.full((1,3,91), -10.)
            logits[0,:,1] = torch.tensor([2., -3., 4.])
            masks = torch.tensor(np.stack([mask(), mask(110,150), mask(70,110)]))[None]
            return dict(pred_logits=logits, pred_masks=torch.where(masks, 5., -5.))
    segmenter = RFDETRMasks(Model(), 'cpu', 'fp32', .5)
    with Image.new('RGB', (160,100)) as image, segmenter.frame(image):
        assert segmenter.instance_ids == [2,0]
        restored, score = segmenter.predict_instance(0)
        np.testing.assert_array_equal(restored, mask())
        assert score == pytest.approx(torch.tensor(2.).sigmoid().item())
        assert segmenter.predict_instance(1) is None
    assert segmenter.masks is segmenter.instance_ids is segmenter.scores is None


@pytest.mark.parametrize('missing', [False, True])
def test_retained_mask_replay_uses_query_id_not_rank_or_prompt_variant(missing):
    from stash_ai.recognition.body_collection import quality
    from stash_ai.recognition.body_mask_backfill import _restore_cue
    data = config_defaults()
    settings = data['sapiensid']['pose']['refinement']
    settings.update(enabled=True, backend='rfdetr_seg_large')
    pixels = mask()
    details = dict(status='masked', reason=None, bounds=BOX, detector_bounds=BOX,
                   prompt_points=[[35,20],[32,18]], mask_pixels=3200,
                   mask_instance=88, mask_variant=1, retry_mask_choices=3)
    retain_quality_mask(details, pixels, BOX)
    person = dict(bounds=BOX, keypoints=pose_points().tolist(), score=.95, pose_refinement=details)
    proposals = [(42, mask(70,110), .99)] + ([] if missing else [(88, pixels, .95)])
    segmenter = Instances(proposals, settings)
    segmenter.predict = segmenter.predict_options = segmenter.predict_frames = (
        lambda *a, **kw: pytest.fail('Instance replay must not reselect from prompts'))
    collector = SimpleNamespace(config=SimpleNamespace(data=data), model=SimpleNamespace(
        pose_refiner=SimpleNamespace(mask_frame=segmenter.frame)))
    rng = np.random.default_rng(12)
    with Image.fromarray(rng.integers(0,256,(100,160,3),dtype=np.uint8)) as image:
        measured, _ = quality(image, person, (), data['sapiensid']['collection'])
        details.pop('overlap_mask')
        db = sqlite3.connect(':memory:'); db.row_factory = sqlite3.Row
        try:
            db.executescript('CREATE TABLE body_observations(id, geometry, quality); '
                             'CREATE TABLE body_quality_masks(observation, mask);')
            db.execute('INSERT INTO body_observations VALUES (?,?,?)',
                       ('body', json.dumps(person), json.dumps(dict(quality=measured))))
            db.commit()
            rows = db.execute('SELECT * FROM body_observations').fetchall()
            _restore_cue(collector, db, image, rows, 'attempt')
            stored = db.execute('SELECT mask FROM body_quality_masks').fetchone()
            metadata = json.loads(db.execute('SELECT quality FROM body_observations').fetchone()[0])
            if missing:
                assert stored is None and metadata['mask_replay'] == 'measurement_mismatch'
            else:
                assert zlib.decompress(stored[0]) == np.packbits(pixels[10:90,20:60]).tobytes()
                assert metadata['mask_replay'] == 'verified'
        finally:
            db.close()


def test_only_rf_mask_contract_changes():
    from stash_ai.recognition.mask_backends import model_contract
    assert model_contract('rfdetr_seg_large')['adapter'] == 2
    assert model_contract('hq_sam2_large')['adapter'] == model_contract('sam3')['adapter'] == 1
