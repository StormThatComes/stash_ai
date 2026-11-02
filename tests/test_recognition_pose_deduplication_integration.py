"""Shared single/batch inference, SAM ownership, audit indices and sweep exports."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from stash_ai.recognition import detection_audit, pmpose_pose, pose_deduplication_audit as audit
from stash_ai.recognition.config import defaults
from stash_ai.recognition.detection_masks import MaskOptions
from stash_ai.recognition.models import checked_detection
from stash_ai.recognition.pose_deduplication import DeduplicatedPMPose
from stash_ai.recognition.pose_deduplication_config import defaults as policy
from stash_ai.recognition.pose_refinement import RefinedPMPose
from stash_ai.recognition.pose_refinement_config import defaults as refinement_policy
from stash_ai.recognition.sapiensid_backend import SapiensIDFaces
from tests.test_recognition_detection_masks import RefinedDetector, render
from tests.test_recognition_detection_audit import setup
from tests.test_recognition_detection_sweep import table
from tests.test_recognition_pose_refinement import Pose, Segmenter, pose_points, prediction
from tests.test_recognition_pose_deduplication import BOXES


FACE = checked_detection([25, 10, 30, 25, 38, 18, 32, 18, 35, 20, 32, 28, 38, 28, .99])


class CandidatePose(Pose):
    def predict_many(self, frames):
        self.calls.append(('first', len(frames)))
        result = []
        for image, bounds in frames:
            values = prediction([pose_points() + ([80, 0] if box[0] >= 100 else [0, 0]) for box in bounds])
            if len(bounds):
                values[1][0] = .8
            result.append(values)
        return result


def model(masked=True, *, mode='all'):
    pose = CandidatePose()
    segmenter = Segmenter()
    settings = refinement_policy()
    settings['mode'] = mode
    return DeduplicatedPMPose(pose, dict(policy(), enabled=True), RefinedPMPose(pose, segmenter, settings) if masked else None)


@pytest.mark.parametrize('masked', [False, True])
def test_selection_precedes_sam_and_keeps_single_and_batched_results_identical(masked):
    estimator = model(masked)
    with Image.new('RGB', (160, 100)) as image, Image.new('RGB', (200, 120)) as other:
        single = estimator.predict(image, BOXES)
        batched = estimator.predict_many([(image, BOXES), (other, BOXES), (other, [])])
    for row in batched[:2]:
        for index in (0, 1):
            np.testing.assert_array_equal(row[index], single[index])
        for key in row[2]:
            np.testing.assert_array_equal(row[2][key], single[2][key])
        assert row[3:] == single[3:]
    assert batched[2][0].shape == (0, 17, 2)
    assert batched[2][4]['kept_indices'] == []
    assert single[4]['kept_indices'] == [1]
    assert 'audit' not in single[4]  # No detailed pair records in normal collection.
    if masked:
        assert single[3][0]['status'] == 'refined'  # Duplicate no longer blocks reliable prompts.
        assert estimator.refined.segmenter.frames == estimator.refined.segmenter.prompts == 3
        assert estimator.pose.calls == [('first', 1), ('masked', 1), ('first', 3), ('masked', 1), ('masked', 1)]
    else:
        assert single[3] is None
        assert estimator.pose.calls == [('first', 1), ('first', 3)]


def test_overlap_mode_checks_survivors_without_resurrecting_duplicates():
    estimator = model(mode='overlap')
    with Image.new('RGB', (160, 100)) as image:
        result = estimator.predict(image, BOXES)
    assert result[4]['kept_indices'] == [1]
    assert result[3][0]['reason'] == 'no_overlap'
    assert estimator.refined.segmenter.frames == 0


@pytest.mark.parametrize('deduplicated', [False, True])
@pytest.mark.parametrize('masked', [False, True])
def test_pipeline_assembly_only_loads_required_stages(monkeypatch, deduplicated, masked):
    from stash_ai.recognition import sam_pose2seg
    data = defaults()
    data['sapiensid']['pose']['backend'] = 'pmpose'
    data['sapiensid']['pose']['deduplication']['enabled'] = deduplicated
    data['sapiensid']['pose']['refinement']['enabled'] = masked
    calls = []
    monkeypatch.setattr(sam_pose2seg, 'load', lambda *_: calls.append('sam') or Segmenter())
    pose = CandidatePose()
    actual = pmpose_pose._pipeline(pose, SimpleNamespace(data=data), 'cpu')
    assert calls == (['sam'] if masked else [])
    if deduplicated:
        assert isinstance(actual, DeduplicatedPMPose) and actual.pose is pose
        assert (actual.refined is not None) == masked
    elif masked:
        assert isinstance(actual, RefinedPMPose) and actual.pose is pose
    else:
        assert actual is pose


class ArrayTensor:
    """Detector output protocol used by the real adapter, without torch inference."""
    def __init__(self, array): self.array = np.asarray(array)
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.array


@pytest.mark.parametrize('masked', [False, True])
def test_production_adapter_remaps_scores_faces_and_body_people_in_single_and_batch(monkeypatch, masked):
    from stash_ai.recognition import sapiensid_backend as backend, sapiensid_inference, sam_pose2seg
    data = defaults()
    data['models']['device'] = 'cpu'
    data['sapiensid']['enabled'] = data['sapiensid']['collection']['enabled'] = True
    settings = data['sapiensid']['pose']
    settings.update(backend='pmpose', detector='dfine_x')
    settings['deduplication']['enabled'] = True
    settings['refinement']['enabled'] = masked
    class Boxes:
        def predict(self, source, **kwargs):
            assert kwargs['iou'] == .7
            frames = source if isinstance(source, list) else [source]
            return [SimpleNamespace(boxes=SimpleNamespace(
                xyxy=ArrayTensor(BOXES), conf=ArrayTensor([.99, .8]))) for image in frames]
    monkeypatch.setattr(backend, 'load_pose', lambda _: Boxes())
    monkeypatch.setattr(sapiensid_inference, 'resolve_device', lambda _: 'cpu')
    monkeypatch.setattr(sam_pose2seg, 'load', lambda *_: Segmenter())
    monkeypatch.setattr(pmpose_pose, 'load', lambda cfg, device: pmpose_pose._pipeline(CandidatePose(), cfg, device))
    detector = backend.SapiensIDFaces(SimpleNamespace(data=data), {}, load_identity=False, load_detector=False)
    try:
        with Image.new('RGB', (160, 100)) as image, image.copy() as other:
            detector.detect(image, faces=[FACE])
            expected = deepcopy(detector.frame_people)
            assert len(expected) == 1 and detector.face_matches == [0]
            assert expected[0]['score'] == .8  # Preserve winning original detector score, not .99.
            assert expected[0]['pose_deduplication'] == dict(detector_index=1, suppressed_detector_indices=[0])
            assert detector._person(image, FACE) == expected[0]
            detector.detect_many([(image, [FACE]), (other, [])])
            for frame, faces in ((image, [FACE]), (other, [])):
                detector.detect(frame, faces=faces)
                assert detector.frame_people == expected
                assert detector.face_matches == ([0] if faces else [])
    finally:
        detector.close()
    assert detector.pose_refiner is None and not detector._prepared_frames


def audit_detector(enabled=True):
    detector = RefinedDetector(boxes=BOXES)
    detector.config.data['sapiensid']['pose']['deduplication']['enabled'] = enabled
    raw = CandidatePose()
    refined = RefinedPMPose(raw, Segmenter(), refinement_policy())
    detector.pose = DeduplicatedPMPose(raw, dict(policy(), enabled=True), refined) if enabled else refined
    return detector


@pytest.mark.parametrize('view', ['overlay', 'panels', 'off'])
def test_audit_retains_original_candidates_and_remaps_masks_to_survivors(view):
    detector = audit_detector()
    with render(detector, MaskOptions(view=view)) as (png, record):
        assert len(record['people']) == 1
        assert record['people'][0]['pose_deduplication']['detector_index'] == 1
        details = record['pose_deduplication']
        assert details['kept_indices'] == [1]
        assert details['detections'][0]['suppressed_by'] == 1
        assert details['detections'][1]['person_index'] == 0
        assert details['detections'][1]['detector_score'] == .98
        assert png.getpixel((90, 60)) == (180, 180, 180)  # Removed original D0 box.
        assert details['legend_offset'][1] >= 100
        if view != 'off':
            assert record['mask_visualization']['people'][0]['displayed_stage'] == 'cleaned'
        json.dumps(record, allow_nan=False)


def test_audit_context_does_not_retain_frames_or_cross_contaminate_other_images():
    estimator = model()
    with Image.new('RGB', (160, 100)) as image, image.copy() as other:
        with pytest.raises(RuntimeError, match='export failed'):
            with audit.capture(image) as reports:
                assert not audit.active(other)
                estimator.predict(image, BOXES)
                assert len(reports) == 1
                raise RuntimeError('export failed')
        assert reports == [] and not audit.active(image)
        assert 'audit' not in estimator.predict(image, BOXES)[4]


def test_multiple_survivors_keep_mask_indices_colors_and_original_candidate_mapping():
    detector = audit_detector()
    detector.boxes = [*BOXES, [100, 0, 140, 100]]
    with render(detector) as (png, record):
        assert [p['pose_deduplication']['detector_index'] for p in record['people']] == [1, 2]
        assert [p['person_index'] for p in record['mask_visualization']['people']] == [0, 1]
        assert all(p['pose_refinement']['status'] == 'refined' for p in record['people'])
        assert png.getpixel((147, 65)) == (30, 77, 60)  # Surviving P1 gets green, not original D2 blue.


def test_sweep_compares_enabled_against_disabled_without_changing_box_nms(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch)
    case.cfg.data['sapiensid']['pose']['backend'] = 'pmpose'
    case.cfg.data['sapiensid']['pose']['refinement']['enabled'] = True
    case.args.mask_view = 'panels'
    table(case, '- {parameter: sapiensid.pose.deduplication.enabled, value: true}')
    detectors = []
    def factory(config, paths):
        assert config.data['sapiensid']['pose']['nms_threshold'] == .7
        detector = audit_detector(config.data['sapiensid']['pose']['deduplication']['enabled'])
        detectors.append(detector)
        return detector
    case.services.assets = lambda *_, **kw: {}
    assert detection_audit.execute(case.args, case.cfg, services=case.services, detector_factory=factory) == 0
    manifest = json.loads((case.destination / 'manifest.json').read_text())
    frames = manifest['scene']['frames']
    assert len(frames[0]['people']) == 2 and 'pose_deduplication' not in frames[0]
    assert len(frames[1]['people']) == 1 and frames[1]['pose_deduplication']['kept_indices'] == [1]
    assert all(p['pose_refinement']['reason'] == 'no_reliable_prompts' for p in frames[0]['people'])
    assert frames[1]['people'][0]['pose_refinement']['status'] == 'refined'
    with Image.open(case.destination / manifest['scene']['file']) as png:
        sx, sy = frames[1]['source_offset']
        assert png.getpixel((sx+90, sy+60)) == (180, 180, 180)
        assert frames[1]['pose_deduplication']['legend_offset'][1] > 160
    assert all(d.closed and d.calls == d.releases == 1 for d in detectors)
    assert not case.stash.writes and len(list(case.destination.iterdir())) == 2
