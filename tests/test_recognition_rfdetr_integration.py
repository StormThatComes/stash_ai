"""RF-DETR dispatch through real pose/face adapters and audit rendering."""
import copy
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from stash_ai.recognition import rfdetr_detector as rf, sapiensid_backend as backend
from stash_ai.recognition import sapiensid_assets, sapiens_pose, vitpose_pose, pmpose_pose
from tests.test_recognition_rfdetr import configuration


@pytest.mark.parametrize('selected', ['sapiens', 'vitpose_plus_large', 'pmpose'])
def test_rfdetr_replaces_box_model_in_single_and_batched_pose_pipeline(tmp_path, monkeypatch, selected):
    torch = pytest.importorskip('torch')
    config = configuration(tmp_path, backend=selected)
    config.data['sapiensid']['collection']['enabled'] = True
    monkeypatch.setattr(sapiensid_assets, 'pose_asset', lambda _: pytest.fail('YOLO asset requested'))
    def model(pixels):
        logits = torch.full((len(pixels), 1, 91), -20.); logits[..., 1] = 10.
        return dict(pred_logits=logits, pred_boxes=torch.tensor([[[.5, .5, .8, .8]]]).repeat(len(pixels), 1, 1))
    monkeypatch.setattr(rf, 'load', lambda _: rf.RFDETRDetector(model, device='cpu', batch_size=2))
    seen = []
    class Refiner:
        def predict(self, image, bounds):
            seen.append(bounds)
            points = np.full((len(bounds), 17, 2), 30., dtype=np.float32)
            scores = np.ones((len(bounds), 17), dtype=np.float32)
            return ((points, scores, dict(presence=scores, visibility=scores))
                    if selected == 'pmpose' else (points, scores))
        def predict_many(self, frames):
            return [self.predict(image, bounds) for image, bounds in frames]
    chosen = {'sapiens': sapiens_pose, 'vitpose_plus_large': vitpose_pose, 'pmpose': pmpose_pose}[selected]
    monkeypatch.setattr(chosen, 'load', lambda *args: Refiner())
    for other in (sapiens_pose, vitpose_pose, pmpose_pose):
        if other is not chosen:
            monkeypatch.setattr(other, 'load', lambda *args: pytest.fail('unselected pose loaded'))
    detector = backend.SapiensIDFaces(config, {}, load_identity=False, load_detector=False)
    try:
        with Image.new('RGB', (100, 200)) as image:
            detector.detect(image, faces=[], force_pose=True)
            expected = copy.deepcopy(detector.frame_people)
            detector.detect_many([(image, [])])
            detector.detect(image, faces=[])
            assert detector.frame_people == expected and len(expected) == 1
            assert len(detector.person_geometry(image, expected[0])[1]) == 22
        assert len(seen) == 2
        assert detector.provenance['recognizer']['person_detector']['detector'] == 'rfdetr_xl'
        assert 'pose_sha256' not in detector.provenance['recognizer']
    finally:
        detector.close()
    assert detector.pose is None and detector.pose_refiner is None


def test_rfdetr_pmpose_dedup_sam_and_audit_share_survivor_indices(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    from stash_ai.recognition import sam_pose2seg, detection_audit
    from stash_ai.recognition.models import YuNetDetector
    from tests.test_recognition_pose_deduplication_integration import CandidatePose, FACE, BOXES
    from tests.test_recognition_pose_refinement import Segmenter
    config = configuration(tmp_path, backend='pmpose')
    settings = config.data['sapiensid']['pose']
    settings['deduplication']['enabled'] = settings['refinement']['enabled'] = True
    settings['refinement']['mode'] = 'all'
    def model(pixels):
        xyxy = torch.tensor(BOXES, dtype=torch.float32) / torch.tensor([160, 100, 160, 100])
        boxes = torch.cat(((xyxy[:, :2]+xyxy[:, 2:])/2, xyxy[:, 2:]-xyxy[:, :2]), dim=1)
        logits = torch.full((len(pixels), 2, 91), -20.)
        logits[..., 1] = torch.logit(torch.tensor([.99, .8]))
        return dict(pred_logits=logits, pred_boxes=boxes.unsqueeze(0).repeat(len(pixels), 1, 1))
    monkeypatch.setattr(rf, 'load', lambda _: rf.RFDETRDetector(model, device='cpu', batch_size=2))
    monkeypatch.setattr(pmpose_pose, 'load', lambda cfg, device: pmpose_pose._pipeline(CandidatePose(), cfg, device))
    monkeypatch.setattr(sam_pose2seg, 'load', lambda *args: Segmenter())
    monkeypatch.setattr(YuNetDetector, 'detect', lambda *args, **kwargs: [FACE])
    detector = backend.SapiensIDFaces(config, {}, load_identity=False, load_detector=False)
    detector.has_detector = True  # Face inference above is stubbed; retain the real pose/association path.
    try:
        with Image.new('RGB', (160, 100)) as image:
            cue = SimpleNamespace(start=0, end=1, rect=[0, 0, 160, 100])
            with detection_audit.render_frame(image, detector, '1', 0, cue) as (png, record):
                assert len(record['people']) == 1
                assert detector.face_matches == [0]
                assert record['pose_deduplication']['kept_indices'] == [1]
                assert record['people'][0]['pose_refinement']['status'] == 'refined'
                assert record['mask_visualization']['people'][0]['person_index'] == 0
                json.dumps(record, allow_nan=False)
    finally:
        detector.close()
