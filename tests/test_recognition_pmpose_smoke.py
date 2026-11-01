"""Opt-in real PMPose-B CPU inference against upstream reference predictions."""
import os

import pytest


@pytest.mark.skipif(os.getenv('STASH_AI_PMPOSE_SMOKE') != '1', reason='explicit 618 MB PMPose smoke')
def test_pretrained_pmpose_reference_and_audit(tmp_path, monkeypatch, request):
    import hashlib
    from io import BytesIO
    import json
    from pathlib import Path
    from types import SimpleNamespace
    import numpy as np
    import requests
    import torch
    import yaml
    from PIL import Image
    from stash_ai.recognition.config import load
    from stash_ai.recognition.models import checked_detection
    from stash_ai.recognition import sapiensid_backend as backend, pmpose_registry
    from stash_ai.recognition.person_collection import audit_input

    previous_threads = torch.get_num_threads()
    request.addfinalizer(lambda: torch.set_num_threads(previous_threads))
    torch.set_num_threads(2)
    reference = json.loads((Path(__file__).parent / 'fixtures/pmpose_b_reference.json').read_text())
    assert reference['provenance']['checkpoint_sha256'] == pmpose_registry.SHA256
    assert reference['provenance']['source_revision'] == pmpose_registry.SOURCE_REVISION
    url = ('https://raw.githubusercontent.com/matplotlib/matplotlib/v3.10.1/'
           'lib/matplotlib/mpl-data/sample_data/grace_hopper.jpg')
    with requests.get(url, timeout=30) as response:
        response.raise_for_status()
        raw = response.content
    assert hashlib.sha256(raw).hexdigest() == reference['provenance']['image_sha256']
    path = tmp_path / 'recognition.yml'
    path.write_text(yaml.safe_dump(dict(version=1, models=dict(device='cpu'), sapiensid=dict(
        enabled=True, pose=dict(backend='pmpose',
            pmpose_directory=os.getenv('STASH_AI_PMPOSE_DIRECTORY', ''))))))
    config = load(path)
    # Fixed boxes isolate the pose model; detector dispatch has separate tests.
    bounds = [reference['provenance']['bounds']]
    boxes = SimpleNamespace(xyxy=torch.tensor(bounds), conf=torch.tensor([.99]))
    monkeypatch.setattr(backend, 'load_pose', lambda _: SimpleNamespace(
        predict=lambda **kwargs: [SimpleNamespace(boxes=boxes)]))
    detector = backend.SapiensIDFaces(config, {}, load_identity=False, load_detector=False)
    face = checked_detection([182.86642, 120.47789, 170.25731, 212.48526,
        227.11728, 193.32774, 307.84845, 195.3718, 266.4793, 240.008,
        228.50974, 267.86563, 302.18546, 269.56985, .94316435])
    try:
        with Image.open(BytesIO(raw)).convert('RGB') as image:
            points, scores, probabilities = detector.pose_refiner.predict(image, bounds)
            np.testing.assert_allclose(points[0], reference['points'], atol=.02, rtol=0)
            np.testing.assert_allclose(scores[0], reference['oks'], atol=2e-5, rtol=0)
            for key in ('presence', 'visibility'):
                np.testing.assert_allclose(probabilities[key][0], reference[key], atol=2e-5, rtol=0)
            detector.detect(image, faces=[face])
            report = audit_input(detector, image, face, tmp_path / 'audit')
            assert report['identity_rejection'] is None
            assert 'pmpose-b' in report['identity_input']
            person = report['person']
            assert person['keypoint_score_kind'] == 'expected_oks'
            assert len(person['keypoint_presence']) == len(person['keypoint_visibility']) == 17
            assert all(x >= 0 and y >= 0 for x, y in person['keypoints'][:7])
            assert person['keypoints'][7:] == [[-1., -1.]] * 10
            _, normalized, mask = detector.audit_input(image, face)
            assert normalized.shape == (1, 22, 2) and mask.shape == (1, 384, 384)
            np.testing.assert_allclose(report['normalized_keypoints'], normalized[0].numpy())
            for key in ('source_file', 'annotated_input_file', 'input_file', 'mask_file'):
                assert (tmp_path / report[key]).is_file()
    finally:
        detector.close()
    assert detector.pose_refiner is None
    assert not list(tmp_path.rglob('model.pth'))  # No SapiensID identity weights.
