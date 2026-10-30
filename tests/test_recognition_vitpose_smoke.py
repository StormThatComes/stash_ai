"""Opt-in, real pretrained YOLO + ViTPose++ Large CPU integration smoke."""
import os

import pytest


@pytest.mark.skipif(os.getenv('STASH_AI_VITPOSE_SMOKE') != '1', reason='explicit 1.74 GB ViTPose smoke')
def test_pretrained_vitpose_large_person_geometry_and_audit(tmp_path, request):
    import hashlib
    from io import BytesIO
    import numpy as np
    import requests
    import torch
    from PIL import Image
    from stash_ai.recognition.config import load
    from stash_ai.recognition.models import checked_detection
    from stash_ai.recognition.sapiensid_backend import SapiensIDFaces
    from stash_ai.recognition.person_collection import audit_input

    previous_threads = torch.get_num_threads()
    request.addfinalizer(lambda: torch.set_num_threads(previous_threads))
    torch.set_num_threads(2)
    url = ('https://raw.githubusercontent.com/matplotlib/matplotlib/v3.10.1/'
           'lib/matplotlib/mpl-data/sample_data/grace_hopper.jpg')
    with requests.get(url, timeout=30) as response:
        response.raise_for_status()
        raw = response.content
    assert hashlib.sha256(raw).hexdigest() == 'a8ca6d734765703b09728ab47fe59f473d93ae3967fc24c7c0288c3c7adb7130'
    path = tmp_path / 'recognition.yml'
    path.write_text('version: 1\nmodels: {device: cpu}\nsapiensid:\n'
                    '  enabled: true\n  pose: {backend: vitpose_plus_large, batch_size: 1}\n')
    config = load(path)
    face = checked_detection([182.86642, 120.47789, 170.25731, 212.48526,
        227.11728, 193.32774, 307.84845, 195.3718, 266.4793, 240.008,
        228.50974, 267.86563, 302.18546, 269.56985, .94316435])
    detector = SapiensIDFaces(config, {}, load_identity=False, load_detector=False)
    try:
        assert detector.encoder is None
        with Image.open(BytesIO(raw)).convert('RGB') as image:
            detector.detect(image, faces=[face])
            report = audit_input(detector, image, face, tmp_path / 'audit')
            assert report['identity_rejection'] is None
            assert 'vitpose-plus-large' in report['identity_input']
            assert len(report['person']['keypoints']) == 17
            _, points, mask = detector.audit_input(image, face)
            assert points.shape == (1, 22, 2) and mask.shape == (1, 384, 384)
            np.testing.assert_allclose(report['normalized_keypoints'], points[0].numpy())
            for key in ('source_file', 'annotated_input_file', 'input_file', 'mask_file'):
                assert (tmp_path / report[key]).is_file()
    finally:
        detector.close()
    assert not list(tmp_path.rglob('model.pth'))  # Audit does not fetch SapiensID identity weights.
