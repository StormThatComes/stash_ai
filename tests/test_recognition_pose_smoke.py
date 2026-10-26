"""Explicit real-weight CPU smoke; no SapiensID identity checkpoint needed."""
import os

import pytest


@pytest.mark.skipif(os.getenv('STASH_AI_POSE_SMOKE')!='1',reason='explicit YOLO26m pose smoke')
def test_default_yolo26m_associates_face_and_exports_exact_person_input(tmp_path):
    import hashlib
    from io import BytesIO
    import numpy as np
    import requests
    from PIL import Image
    from stash_ai.recognition.config import load
    from stash_ai.recognition.models import checked_detection
    from stash_ai.recognition.person_collection import audit_input
    from stash_ai.recognition.sapiensid_backend import SapiensIDFaces, CONTRACT
    from stash_ai.recognition.sapiensid_registry import POSE_NAME, POSE_BYTES, POSE_SHA256

    # Grace Hopper, US Navy/public domain; the same pinned portrait as face smoke tests.
    url='https://raw.githubusercontent.com/matplotlib/matplotlib/v3.10.1/lib/matplotlib/mpl-data/sample_data/grace_hopper.jpg'
    with requests.get(url,timeout=30) as response:
        response.raise_for_status();data=response.content
    assert hashlib.sha256(data).hexdigest()=='a8ca6d734765703b09728ab47fe59f473d93ae3967fc24c7c0288c3c7adb7130'
    path=tmp_path/'recognition.yml'
    path.write_text('version: 1\nmodels: {device: cpu}\nsapiensid: {enabled: true}\n')
    config=load(path)
    face=checked_detection([182.86642,120.47789,170.25731,212.48526,227.11728,193.32774,
                            307.84845,195.3718,266.4793,240.008,228.50974,267.86563,
                            302.18546,269.56985,.94316435])
    detector=SapiensIDFaces(config,{},load_identity=False,load_detector=False)
    try:
        assert detector.encoder is None
        with Image.open(BytesIO(data)) as image:
            detector.detect(image,faces=[face])
            result=audit_input(detector,image,face,tmp_path/'portrait')
            assert result['identity_rejection'] is None and result['identity_input']==CONTRACT
            assert len(result['person']['keypoints'])==17
            assert len(result['normalized_keypoints'])==22
            batch,points,mask=detector.audit_input(image,face)
            expected=((batch[0].permute(1,2,0).numpy()+1)*127.5).clip(0,255).round().astype('uint8')
            with Image.open(tmp_path/result['input_file']) as actual:
                np.testing.assert_array_equal(np.asarray(actual),expected)
            for field in ('source_file','annotated_input_file','mask_file'):
                assert (tmp_path/result[field]).is_file()
            np.testing.assert_allclose(result['normalized_keypoints'],points[0].numpy())
            assert mask.shape==(1,384,384)
        assert detector.pose.predictor.model.end2end is False
    finally:detector.close()
    checkpoint=tmp_path/config.data['sapiensid']['pose']['path']
    assert checkpoint.name==POSE_NAME and checkpoint.stat().st_size==POSE_BYTES
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==POSE_SHA256
    assert not list(tmp_path.rglob('model.pth'))  # Audit never fetches identity weights.
