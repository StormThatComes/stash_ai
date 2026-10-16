"""Explicit CPU loading/alignment smoke, not a face accuracy benchmark."""
import os
import pytest


@pytest.mark.skipif(os.getenv('STASH_AI_RECOGNITION_SMOKE') != '1', reason='explicit pinned face-model smoke')
@pytest.mark.parametrize('recognizer,dimensions', [('sface', 128), ('auraface', 512)])
def test_pinned_face_models_on_cpu(tmp_path, recognizer, dimensions):
    import numpy as np
    from PIL import Image
    from stash_ai.recognition.config import load
    from stash_ai.recognition.models import ensure_assets, load_backend, checked_detection
    path = tmp_path / 'facial_recognition.yml'
    path.write_text(f'version: 1\nmodels: {{recognizer: {recognizer}, device: cpu}}\n', encoding='utf-8')
    config = load(path)
    backend = load_backend(config, ensure_assets(config))
    try:
        with Image.new('RGB', (128, 128), (100, 120, 140)) as image:
            faces = backend.detect(image)
            assert isinstance(faces, list)
            detection = checked_detection([0, 0, 128, 128, 38, 44, 88, 44, 64, 66, 42, 90, 86, 90, .99])
            embedding = backend.embed(image, detection)
            assert embedding.shape == (dimensions,)
            assert backend.device == 'cpu'
            with backend.prepared_frame(image):
                assert [face.row for face in backend.detect(image)] == [face.row for face in faces]
                assert np.allclose(backend.embed(image, detection), embedding, atol=1e-7)
            assert backend._prepared_frame is None
    finally:
        backend.close()


@pytest.mark.skipif(os.getenv('STASH_AI_RECOGNITION_SMOKE') != '1', reason='explicit pinned face-model smoke')
def test_ort_contract_matches_cpu_sface_and_handles_non_square_yunet(tmp_path):
    import cv2
    import numpy as np
    from PIL import Image
    from stash_ai.recognition.config import load
    from stash_ai.recognition.models import ensure_assets, load_backend, checked_detection
    from stash_ai.recognition.runtime import runtime
    from stash_ai.recognition.onnx_models import OnnxYuNet, OnnxSFace
    from tests.helpers.recognition import textured_image
    config = load(root=tmp_path, device='cpu')
    paths = ensure_assets(config)
    cpu = load_backend(config, paths)
    detector = OnnxYuNet(config, paths['yunet'], cv2, runtime(config))
    recognizer = OnnxSFace(config, paths['sface'], runtime(config))
    try:
        for shape in ((128,128), (160,90)):
            with Image.new('RGB',shape,(100,120,140)) as image:
                assert isinstance(detector.detect(cpu._bgr(image)),list)
        face = checked_detection([0,0,128,128,38,44,88,44,64,66,42,90,86,90,.99])
        with textured_image() as image:
            aligned = cpu.recognizer.alignCrop(cpu._bgr(image), np.asarray(face.row,dtype=np.float32))
            reference = cpu.embed(image,face)
            converted = recognizer.embed(aligned)
            assert np.allclose(reference,converted,atol=2e-5)
            assert float(np.dot(reference,converted)) > .99999
    finally:
        cpu.close()
        detector.close()
        recognizer.close()


@pytest.mark.skipif(os.getenv('STASH_AI_RECOGNITION_SMOKE') != '1', reason='explicit pinned visibility-model smoke')
def test_faceocc_checkpoint_detects_lower_face_covering_on_public_domain_portrait(tmp_path):
    import hashlib
    from io import BytesIO
    import requests
    from PIL import Image, ImageDraw
    from stash_ai.recognition.config import load
    from stash_ai.recognition.models import checked_detection, ensure_assets
    from stash_ai.recognition.visibility import FaceVisibility
    # Grace Hopper, US Navy/public domain; immutable content checked independently
    # of the matplotlib tag. Download only in the explicit model smoke jobs.
    url = 'https://raw.githubusercontent.com/matplotlib/matplotlib/v3.10.1/lib/matplotlib/mpl-data/sample_data/grace_hopper.jpg'
    with requests.get(url, timeout=30) as response:
        response.raise_for_status()
        data = response.content
    assert hashlib.sha256(data).hexdigest() == 'a8ca6d734765703b09728ab47fe59f473d93ae3967fc24c7c0288c3c7adb7130'
    config = load(root=tmp_path, device='cpu')
    model = FaceVisibility(config, ensure_assets(config))
    face = checked_detection([182.86642, 120.47789, 170.25731, 212.48526, 227.11728, 193.32774,
                              307.84845, 195.3718, 266.4793, 240.008, 228.50974, 267.86563,
                              302.18546, 269.56985, .94316435])
    try:
        with Image.open(BytesIO(data)) as original, original.convert('RGB') as image, image.copy() as covered:
            x, y, width, height = face.box
            ImageDraw.Draw(covered).rectangle((x, y + height * .6, x + width, y + height), fill=(70, 90, 160))
            visible = model.measure(image, face)
            occluded = model.measure(covered, face)
            batched = model.measure_many([(image,face),(covered,face)])
            assert [v['visibility']['face_pixels'] for v in batched]==[visible['visibility']['face_pixels']]*2
            assert [v['visible_face_fraction'] for v in batched]==pytest.approx(
                [visible['visible_face_fraction'],occluded['visible_face_fraction']],abs=.001)
        assert visible['visible_face_fraction'] > .8
        assert occluded['visible_face_fraction'] < .8
        assert visible['visible_face_fraction'] - occluded['visible_face_fraction'] > .2
        assert visible['visibility']['face_pixels'] == occluded['visibility']['face_pixels']
        assert model.device == 'cpu'
    finally:
        model.close()


@pytest.mark.skipif(os.getenv('STASH_AI_RECOGNITION_SMOKE') != '1', reason='explicit pinned expression-model smoke')
def test_pinned_expression_task_associates_mesh_and_scores_public_domain_portrait(tmp_path):
    import hashlib
    from io import BytesIO
    import requests
    from PIL import Image
    from stash_ai.recognition.config import load
    from stash_ai.recognition.expression import FaceExpression, cached
    from stash_ai.recognition.models import checked_detection, ensure_assets
    with requests.get('https://raw.githubusercontent.com/matplotlib/matplotlib/v3.10.1/lib/matplotlib/mpl-data/sample_data/grace_hopper.jpg', timeout=30) as response:
        response.raise_for_status(); data = response.content
    assert hashlib.sha256(data).hexdigest() == 'a8ca6d734765703b09728ab47fe59f473d93ae3967fc24c7c0288c3c7adb7130'
    path = tmp_path / 'expression.yml'
    path.write_text('version: 1\nmodels: {device: cpu}\nportraits: {weights: {closed_mouth: 2}}\n')
    config = load(path)
    model = FaceExpression(config, ensure_assets(config))
    face = checked_detection([182.86642, 120.47789, 170.25731, 212.48526, 227.11728, 193.32774,
                              307.84845, 195.3718, 266.4793, 240.008, 228.50974, 267.86563,
                              302.18546, 269.56985, .94316435])
    try:
        with Image.open(BytesIO(data)) as image:
            result = model.measure(image, face)
        assert cached(result) and result['expression']['device'] == 'cpu'
        assert result['closed_mouth'] > .8 and result['open_eyes'] > .5
        assert result['open_mouth'] + result['closed_mouth'] == pytest.approx(1)
    finally:
        model.close()
