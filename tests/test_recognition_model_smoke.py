"""Explicit CPU loading/alignment smoke, not a face accuracy benchmark."""
import os
import pytest


@pytest.mark.skipif(os.getenv('STASH_AI_RECOGNITION_SMOKE') != '1', reason='explicit pinned face-model smoke')
@pytest.mark.parametrize('recognizer,dimensions', [('sface', 128), ('auraface', 512)])
def test_pinned_face_models_on_cpu(tmp_path, recognizer, dimensions):
    from PIL import Image
    from stash_ai.recognition.config import load
    from stash_ai.recognition.models import ensure_assets, load_backend, checked_detection
    path = tmp_path / 'facial_recognition.yml'
    path.write_text(f'version: 1\nmodels: {{recognizer: {recognizer}, device: cpu}}\n', encoding='utf-8')
    config = load(path)
    backend = load_backend(config, ensure_assets(config))
    try:
        with Image.new('RGB', (128, 128), (100, 120, 140)) as image:
            assert isinstance(backend.detect(image), list)
            detection = checked_detection([0, 0, 128, 128, 38, 44, 88, 44, 64, 66, 42, 90, 86, 90, .99])
            embedding = backend.embed(image, detection)
            assert embedding.shape == (dimensions,)
            assert backend.device == 'cpu'
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
