"""Explicit CPU loading/alignment smoke, not a face accuracy benchmark."""
import os
import pytest


@pytest.mark.skipif(os.getenv('STASH_AI_RECOGNITION_SMOKE') != '1', reason='explicit pinned face-model smoke')
def test_pinned_face_models_on_cpu(tmp_path):
    from PIL import Image
    from stash_ai.recognition.config import load
    from stash_ai.recognition.models import ensure_assets, load_backend, checked_detection
    config = load(root=tmp_path)
    backend = load_backend(config, ensure_assets(config))
    try:
        with Image.new('RGB', (128, 128), (100, 120, 140)) as image:
            assert isinstance(backend.detect(image), list)
            detection = checked_detection([0, 0, 128, 128, 38, 44, 88, 44, 64, 66, 42, 90, 86, 90, .99])
            embedding = backend.embed(image, detection)
            assert embedding.shape == (128,)
            assert backend.device == 'cpu'
    finally:
        backend.close()
