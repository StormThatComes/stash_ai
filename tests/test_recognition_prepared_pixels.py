import numpy as np
import pytest
from PIL import Image

from stash_ai.recognition.backend import LazyBackend, prepared_frame
from stash_ai.recognition.models import YuNetDetector
from tests.helpers.recognition import Backend


def test_one_owned_bgr_conversion_serves_detection_and_all_face_embeddings(monkeypatch):
    model = YuNetDetector.__new__(YuNetDetector)
    model.detect = lambda image: model._bgr(image)
    model.embed = lambda image, face: model._bgr(image)
    owner = LazyBackend(None, {}, lambda *args: model)
    lease = owner.borrow()
    calls, original = [], Image.Image.convert
    def convert(image, *args, **kwargs):
        calls.append(1)
        return original(image, *args, **kwargs)
    monkeypatch.setattr(Image.Image, 'convert', convert)
    pixels = np.array([[[1, 2, 3], [10, 20, 30]]], dtype='uint8')
    with Image.fromarray(pixels) as image:
        with prepared_frame(lease, image):
            first = lease.detect(image)
            assert first.flags.c_contiguous
            assert np.array_equal(first, pixels[:, :, ::-1])
            for _ in range(10):
                assert lease.embed(image, None) is first
        assert model._prepared_frame is None and len(calls) == 1
        image.putpixel((0, 0), (7, 8, 9))
        assert lease.detect(image)[0, 0].tolist() == [9, 8, 7]
        assert len(calls) == 2
    owner.close()


def test_failed_embedding_releases_prepared_pixels_and_invalidates_only_its_model():
    first = YuNetDetector.__new__(YuNetDetector)
    def fail(*args):
        raise ValueError('embedding failed')
    first.embed = fail
    replacement = Backend()
    models = iter((first, replacement))
    owner = LazyBackend(None, {}, lambda *args: next(models))
    lease = owner.borrow()
    with Image.new('RGB', (3, 3)) as image:
        with pytest.raises(ValueError, match='embedding failed'):
            with prepared_frame(lease, image):
                lease.embed(image, None)
        assert first._prepared_frame is None and not lease.active
        assert owner.borrow().model is replacement
        with pytest.raises(RuntimeError, match='no longer active'):
            with prepared_frame(lease, image):
                pass
    owner.close()


def test_custom_backend_without_preparation_still_uses_its_existing_interface():
    owner = LazyBackend(None, {}, lambda *args: Backend())
    lease = owner.borrow()
    with Image.new('RGB', (3, 3)) as image, prepared_frame(lease, None):
        assert lease.embed(image, None).tolist() == [1., 0., 0., 0.]
    owner.close()


def test_prepared_frame_cleanup_keeps_the_primary_failure():
    model = Backend()
    model.prepare_frame = lambda image: None
    def fail(previous):
        raise RuntimeError('release failed')
    model.release_frame = fail
    owner = LazyBackend(None, {}, lambda *args: model)
    with pytest.raises(ValueError, match='ranking failed') as caught:
        with prepared_frame(owner.borrow(), None):
            raise ValueError('ranking failed')
    assert owner.model is None and model.closed
    assert any('release failed' in note for note in caught.value.__notes__)
    owner.close()
