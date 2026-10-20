"""Contracts for SapiensID v1 readiness; no checkpoint/model inference."""
import hashlib

import numpy as np
import pytest

from stash_ai.recognition.sapiensid_v1 import (
    INPUT_SIZE, EMBEDDING_DIMENSIONS, PREPROCESSING, preprocess,
    verify_wb4m_checkpoint,
)


@pytest.mark.parametrize('size', [(384, 384), (120, 240), (241, 120), (1, 3), (63, 64)])
def test_preprocess_matches_upstream_tensor_squarepad_pipeline(size):
    torch = pytest.importorskip('torch')
    torchvision = pytest.importorskip('torchvision')
    from PIL import Image
    from torchvision.transforms import functional as F, InterpolationMode
    image = Image.new('RGB', size, color=(16, 128, 241))
    actual = preprocess(image)
    expected = F.to_tensor(image)
    _, height, width = expected.shape
    dw, dh = max(0, height-width), max(0, width-height)
    expected = F.pad(expected, (dw//2, dh//2, dw-dw//2, dh-dh//2), fill=1)
    expected = F.resize(expected, [384, 384], interpolation=InterpolationMode.BILINEAR,
                        antialias=None)
    expected = F.normalize(expected, [.5]*3, [.5]*3).unsqueeze(0)
    assert actual.shape == (1, 3, *INPUT_SIZE)
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)
    assert PREPROCESSING['embedding_dimensions'] == EMBEDDING_DIMENSIONS == 4096


def test_square_padding_is_white_and_centered():
    torch = pytest.importorskip('torch')
    from PIL import Image
    result = preprocess(Image.new('RGB', (64, 32), (255, 0, 0)))
    assert torch.allclose(result[0, :, 10, 192], torch.tensor([1., 1., 1.]))
    assert torch.allclose(result[0, :, 192, 192], torch.tensor([1., -1., -1.]))


def test_verify_checkpoint_hash_and_size(tmp_path):
    data = b'a short checkpoint fixture'
    source = tmp_path / 'model.pth'
    source.write_bytes(data)
    expected = dict(expected_size=len(data), expected_sha256=hashlib.sha256(data).hexdigest())
    assert verify_wb4m_checkpoint(source, **expected) == source
    with pytest.raises(ValueError, match='size mismatch'):
        verify_wb4m_checkpoint(source, expected_size=len(data)+1,
                               expected_sha256=expected['expected_sha256'])
    with pytest.raises(ValueError, match='SHA256'):
        verify_wb4m_checkpoint(source, expected_size=len(data), expected_sha256='0'*64)
    with pytest.raises(ValueError, match='integrity metadata'):
        verify_wb4m_checkpoint(source, expected_size=0, expected_sha256='0'*64)
