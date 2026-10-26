"""SapiensID v1 preprocessing and local checkpoint contracts."""
import hashlib
import os
from pathlib import Path
import stat

INPUT_SIZE = (384, 384)
EMBEDDING_DIMENSIONS = 4096
# Independent checkpoint conversion metadata (peabody124/sapiensid-eqx).
# Upstream distributes its PyTorch checkpoint via Google Drive, not HF.
WB4M_BYTES = 1646175550
WB4M_SHA256 = '8df1e2d7a64fbc87969326862aeafccc83eb5ee0462a4f007d7f09603782cf02'

PREPROCESSING = {
    'source': 'https://github.com/mk-minchul/sapiensid',
    'input': 'float32 RGB NCHW (1,3,384,384)',
    'padding': 'square, symmetric, white in ToTensor space (fill=1)',
    'resize': 'torchvision bilinear on float tensor, antialias=None',
    'normalization': '(pixel / 255 - 0.5) / 0.5',
    'embedding_dimensions': EMBEDDING_DIMENSIONS,
    'requires': 'person detection, 22 raw keypoints (KP19 reduction inside model)',
    'license': 'CC-BY-NC-4.0 (upstream)',
}


def preprocess(image):
    """Match upstream SapiensModel.make_test_transform for RGB PIL inputs.

    Returns CPU torch float32 tensor [1,3,384,384]. Not a model inference,
    keypoint transform, body detector or full SapiensID adapter.
    """
    from PIL import Image
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as F
    if not isinstance(image, Image.Image) or min(image.size) < 1:
        raise ValueError('SapiensID requires a non-empty PIL image.')
    with image.convert('RGB') as rgb:
        pixels = F.to_tensor(rgb)
    _, height, width = pixels.shape
    pad_width, pad_height = max(height-width, 0), max(width-height, 0)
    # The upstream SquarePad applies the asymmetric remainder to right/bottom.
    padding = (pad_width // 2, pad_height // 2,
               pad_width-pad_width // 2, pad_height-pad_height // 2)
    if any(padding):
        pixels = F.pad(pixels, padding, fill=1, padding_mode='constant')
    pixels = F.resize(pixels, list(INPUT_SIZE),
                      interpolation=InterpolationMode.BILINEAR, antialias=None)
    pixels = F.normalize(pixels, mean=[.5, .5, .5], std=[.5, .5, .5])
    return pixels.unsqueeze(0).contiguous()


def verify_wb4m_checkpoint(path, *, expected_size=WB4M_BYTES, expected_sha256=WB4M_SHA256):
    """Verify a local upstream PyTorch wb4m checkpoint before any torch.load.

    Asset acquisition belongs to sapiensid_assets; this verifier is offline.
    Hash source is the separately published Equinox conversion card, also
    checked against the original upstream archive by the download smoke test.
    This verifies bytes, not their suitability for the current face pipeline.
    """
    if (type(expected_size) is not int or expected_size <= 0
            or not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(c not in '0123456789abcdef' for c in expected_sha256)):
        raise ValueError('Invalid expected checkpoint integrity metadata.')
    source = Path(path)
    with source.open('rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
            raise ValueError('SapiensID checkpoint size mismatch.')
        digest = hashlib.sha256()
        actual_bytes = 0
        for block in iter(lambda: stream.read(1048576), b''):
            actual_bytes += len(block)
            if actual_bytes > expected_size:
                raise ValueError('SapiensID checkpoint changed during verification.')
            digest.update(block)
        after = os.fstat(stream.fileno())
        if (actual_bytes != expected_size or after.st_size != expected_size
                or digest.hexdigest() != expected_sha256):
            raise ValueError('SapiensID checkpoint SHA256/size mismatch.')
    return source


def prepare_person_input(image, points, *, device="cpu"):
    """Build the exact model tensors shared by inference and alignment audit."""
    import torch
    import numpy as np
    raw = np.asarray(points, dtype=np.float32)
    if raw.shape != (22, 2) or not np.isfinite(raw).all():
        raise ValueError('SapiensID requires 22 finite pixel landmark pairs.')
    side = max(image.size)
    padding = np.array([(side-image.width)//2, (side-image.height)//2], dtype=np.float32)
    missing = (raw == -1).all(axis=1)
    if not np.all(missing | ((raw >= 0).all(axis=1) &
                             (raw < np.array(image.size)).all(axis=1))):
        raise ValueError('SapiensID pixel landmarks must be inside the crop or (-1,-1).')
    normalized = (raw + padding) / side
    normalized[missing] = -1
    batch = preprocess(image).to(device)
    keypoints = torch.from_numpy(normalized[None]).to(device)
    # Official whitespace mask: exclude square white padding and contiguous
    # white border rows/columns, retaining the content rectangle.
    unnormalized = (batch + 1) / 2
    summed = unnormalized.sum(dim=1)
    rows, cols = summed.mean(dim=2) < 2.97, summed.mean(dim=1) < 2.97
    def bounds(mask):
        first = mask.float().argmax(dim=1)
        last = 383 - mask.flip(1).float().argmax(dim=1)
        return torch.where(mask.any(1), first, 0), torch.where(mask.any(1), last, 383)
    top, bottom = bounds(rows)
    left, right = bounds(cols)
    y, x = torch.arange(384, device=device)[None, :, None], torch.arange(384, device=device)[None, None, :]
    masks = (y >= top[:,None,None]) & (y <= bottom[:,None,None]) & (x >= left[:,None,None]) & (x <= right[:,None,None])
    return batch, keypoints, masks
