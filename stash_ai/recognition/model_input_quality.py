"""Sharpness in encoder coordinates, with conservative resampled mask support."""
import numpy as np

from .blur_quality import batch, records
from .sharpness_policy import INPUT_SPACE as SPACE


def face_input(image, face, affine, pixels=None):
    """Keep exact encoder pixels and map their centers into the FaceOcc mask.

    OpenCV uses integer pixel centers; PIL affine sampling uses half centers.
    The mapping is geometric, never a resize of the already resampled 256px RGB.
    """
    import cv2
    from .aura import similarity_matrix
    try:
        matrix = similarity_matrix(face.landmarks)
    except ValueError as error:
        from .visibility import VisibilityGeometryError
        raise VisibilityGeometryError('Cannot align recognition sharpness input: '+str(error)) from error
    if pixels is None:
        with image.convert('RGB') as rgb:
            pixels = cv2.warpAffine(np.asarray(rgb), matrix, (112, 112), borderValue=0.)
    gray = np.asarray(pixels, np.float32) @ np.array([.299, .587, .114], np.float32)
    if gray.shape != (112, 112):
        raise ValueError('Face sharpness requires the recognizer 112x112 RGB input.')
    y, x = np.indices((112, 112), dtype=np.float32)
    source = cv2.invertAffineTransform(matrix)
    sx = source[0, 0]*x + source[0, 1]*y + source[0, 2]
    sy = source[1, 0]*x + source[1, 1]*y + source[1, 2]
    real = (sx >= 0) & (sy >= 0) & (sx <= image.width-1) & (sy <= image.height-1)
    inverse = cv2.invertAffineTransform(np.asarray(affine, np.float64).reshape(2, 3))
    mx = inverse[0, 0]*(sx+.5) + inverse[0, 1]*(sy+.5) + inverse[0, 2] - .5
    my = inverse[1, 0]*(sx+.5) + inverse[1, 1]*(sy+.5) + inverse[1, 2] - .5
    return gray, real, mx.astype(np.float32), my.astype(np.float32)


def project_masks(masks, inputs, xp=np):
    """Require every bilinear mask tap, including holes, to be foreground."""
    def array(value):
        return np.asarray(value) if xp is np else xp.as_tensor(value, device=masks.device)
    x, y = (array(np.stack([item[i] for item in inputs])) for i in (2, 3))
    left = xp.floor(x).astype(np.int64) if xp is np else xp.floor(x).long()
    top = xp.floor(y).astype(np.int64) if xp is np else xp.floor(y).long()
    n, height, width = masks.shape
    index = np.arange(n)[:, None, None] if xp is np else xp.arange(n, device=masks.device)[:, None, None]
    valid = array(np.stack([item[1] for item in inputs]))
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        xx, yy = left+dx, top+dy
        needed = ((x-left > 1e-6) if dx else (x-left < 1-1e-6)) & ((y-top > 1e-6) if dy else (y-top < 1-1e-6))
        inside = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
        valid = valid & (~needed | (inside & masks[index, yy.clip(0, height-1), xx.clip(0, width-1)]))
    return valid


def faces(inputs, masks, *, diagnostics=True, xp=np):
    pixels = np.stack([item[0] for item in inputs])
    if xp is not np:
        pixels = xp.as_tensor(pixels, device=masks.device)
    summaries = batch(pixels, project_masks(masks, inputs, xp), xp)
    summaries = summaries if xp is np else summaries.cpu().tolist()
    return [dict(value, sharpness_space=SPACE, sharpness_input_size=[112, 112])
            for value in records(summaries, diagnostics=diagnostics)]


def body(image, mask=None, encoder_mask=None):
    """Use the same square padding and float bilinear resize as SapiensID."""
    import torch
    from PIL import Image
    from .sapiensid_v1 import preprocess, square_resize
    with image.convert('RGB') as rgb:
        source = np.asarray(rgb).copy()
    support = np.ones(source.shape[:2], bool) if mask is None else np.asarray(mask, bool).copy()
    if encoder_mask is not None:
        encoder_mask = np.asarray(encoder_mask, bool)
        source[~encoder_mask] = 127
        # Do not count the synthetic outline even when mask-based measurement
        # is disabled; retain both constant background and foreground interiors.
        import cv2
        kernel = np.ones((3, 3), 'uint8')
        homogeneous = (cv2.erode(encoder_mask.astype('uint8'), kernel)
                       | cv2.erode((~encoder_mask).astype('uint8'), kernel))
        support &= homogeneous.astype(bool)
    with Image.fromarray(source) as pixels:
        prepared = preprocess(pixels)[0]
    gray = (((prepared+1)*127.5)*prepared.new_tensor([.299, .587, .114])[:, None, None]).sum(dim=0)
    valid = square_resize(torch.from_numpy(support.astype(np.float32))[None], fill=0)[0] >= 1-1e-6
    value = records(batch(gray[None], valid[None], torch).tolist(), diagnostics=True)[0]
    return dict(value, sharpness_space=SPACE, sharpness_input_size=[384, 384],
                sharpness_source_size=list(image.size), sharpness_scale=384/max(image.size))
