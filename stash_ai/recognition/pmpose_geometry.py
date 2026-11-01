"""PMPose RGB crops, optional upstream mask conditioning and DARK/UDP decoding.

Adapted from OpenMMLab's Apache-2.0 MMPose geometry and the PMPose fork.
See THIRD_PARTY_NOTICES.md for sources and attribution.
"""
import numpy as np

from . import pmpose_registry as registry


def prepare(image, box, mask=None):
    import cv2
    box = np.asarray(box, dtype=np.float32).copy()
    box[:2] = np.maximum(box[:2], 0)
    box[2:] = np.minimum(box[2:], image.size)
    if not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        raise ValueError('PMPose requires non-empty person boxes intersecting the image.')
    center = (box[:2] + box[2:]) * .5
    scale = (box[2:] - box[:2]) * registry.PADDING
    width, height = registry.INPUT_SIZE
    scale = np.array([max(scale[0], scale[1] * width / height),
                      max(scale[1], scale[0] * height / width)], dtype=np.float32)
    origin = center - scale * .5
    factors = (np.asarray(registry.INPUT_SIZE) - 1) / scale
    matrix = np.array([[factors[0], 0, -origin[0] * factors[0]],
                       [0, factors[1], -origin[1] * factors[1]]], dtype=np.float32)
    with image.convert('RGB') as rgb:
        pixels = np.asarray(rgb)
        if mask is not None:
            mask = np.asarray(mask)
            if mask.shape != (image.height, image.width) or mask.dtype != np.bool_:
                raise ValueError('PMPose mask must be a full-frame boolean instance mask.')
            # Upstream MaskBackground(alpha=.25), BEFORE affine sampling and
            # normalization. Keep 75% background context; do not black it out.
            masked = pixels * mask[:, :, None]
            pixels = cv2.addWeighted(pixels, .75, masked, .25, 0)
        crop = cv2.warpAffine(pixels, matrix, (width, height), flags=cv2.INTER_LINEAR)
    crop = (crop.astype(np.float32) - np.asarray(registry.MEAN, dtype=np.float32))
    crop /= np.asarray(registry.STD, dtype=np.float32)
    return np.ascontiguousarray(crop.transpose(2, 0, 1)), center, scale


def decode(heatmaps, center, scale):
    import cv2
    width, height = registry.HEATMAP_SIZE
    if (heatmaps.shape != (17, height, width) or not np.isfinite(heatmaps).all()
            or np.any((heatmaps < 0) | (heatmaps > 1))):
        raise ValueError('PMPose returned malformed/non-finite COCO17 heatmaps.')
    peaks = heatmaps.reshape(17, -1).max(axis=1)
    points = np.full((17, 2), -1., dtype=np.float32)
    for joint in np.flatnonzero(peaks > 0):
        heatmap = heatmaps[joint]
        y, x = np.unravel_index(heatmap.argmax(), heatmap.shape)
        blurred = cv2.GaussianBlur(np.pad(heatmap, 5), (11, 11), 0)[5:-5, 5:-5]
        blurred *= peaks[joint] / (blurred.max() + 1e-12)
        logmap = np.pad(np.log(np.clip(blurred, 1e-3, 50.)), 1, mode='edge')
        row, col = y + 1, x + 1
        value = logmap[row, col]
        left, right = logmap[row, col - 1], logmap[row, col + 1]
        top, bottom = logmap[row - 1, col], logmap[row + 1, col]
        gradient = np.array([(right - left) * .5, (bottom - top) * .5])
        cross = .5 * (logmap[row + 1, col + 1] - right - bottom + 2 * value
                       - left - top + logmap[row - 1, col - 1])
        hessian = np.array([[right - 2 * value + left, cross],
                            [cross, bottom - 2 * value + top]], dtype=np.float64)
        offset = np.linalg.pinv(hessian + np.eye(2) * np.finfo(np.float32).eps) @ gradient
        points[joint] = np.array([x, y]) - offset
    valid = peaks > 0
    points[valid] = points[valid] / [width - 1, height - 1] * scale + center - scale * .5
    if not np.isfinite(points).all():
        raise ValueError('PMPose returned non-finite decoded coordinates.')
    return points
