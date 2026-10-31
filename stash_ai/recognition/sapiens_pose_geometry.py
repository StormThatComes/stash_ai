"""Sapiens RGB affine crops and distribution-aware UDP heatmap decoding.

Uses the published Sapiens v1 training normalization and target aspect ratio.
UDP: https://arxiv.org/abs/1911.07524; DARK: https://arxiv.org/abs/1910.06278.
"""
import numpy as np

from . import sapiens_pose_registry as registry


def prepare(image, box):
    import cv2
    width, height = registry.INPUT_SIZE
    box = np.asarray(box, dtype=np.float32)
    center = (box[:2] + box[2:]) * .5
    scale = (box[2:] - box[:2]) * registry.PADDING
    aspect = width / height
    scale = np.array([max(scale[0], scale[1] * aspect),
                      max(scale[1], scale[0] / aspect)], dtype=np.float32)
    origin = center - scale * .5
    factors = (np.array([width, height]) - 1) / scale
    matrix = np.array([[factors[0], 0, -origin[0] * factors[0]],
                       [0, factors[1], -origin[1] * factors[1]]], dtype=np.float32)
    if image.mode == 'RGB':
        rgb = np.asarray(image)
    else:
        with image.convert('RGB') as converted:
            rgb = np.array(converted)
    crop = cv2.warpAffine(rgb, matrix, (width, height), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    crop = (crop.astype(np.float32) - np.asarray(registry.MEAN, dtype=np.float32))
    crop /= np.asarray(registry.STD, dtype=np.float32)
    return np.ascontiguousarray(crop.transpose(2, 0, 1)), center, scale


def decode(heatmaps, center, scale):
    """Decode only the selected COCO17 maps, preserving non-positive missing points."""
    import cv2
    width, height = registry.HEATMAP_SIZE
    if heatmaps.shape != (17, height, width) or not np.isfinite(heatmaps).all():
        raise ValueError('Sapiens Pose returned malformed/non-finite COCO17 heatmaps.')
    points = np.full((17, 2), -1., dtype=np.float32)
    scores = heatmaps.reshape(17, -1).max(axis=1)
    border = registry.BLUR_KERNEL // 2
    for joint in np.flatnonzero(scores > 0):
        heatmap = heatmaps[joint]
        y, x = np.unravel_index(heatmap.argmax(), heatmap.shape)
        # Smooth with zero padding, preserve the original maximum, then use
        # derivatives of log response for subpixel localization.
        blurred = cv2.GaussianBlur(np.pad(heatmap, border),
            (registry.BLUR_KERNEL, registry.BLUR_KERNEL), 0)[border:-border, border:-border]
        peak = blurred.max()
        if peak <= 0:
            points[joint] = (x, y)
            continue
        logmap = np.pad(np.log(np.clip(blurred * (scores[joint] / peak), 1e-3, 50.)), 1, mode='edge')
        row, col = y + 1, x + 1
        value = logmap[row, col]
        left, right = logmap[row, col - 1], logmap[row, col + 1]
        top, bottom = logmap[row - 1, col], logmap[row + 1, col]
        gradient = np.array([(right - left) * .5, (bottom - top) * .5])
        cross = .5 * (logmap[row + 1, col + 1] - right - bottom + 2 * value
                       - left - top + logmap[row - 1, col - 1])
        hessian = np.array([[right - 2 * value + left, cross],
                            [cross, bottom - 2 * value + top]], dtype=np.float64)
        hessian += np.eye(2) * np.finfo(np.float32).eps
        try:
            offset = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            offset = np.zeros(2)
        points[joint] = np.array([x, y]) - offset
    valid = scores > 0
    points[valid] = points[valid] / [width - 1, height - 1] * scale + center - scale * .5
    if not np.isfinite(points).all():
        raise ValueError('Sapiens Pose returned non-finite decoded coordinates.')
    return points, np.clip(scores, 0., 1.)
