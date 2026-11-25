"""Contrast-normalized directional reblur, shared by NumPy and Torch batches.

Scores describe resolved structure, not the cause of blur. Every filter tap must
be inside the source mask. Noise power is estimated locally with a mixed second
derivative and removed from gradient energies before comparing two reblur scales.
No model, image resizing, or device-to-host image transfer is required.
"""
import math

import numpy as np

from .sharpness_policy import ALGORITHM
RADIUS = 7  # smoothing (2), derivative (1), largest reblur (4)
TILE = 32
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))
SMOOTH = (1/16, 4/16, 6/16, 4/16, 1/16)
REBLUR = (SMOOTH, tuple(math.comb(8, k)/256 for k in range(9)))


def _line(values, weights, dy, dx):
    """Valid line convolution; trim equally on both axes for common supports."""
    radius = len(weights)//2
    height, width = values.shape[-2:]
    result = values[..., radius:height-radius, radius:width-radius]*0
    for offset, weight in enumerate(weights, -radius):
        result = result + weight*values[..., radius+dy*offset:height-radius+dy*offset,
                                        radius+dx*offset:width-radius+dx*offset]
    return result


def _noise_gains():
    """Squared filter norms: expected response variance for unit white noise."""
    # An impulse also exercises exactly the filters used on the real pixels.
    impulse = np.zeros((1, 41, 41), np.float64)
    impulse[0, 20, 20] = 1
    smooth = _smooth(impulse)
    result = []
    for dy, dx in DIRECTIONS:
        gradient = _line(smooth, (-.5, 0., .5), dy, dx)
        result.append(tuple(float((v*v).sum()) for v in
                            (gradient, *(_line(gradient, w, dy, dx) for w in REBLUR))))
    return tuple(result)


def _smooth(values):
    # Separable binomial 5x5, without padding or changing the pixel scale.
    horizontal = sum(w*values[..., :, i:values.shape[-1]-4+i] for i, w in enumerate(SMOOTH))
    return sum(w*horizontal[..., i:horizontal.shape[-2]-4+i, :] for i, w in enumerate(SMOOTH))


NOISE_GAINS = _noise_gains()


def _zeros(shape, like):
    return np.zeros(shape, like.dtype) if isinstance(like, np.ndarray) else like.new_zeros(shape)


def _pool(values):
    """Bounded local sums, with zero weight for partial tiles."""
    n, height, width = values.shape
    rows, cols = (height+TILE-1)//TILE, (width+TILE-1)//TILE
    padded = _zeros((n, rows*TILE, cols*TILE), values)
    padded[:, :height, :width] = values
    return padded.reshape(n, rows, TILE, cols, TILE).sum(axis=(2, 4)).reshape(n, -1)


def _interior(mask):
    """Integral-image erosion of the complete 15x15 filter footprint."""
    n, height, width = mask.shape
    summed = _zeros((n, height+1, width+1), mask)
    summed[:, 1:, 1:] = mask.cumsum(axis=1).cumsum(axis=2)
    size = 2*RADIUS+1
    return (summed[:, size:, size:]-summed[:, :-size, size:]
            -summed[:, size:, :-size]+summed[:, :-size, :-size]) == size*size


def _quartile(values, valid, xp):
    """Lower quartile of informative tiles; empty directions remain absent."""
    ordered = xp.where(valid, values, float('inf'))
    counts = valid.sum(axis=1)
    if xp is np:
        ordered = np.sort(ordered, axis=1)
        index = ((counts-1).clip(0, None)//4).astype(np.int64)
        return np.take_along_axis(ordered, index[:, None], axis=1)[:, 0]
    ordered = ordered.sort(dim=1).values
    index = (counts-1).clamp_min(0)//4
    return ordered.gather(1, index[:, None])[:, 0]


def batch(values, masks, xp=np):
    """Return N x 7 summaries: legacy four, focus, motion, split confidence.

    Inputs are finite float32 N,H,W arrays on the same device, masks are bool.
    NumPy and Torch execute the same arithmetic and reduction policy.
    """
    finite = xp.isfinite(values)
    masks = masks & finite
    values = xp.where(finite, values, 0.)
    weights = xp.where(masks, values*0+1, 0.)
    counts = weights.sum(axis=(1, 2)).clip(1, None)
    means = (values*weights).sum(axis=(1, 2))/counts
    contrast = (((values-means[:, None, None])**2)*weights).sum(axis=(1, 2))/counts
    zeros = contrast*0
    if min(values.shape[-2:]) <= 2*RADIUS:
        return xp.stack((zeros, zeros, contrast, zeros, zeros, zeros, zeros), axis=1)
    support = _interior(weights)
    weight = xp.where(support, values[:, RADIUS:-RADIUS, RADIUS:-RADIUS]*0+1, 0.)
    count = _pool(weight)
    denominator = count.clip(1, None)

    # Mixed derivative rejects straight edges and smooth shading. Its kernel
    # norm is 6; the absolute Gaussian mean is sigma*sqrt(2/pi).
    horizontal = values[:, :, :-2]-2*values[:, :, 1:-1]+values[:, :, 2:]
    highpass = horizontal[:, :-2]-2*horizontal[:, 1:-1]+horizontal[:, 2:]
    highpass = highpass[:, 6:-6, 6:-6]
    noise = _pool(abs(highpass)*weight)/denominator*(math.sqrt(math.pi/2)/6)
    noise_power = noise**2
    smooth = _smooth(values)
    scores = []
    for (dy, dx), gains in zip(DIRECTIONS, NOISE_GAINS):
        gradient = _line(smooth, (-.5, 0., .5), dy, dx)
        samples = [gradient[:, 4:-4, 4:-4]]
        for kernel in REBLUR:
            blurred = _line(gradient, kernel, dy, dx)
            trim = 4-len(kernel)//2
            samples.append(blurred[:, trim:-trim, trim:-trim] if trim else blurred)
        energies = []
        for sample, gain in zip(samples, gains):
            mean = _pool(sample*weight)/denominator
            variance = (_pool(sample*sample*weight)/denominator-mean**2).clip(0, None)
            energies.append((variance-noise_power*gain).clip(0, None))
        base = energies[0]
        # Ignore noise-only tiles and absent directions, never treat them as
        # evidence for sharpness or as a directional-blur penalty.
        valid = (count >= 16) & (base > 3*noise_power*gains[0]+.01)
        # Reblur can move an edge from a neighbouring tile into this one. An
        # energy gain then describes the tile boundary, not a blurry edge.
        # Compare only tiles where both scales have a measurable energy loss.
        valid = valid & (energies[1] <= base) & (energies[2] <= base)
        losses = [(1-energy/base.clip(1e-8, None)).clip(0, 1) for energy in energies[1:]]
        local = (losses[0]+losses[1])*.5
        scores.append(_quartile(local, valid, xp))
    directional = xp.stack(scores, axis=1)
    score = directional.min(axis=1) if xp is np else directional.min(dim=1).values
    score = xp.where(xp.isfinite(score), 100*score, 0.)
    pixels = weight.sum(axis=(1, 2))
    estimated_noise = (noise*count).sum(axis=1)/pixels.clip(1, None)
    from .split_blur import estimate
    split = estimate(smooth, support, noise, xp)
    return xp.concatenate((xp.stack((score, pixels, contrast, estimated_noise), axis=1), split), axis=1)


def records(summaries, *, diagnostics):
    # Both legacy selectors now expose the same units for every quality gate.
    return [dict(sharpness=float(score), sharpness_normalized=float(score),
                 sharpness_pixels=int(count), sharpness_contrast=float(contrast),
                 sharpness_noise=float(noise) if diagnostics else None,
                 sharpness_algorithm=ALGORITHM,
                 focus_sharpness=float(focus) if confidence >= .6 else None,
                 motion_sharpness=float(motion) if confidence >= .6 else None,
                 sharpness_confidence=float(confidence),
                 sharpness_split_method="edge-spread-covariance-v1",
                 sharpness_split_status="estimated" if confidence >= .6 else "unavailable")
            for score, count, contrast, noise, focus, motion, confidence in summaries]


def measure(values, mask=None, *, diagnostics=True):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError('Sharpness pixels must be a grayscale 2D array.')
    mask = np.ones(values.shape, dtype=bool) if mask is None else np.asarray(mask)
    if mask.shape != values.shape or mask.dtype != np.bool_:
        raise ValueError('Sharpness mask must be boolean and match the measured pixels.')
    return records(batch(values[None], mask[None]), diagnostics=diagnostics)[0]
