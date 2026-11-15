"""Select SAM prompts and settle mask ownership before running masked PMPose."""
from contextlib import ExitStack
from dataclasses import dataclass
import math

import numpy as np

from .pose_mask_capture import mask_observer
from .profiling import pipeline_stage


def inside(mask, points):
    if not len(points):
        return np.empty(0, dtype=bool)
    xy = np.floor(points).astype(np.int64)
    valid = ((xy >= 0).all(axis=1) & (xy[:, 0] < mask.shape[1]) & (xy[:, 1] < mask.shape[0]))
    result = np.zeros(len(points), dtype=bool)
    result[valid] = mask[xy[valid, 1], xy[valid, 0]]
    return result


@pipeline_stage('mask_validation')
def check_mask(mask, score, prompts, foreign, box, settings, size, *, observe=None):
    """Return an admitted mask/box or an explicit, auditable rejection reason."""
    import cv2
    width, height = size
    if mask.dtype != np.bool_ or mask.shape != (height, width):
        raise ValueError('SAM-pose2seg mask must match the full source frame.')
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('SAM-pose2seg returned an invalid mask score.')
    if observe is not None:
        observe('raw', mask)
    if score < settings['min_mask_score']:
        return None, None, 'low_mask_score'
    if not mask.any():
        return None, None, 'empty_mask'
    # Remove tiny disconnected speckles without throwing away a disconnected
    # but substantial hand/foot. Never force a single connected component.
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= max(4, math.ceil(int(mask.sum()) * .001))
    keep[0] = False
    mask = keep[labels]
    if observe is not None:
        observe('cleaned', mask)
    if not mask.any():
        return None, None, 'empty_mask'
    if inside(mask, prompts).mean() < settings['min_prompt_coverage']:
        return None, None, 'missing_prompt_support'
    if len(foreign) and inside(mask, foreign).mean() > settings['max_foreign_coverage']:
        return None, None, 'foreign_core_in_mask'
    y, x = np.nonzero(mask)
    mask_box = [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]
    union = [min(box[0], mask_box[0]), min(box[1], mask_box[1]),
             max(box[2], mask_box[2]), max(box[3], mask_box[3])]
    w, h = box[2] - box[0], box[3] - box[1]
    ratio = (union[2]-union[0]) * (union[3]-union[1]) / (w*h)
    expansion = max((box[0]-union[0])/w, (box[1]-union[1])/h,
                    (union[2]-box[2])/w, (union[3]-box[3])/h)
    if ratio > settings['max_box_area_ratio'] or expansion > settings['max_side_expansion']:
        return None, None, 'excessive_expansion'
    # Expansion only: no shrink around a visible torso when legs are occluded.
    return mask, union if settings['expand_boxes'] else list(box), None


def spread_indices(points, indices, limit):
    """Keep the most visible seed, then cover distinct regions and distant landmarks."""
    if limit <= 1 or len(indices) <= 1:
        return np.asarray(indices[:limit], dtype=np.intp)
    chosen = [indices[0]]
    remaining = list(indices[1:])

    def region(k):
        return 0 if k < 5 else (1 if k < 11 else 2)  # head, shoulders, hips

    while remaining and len(chosen) < limit:
        regions = {region(k) for k in chosen}
        # Input is visibility ordered; max keeps that order on equal distances.
        next_point = max(remaining, key=lambda k: (
            region(k) not in regions,
            np.sum((points[chosen] - points[k]) ** 2, axis=1).min()))
        chosen.append(next_point)
        remaining.remove(next_point)
    return np.asarray(chosen, dtype=np.intp)


@dataclass
class Candidate:
    bits: np.ndarray
    pixels: int
    box: list
    indices: np.ndarray


def shared_masks(candidates, neighbours):
    shared = set()
    for i, candidate in candidates.items():
        for j in neighbours[i]:
            if j >= i or j not in candidates:
                continue
            other = candidates[j]
            intersection = np.unpackbits(candidate.bits & other.bits).sum()
            if intersection > .15 * min(candidate.pixels, other.pixels):
                shared.update((i, j))
    return shared


def select_masks(owner, image, bounds, points, cores, jobs, records, initial_masks):
    settings = owner.settings
    minimum = settings['min_prompt_points']
    active = list(jobs)
    candidates, neighbours = {}, {}
    blocked = set()
    conflicts = set()
    rejected = set()
    packed_size = (image.width * image.height + 7) // 8
    supplied = iter(initial_masks) if initial_masks is not None else None
    with ExitStack() as stack:
        frame_open = False
        while active:
            if supplied is None and not frame_open:
                stack.enter_context(owner.segmenter.frame(image))
                frame_open = True
            for i, available, foreign in active:
                record = records[i]
                attempt = record.get('mask_attempts', 0) + 1
                indices = available[:minimum + attempt - 1]
                positive, peers, negative = owner._prompts(bounds, points, cores, i, indices)
                neighbours[i] = peers
                if supplied is not None:
                    mask, score = next(supplied)
                elif negative:
                    mask, score = owner.segmenter.predict(positive, negative)
                else:
                    mask, score = owner.segmenter.predict(positive)
                record.update(status='rejected', mask_score=score, mask_attempts=attempt,
                              prompt_indices=indices.tolist(), prompt_points=positive.tolist())
                if peers:
                    record['negative_points'] = np.asarray(negative).tolist()
                mask, box, reason = check_mask(mask, score, positive, foreign, bounds[i], settings,
                                              image.size, observe=mask_observer(image, record))
                record['reason'] = reason
                record.pop('mask_pixels', None)
                record['separated'] = False
                if reason:
                    rejected.add(i)
                    # A previously conflicting candidate still blocks its peer until the
                    # peer moves clear. It is never submitted after this failed retry.
                    continue
                rejected.discard(i)
                pixels = record['mask_pixels'] = int(mask.sum())
                if not peers:
                    yield i, box, mask, indices
                    continue
                torso = [k for k in cores[i] if k in (5, 6, 11, 12)]
                record['separated'] = bool(len(torso) >= 2 and inside(mask, points[i, torso]).all()
                    and all(len(cores[j]) >= 2 and not inside(mask, points[j, cores[j]]).any() for j in peers))
                if i not in candidates and (len(candidates) + 1) * packed_size > 16 * 1024**2:
                    blocked.update([i, *peers])
                    continue
                candidates[i] = Candidate(np.packbits(mask, axis=None), pixels, box, indices)
            supplied = None
            conflicts = shared_masks(candidates, neighbours) | blocked
            active = [(i, available, foreign) for i, available, foreign in jobs
                      if i not in blocked and (i in rejected or i in conflicts)
                      and len(records[i]['prompt_indices']) < len(available)]

    for i in conflicts:
        records[i].update(status='rejected', reason='shared_mask_pixels', mask_conflict=True, separated=False)
    for i, candidate in candidates.items():
        if not records[i]['reason']:
            mask = np.unpackbits(candidate.bits, count=image.width * image.height)
            yield i, candidate.box, mask.reshape(image.height, image.width).view(np.bool_), candidate.indices
