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
def check_mask(mask, score, prompts, foreign, box, settings, size, *, observe=None, own=()):
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
    if len(own) and len(foreign) and len(stats) > 2:
        def components(points):
            points = np.asarray(points).reshape(-1, 2)
            xy = np.floor(points[inside(mask, points)]).astype(int)
            return labels[xy[:,1], xy[:,0]]
        ours = components(np.concatenate((prompts, own)))
        theirs = components(foreign)
        # Only remove independently supported foreign islands. A connection to
        # any measured target joint protects a disconnected hand or foot.
        if len(ours) and len(theirs):
            for component in set(theirs) - set(ours):
                if np.count_nonzero(theirs == component) >= 2:
                    keep[component] = False
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


def spread_indices(points, indices, limit, *, initial=1):
    """Keep the initial visibility-ranked seeds, then add distinct regions and distant landmarks."""
    if limit <= initial or len(indices) <= initial:
        return np.asarray(indices[:limit], dtype=np.intp)
    chosen = list(indices[:initial])
    remaining = list(indices[initial:])

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
    extent: tuple
    pixels: int
    box: list
    indices: np.ndarray

    def crop(self):
        x,y,r,b = self.extent
        return np.unpackbits(self.bits, count=(r-x)*(b-y)).reshape(b-y,r-x).view(bool)

    def hits(self, points):
        return inside(self.crop(), points - self.extent[:2])


def shared_masks(candidates, evidence):
    shared = set()
    for i, candidate in candidates.items():
        for j, other in candidates.items():
            if j >= i:
                continue
            ax,ay,ar,ab = candidate.extent
            bx,by,br,bb = other.extent
            x,y,r,b = max(ax,bx),max(ay,by),min(ar,br),min(ab,bb)
            if x >= r or y >= b:
                continue
            a = candidate.crop()[y-ay:b-ay, x-ax:r-ax]
            d = other.crop()[y-by:b-by, x-bx:r-bx]
            if np.count_nonzero(a & d) <= .15 * min(candidate.pixels, other.pixels):
                continue
            own_i, own_j = evidence[i], evidence[j]
            def decisive(winner, rival, own, foreign):
                covered = winner.hits(own)
                return (covered.sum() >= 2 and not winner.hits(foreign).any()
                        and (covered & rival.hits(own)).sum() >= 2)
            if decisive(candidate, other, own_i, own_j):
                shared.add(j)
            elif decisive(other, candidate, own_j, own_i):
                shared.add(i)
            else:
                shared.update((i,j))
    return shared


def select_masks(owner, image, bounds, points, cores, jobs, records, initial_masks, evidence, decoder=None):
    settings = owner.settings
    minimum = settings['min_prompt_points']
    active = list(jobs)
    candidates = {}
    blocked = set()
    conflicts = set()
    rejected = set()
    adaptive = settings.get('adaptive_decoder_retries', False)
    leaks = {}
    packed_bytes = 0
    supplied = iter(initial_masks) if initial_masks is not None else None
    with ExitStack() as stack:
        frame_open = False
        while active:
            if supplied is None and decoder is None and not frame_open:
                stack.enter_context(owner.segmenter.frame(image))
                frame_open = True
            for i, available, foreign in active:
                record = records[i]
                attempt = record.get('mask_attempts', 0) + 1
                count = min(settings['max_prompt_points'], record.get('prompt_minimum', minimum) + attempt - 1)
                indices = available[:count]
                if adaptive and attempt > 1 and record.get('reason') in (
                        'missing_prompt_support', 'foreign_core_in_mask', 'missing_independent_support'):
                    unused = [k for k in available if k not in record['prompt_indices']]
                    kept = [k for k in available if k in record['prompt_indices']]
                    if unused:
                        indices = np.asarray((unused + kept)[:count], dtype=np.intp)
                positive, peers, negative = owner._prompts(bounds, points, cores, i, indices, leaks.get(i))
                if supplied is not None:
                    mask, score = next(supplied)
                else:
                    multi = adaptive and attempt > 1
                    options = getattr(owner.segmenter, 'predict_options', None)
                    if decoder is not None:
                        variants = decoder(positive, negative, multi)
                    elif multi and options:
                        variants = options(positive, negative)
                    else:
                        variants = [owner.segmenter.predict(positive, negative) if negative
                                    else owner.segmenter.predict(positive)]
                    def rank(variant):
                        raw, quality = variant
                        clean, _, reason = check_mask(raw, quality, positive, foreign, bounds[i],
                                                      settings, image.size, own=evidence[i])
                        target = raw if clean is None else clean
                        return (reason is None, -int(inside(target, foreign).sum()),
                                int(inside(target, evidence[i]).sum()), quality)
                    mask, score = max(variants, key=rank) if len(variants)>1 else variants[0]
                    if multi:
                        record['retry_mask_choices'] = len(variants)
                if adaptive:
                    leaks[i] = {j: set(cores[j][inside(mask, points[j, cores[j]])]) for j in peers}
                record.update(status='rejected', mask_score=score, mask_attempts=attempt,
                              prompt_indices=indices.tolist(), prompt_points=positive.tolist())
                if peers:
                    record['negative_points'] = np.asarray(negative).tolist()
                mask, box, reason = check_mask(mask, score, positive, foreign, bounds[i], settings,
                                              image.size, observe=mask_observer(image, record), own=evidence[i])
                holdout = [k for k in record.get('support_indices', ()) if k not in indices]
                if reason is None and holdout and not inside(mask, points[i, holdout]).any():
                    reason = 'missing_independent_support'
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
                if len(bounds) == 1:
                    yield i, box, mask, indices
                    continue
                torso = [k for k in cores[i] if k in (5, 6, 11, 12)]
                record['separated'] = bool(peers and len(torso) >= 2 and inside(mask, points[i, torso]).all()
                    and all(len(cores[j]) >= 2 and not inside(mask, points[j, cores[j]]).any() for j in peers))
                yy, xx = np.nonzero(mask)
                x,y,r,b = int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1
                bits = np.packbits(mask[y:b,x:r], axis=None)
                previous_bytes = candidates[i].bits.nbytes if i in candidates else 0
                if packed_bytes - previous_bytes + bits.nbytes > 16 * 1024**2:
                    # Without the candidate we cannot disambiguate a conflicting peer.
                    blocked.update(j for j, other in enumerate(bounds) if
                        other[0] < r and x < other[2] and other[1] < b and y < other[3])
                    blocked.add(i)
                    continue
                packed_bytes += bits.nbytes - previous_bytes
                candidates[i] = Candidate(bits, (x,y,r,b), pixels, box, indices)
            supplied = None
            conflicts = shared_masks(candidates, evidence) | blocked
            active = [(i, available, foreign) for i, available, foreign in jobs
                      if i not in blocked and (i in rejected or i in conflicts)
                      and ((adaptive and records[i]['mask_attempts'] < 3) or
                           (not adaptive and len(records[i]['prompt_indices']) < len(available)))]

    for i in conflicts:
        records[i].update(status='rejected', reason='shared_mask_pixels', mask_conflict=True, separated=False)
    for i, candidate in candidates.items():
        if not records[i]['reason']:
            x,y,r,b = candidate.extent
            mask = np.zeros((image.height,image.width), bool)
            mask[y:b,x:r] = candidate.crop()
            yield i, candidate.box, mask, candidate.indices
