"""Select SAM prompts and settle mask ownership before running masked PMPose."""
from contextlib import ExitStack
from dataclasses import dataclass
import math

import numpy as np

from .pose_mask_capture import mask_observer, mask_history
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
def check_mask(mask, score, prompts, foreign, box, settings, size, *, observe=None, own=(), limbs=(), edits=None, replay=()):
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
    original_keep = keep.copy()
    if len(stats) > 2:
        def components(points):
            points = np.asarray(points).reshape(-1, 2)
            xy = np.floor(points[inside(mask, points)]).astype(int)
            return labels[xy[:, 1], xy[:, 0]]

        theirs = components(foreign)
        for parent, tip in limbs:
            # The visible chain must approach the island from our foreground.
            segment = np.linspace(parent, tip, 9)
            island = components([tip])
            if (len(island) and island[0] not in theirs and
                    inside(mask, segment[:-1]).sum() >= 6 and
                    (not len(foreign) or np.linalg.norm(foreign - tip, axis=1).min() > 3)):
                keep[island[0]] = True
        if len(own) and len(theirs):
            ours = components(np.concatenate((prompts, own)))
            for component in set(theirs) - set(ours):
                if len(ours) and np.count_nonzero(theirs == component) >= 2:
                    keep[component] = False
    for x, y, keep_island in replay:
        if 0 <= x < width and 0 <= y < height and labels[y,x]:
            keep[labels[y,x]] = keep_island
    if edits is not None:
        for label in np.flatnonzero(keep != original_keep):
            y, x = np.argwhere(labels == label)[0]
            edits.append((int(x), int(y), bool(keep[label])))
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

    variant: int = 0
    quality: float = 0.0
    edits: tuple = ()

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


def _regions(indices, xy, distance):
    # Count one spatially independent point per anatomical group.
    group = (0,0,0,0,0,1,1,2,3,2,3,4,4,5,6,5,6)
    groups, witnesses = set(), []
    for k in indices:
        if group[k] not in groups and all(np.linalg.norm(xy[k]-p) >= distance for p in witnesses):
            groups.add(group[k])
            witnesses.append(xy[k])
    return len(groups)


def _assessment(raw, score, i, indices, positive, foreign, bounds, peers, points,
                reliable, evidence, record, settings, size, limbs, observe=None):
    box = bounds[i]
    edits = []
    clean, crop, reason = check_mask(raw, score, positive, foreign, box, settings, size,
                                     observe=observe, own=evidence[i], limbs=limbs, edits=edits)
    target = raw if clean is None else clean
    if reason is None:
        holdout = [k for k in record.get('support_indices', ()) if k not in indices]
        if holdout and not inside(clean, points[i, holdout]).any():
            reason = 'missing_independent_support'
        outside = clean.sum() != clean[box[1]:box[3], box[0]:box[2]].sum()
        sparse = (record.get('prompt_minimum', settings['min_prompt_points']) < settings['min_prompt_points']
                  or any(k in (7,8,13,14) for k in indices)
                  or 'face_prompt' in record or 'temporal_prompts' in record)
        # A plausible alternative can stay inside the detector box while following
        # only one side of the skeleton into a contacting partner.
        body = [k for k in reliable if k >= 5]
        unilateral = len(body) >= 3 and len({k % 2 for k in body}) == 1
        weak_expansion = peers and (outside or unilateral) and not inside(clean, points[i, holdout]).any()
        floor = settings.get('min_recovery_mask_score')
        if reason is None and (sparse or weak_expansion) and score < (
                floor if floor is not None else (1 + settings['min_mask_score']) / 2):
            scale = np.hypot(box[2]-box[0], box[3]-box[1])
            own = [k for k in reliable if inside(clean, [points[i,k]])[0]]
            independent = [k for k in own if k not in indices and
                           np.linalg.norm(positive-points[i,k], axis=1).min() > max(3., .06*scale)]
            # Overriding a user-specified recovery floor is never permitted.
            bilateral_core = {5,6}.issubset(own) or {11,12}.issubset(own)
            supported = (floor is None and independent and bilateral_core and
                         _regions(own, points[i], max(3., .09*scale)) >= 3 and
                         not inside(clean, foreign).any())
            if not supported:
                reason = 'weak_recovery_mask'
    rank = (reason is None, -int(inside(target, foreign).sum()),
            int(inside(target, evidence[i]).sum()), float(score))
    return rank, clean, crop, reason, edits


def _pack(mask, box, indices, variant, score, edits):
    yy, xx = np.nonzero(mask)
    x,y,r,b = int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1
    return Candidate(np.packbits(mask[y:b,x:r], axis=None), (x,y,r,b), len(xx),
                     box, indices, variant, float(score), tuple(edits))


def resolve_options(candidates, options, evidence):
    """Try existing decoder variants for geometrically conflicting local pairs."""
    conflicts = shared_masks(candidates, evidence)
    for i in sorted(conflicts):
        for j in candidates:
            if i == j or not (options.get(i) or options.get(j)):
                continue
            if not shared_masks({i:candidates[i], j:candidates[j]}, evidence):
                continue
            for first in [candidates[i], *options.get(i, ())]:
                for second in [candidates[j], *options.get(j, ())]:
                    proposal = dict(candidates)
                    proposal.update({i:first, j:second})
                    remaining = shared_masks(proposal, evidence)
                    if remaining < conflicts:
                        candidates.update({i:first, j:second})
                        conflicts = remaining
                        break
                else:
                    continue
                break
    return conflicts


def select_masks(owner, image, bounds, points, cores, jobs, records, initial_masks, evidence, decoder=None, valid=None):
    settings = owner.settings
    minimum = settings['min_prompt_points']
    active = list(jobs)
    candidates = {}
    options_by_person = {}
    blocked = set()
    conflicts = set()
    rejected = set()
    adaptive = settings.get('adaptive_decoder_retries', False)
    leaks, tried, exhausted = {}, {}, set()
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
                if adaptive:
                    signature = (tuple(indices), tuple(map(tuple, negative)), attempt > 1)
                    previous = tried.setdefault(i, set())
                    if signature in previous:
                        exhausted.add(i)
                        continue
                    previous.add(signature)
                if supplied is not None:
                    variants = [next(supplied)]
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
                    if multi:
                        record['retry_mask_choices'] = len(variants)
                reliable = np.flatnonzero(valid[i]) if valid is not None else cores[i]
                limbs = [(points[i,p],points[i,k]) for p,k in ((7,9),(8,10),(13,15),(14,16))
                         if p in reliable and k in reliable]
                def assess(variant, observe=None):
                    return _assessment(*variant, i, indices, positive, foreign, bounds, peers, points,
                                       reliable, evidence, record, settings, image.size, limbs, observe)
                assessed = [assess(item) for item in variants]
                order = sorted(range(len(variants)), key=lambda v: assessed[v][0], reverse=True)
                best = order[0]
                raw, score = variants[best]
                observer = mask_observer(image, record)
                rank, mask, box, reason, edits = assessed[best]
                if observer is not None:
                    assess(variants[best], observer)
                    mask_history(image,record).append(dict(attempt=attempt,
                        positive=positive.tolist(), negative=np.asarray(negative).tolist(), chosen=best,
                        variants=[dict(score=float(item[1]), reason=assessment[3])
                                  for item, assessment in zip(variants, assessed)]))
                if adaptive:
                    leaks[i] = {j: set(cores[j][inside(raw, points[j, cores[j]])])
                                for j in range(len(bounds)) if j != i}
                record.update(status='rejected', mask_score=float(score), mask_attempts=attempt,
                              prompt_indices=indices.tolist(), prompt_points=positive.tolist(),
                              mask_variant=best)
                record.pop('component_edits', None)
                if edits:
                    record['component_edits'] = edits
                if peers:
                    record['negative_points'] = np.asarray(negative).tolist()
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
                candidate = _pack(mask, box, indices, best, score, edits)
                packed_bytes -= sum(item.bits.nbytes for item in options_by_person.pop(i, []))
                previous_bytes = candidates[i].bits.nbytes if i in candidates else 0
                if packed_bytes - previous_bytes + candidate.bits.nbytes > 16 * 1024**2:
                    x,y,r,b = candidate.extent
                    blocked.update(j for j, other in enumerate(bounds) if
                        other[0] < r and x < other[2] and other[1] < b and y < other[3])
                    blocked.add(i)
                    continue
                packed_bytes += candidate.bits.nbytes - previous_bytes
                candidates[i] = candidate
                for v in order[1:]:
                    _, alt_mask, alt_box, failure, alt_edits = assessed[v]
                    if failure is None:
                        alt = _pack(alt_mask, alt_box, indices, v, variants[v][1], alt_edits)
                        if packed_bytes + alt.bits.nbytes <= 16 * 1024**2:
                            options_by_person.setdefault(i, []).append(alt)
                            packed_bytes += alt.bits.nbytes
            supplied = None
            conflicts = resolve_options(candidates, options_by_person, evidence) | blocked
            active = [(i, available, foreign) for i, available, foreign in jobs
                      if i not in blocked and i not in exhausted and (i in rejected or i in conflicts)
                      and ((adaptive and records[i]['mask_attempts'] < 3) or
                           (not adaptive and len(records[i]['prompt_indices']) < len(available)))]

    for i in conflicts:
        records[i].update(status='rejected', reason='shared_mask_pixels', mask_conflict=True, separated=False)
    for i, candidate in candidates.items():
        if not records[i]['reason']:
            x,y,r,b = candidate.extent
            mask = np.zeros((image.height,image.width), bool)
            mask[y:b,x:r] = candidate.crop()
            if records[i]['mask_variant'] != candidate.variant:
                records[i].update(mask_variant=candidate.variant, mask_score=candidate.quality,
                                  mask_pixels=candidate.pixels, separated=False, component_edits=list(candidate.edits))
                observer = mask_observer(image, records[i])
                if observer is not None:
                    observer('cleaned', mask)
                    mask_history(image,records[i])[-1]['paired_choice'] = candidate.variant
            yield i, candidate.box, mask, candidate.indices
