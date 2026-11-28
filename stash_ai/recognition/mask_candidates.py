"""Select SAM prompts and settle mask ownership before running masked PMPose."""
from contextlib import ExitStack
from contextlib import contextmanager
from contextvars import ContextVar
from collections import OrderedDict
from dataclasses import dataclass, field
import math

import numpy as np

from .mask_cleanup import clean_mask, inside
from .pose_mask_capture import mask_observer, mask_history
from .profiling import pipeline_stage

_decoded=ContextVar('decoded_pose_masks',default=None)


@contextmanager
def decoded_masks(limit=8*1024**2):
    if _decoded.get() is not None:
        yield
        return
    token=_decoded.set([OrderedDict(),0,limit])
    try:yield
    finally:_decoded.reset(token)


@pipeline_stage('mask_validation')
def check_mask(mask, score, prompts, foreign, box, settings, size, *, observe=None, own=(), limbs=(),
               edits=None, replay=None, cuts=None, replay_cuts=None, details=None):
    """Return an admitted mask/box or an explicit, auditable rejection reason."""
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
    mask = clean_mask(mask, box, settings, prompts=prompts, own=own, foreign=foreign, limbs=limbs,
                      edits=edits, replay=replay, cuts=cuts, replay_cuts=replay_cuts, details=details)
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
    edits: dict = field(default_factory=dict)

    def crop(self):
        pool=_decoded.get(); key=id(self)
        if pool is not None and key in pool[0]:
            pool[0].move_to_end(key)
            return pool[0][key][1]
        x,y,r,b = self.extent
        result=np.unpackbits(self.bits, count=(r-x)*(b-y)).reshape(b-y,r-x).view(bool)
        result.flags.writeable=False
        if pool is not None and result.nbytes<=pool[2]:
            while pool[0] and pool[1]+result.nbytes>pool[2]:
                _,(_,old)=pool[0].popitem(last=False);pool[1]-=old.nbytes
            pool[0][key]=(self,result);pool[1]+=result.nbytes
        return result

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


TORSO_REGIONS = {'missing_hip_region': (11, 12), 'missing_shoulder_region': (5, 6)}


def regional_seeds(points, indices, temporal):
    ordered = list(spread_indices(points, indices, len(indices)))
    if len(ordered) < 2:
        return np.asarray(ordered, dtype=np.intp)
    opposite = (5, 6) if ordered[0] in (11, 12) else (11, 12)
    second = next((k for k in ordered if k in opposite), None)
    if second is not None:
        ordered.remove(second)
        ordered.insert(1, second)
    # A current, locally corroborated limb/face can be an extra prompt after
    # the initial torso span. Keep the user's existing per-attempt budget.
    for k, _ in temporal:
        if k in ordered[2:] and k not in (5, 6, 11, 12):
            ordered.remove(k)
            ordered.insert(2, k)
            break
    return np.asarray(ordered, dtype=np.intp)


def _assessment(raw, score, i, indices, positive, foreign, bounds, peers, points,
                reliable, evidence, record, settings, size, limbs, observe=None):
    box = bounds[i]
    island_edits, bridge_cuts = [], []
    clean, crop, reason = check_mask(raw, score, positive, foreign, box, settings, size,
                                     observe=observe, own=evidence[i], limbs=limbs, edits=island_edits, cuts=bridge_cuts)
    edits = {}
    if island_edits:
        edits['component_edits'] = island_edits
    if bridge_cuts:
        edits['bridge_cuts'] = bridge_cuts
    target = raw if clean is None else clean
    temporal = {k for k, _ in record.get('temporal_prompts', ())}
    if (settings.get('recover_prompt_points') and
            reason in (None, 'missing_prompt_support')):
        support = set(record.get('support_indices', ())) | (temporal & set(indices))
        for missing, group in TORSO_REGIONS.items():
            witnesses = [k for k in group if k in support and
                         (not len(foreign) or np.linalg.norm(foreign-points[i,k],axis=1).min() > 3)]
            if witnesses and not inside(target, points[i, witnesses]).any():
                reason = missing
                break
    if reason is None:
        holdout = [k for k in record.get('support_indices', ()) if k not in indices]
        if holdout and not inside(clean, points[i, holdout]).any():
            reason = 'missing_independent_support'
        outside = clean.sum() != clean[box[1]:box[3], box[0]:box[2]].sum()
        sparse = (record.get('prompt_minimum', settings['min_prompt_points']) < settings['min_prompt_points']
                  or any(k in (7,8,13,14) for k in indices)
                  or 'face_prompt' in record or any(k in temporal for k in indices))
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
                     box, indices, variant, float(score), edits)


def resolve_options(candidates, options, evidence, maximum=256):
    """Choose complete interacting components; independent masks share no budget."""
    with decoded_masks():
        return _resolve_components(candidates,options,evidence,maximum)


def _resolve_components(candidates,options,evidence,maximum):
    domains={i:[c,*options.get(i,())] for i,c in candidates.items()}
    pending=set(domains)
    def connected(i,j):
        return any(a.extent[0]<b.extent[2] and b.extent[0]<a.extent[2] and
                   a.extent[1]<b.extent[3] and b.extent[1]<a.extent[3]
                   for a in domains[i] for b in domains[j])
    while pending:
        component={min(pending)};pending-=component;front=list(component)
        while front:
            i=front.pop(); neighbours={j for j in pending if connected(i,j)}
            component.update(neighbours);pending-=neighbours;front.extend(neighbours)
        subset={i:candidates[i] for i in sorted(component)}
        _resolve_group(subset,options,evidence,maximum)
        candidates.update(subset)
    return shared_masks(candidates,evidence)


def _resolve_group(candidates, options, evidence, maximum):
    from itertools import product
    keys = sorted(candidates)
    domains = [[candidates[i], *options.get(i,())] for i in keys]
    if math.prod(map(len,domains)) > maximum:
        return shared_masks(candidates,evidence)
    best = None
    selected = None
    pair_cache={}
    for choices in product(*domains):
        proposal = dict(zip(keys,choices))
        conflicts=set()
        for n,i in enumerate(keys):
            for j in keys[:n]:
                key=id(proposal[i]),id(proposal[j])
                if key not in pair_cache:
                    pair_cache[key]=shared_masks({i:proposal[i],j:proposal[j]},evidence)
                conflicts.update(pair_cache[key])
        rank = (-len(conflicts),sum(int(c.hits(evidence[i]).sum()) for i,c in proposal.items()),
                sum(c.quality for c in choices))
        if best is None or rank > best:
            best,selected = rank,proposal
    if selected is not None:
        candidates.update(selected)
    return shared_masks(candidates,evidence)


def select_masks(owner, image, bounds, points, cores, jobs, records, initial_masks, evidence, decoder=None, valid=None):
    from . import detection_phases as phases
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
    instance_source = decoder if callable(getattr(decoder, 'iter_instances', None)) else None
    with ExitStack() as stack:
        stack.enter_context(decoded_masks(min(32*1024**2,getattr(owner,'queue_bytes',32*1024**2)//4)))
        frame_open = False
        while active:
            if supplied is None and decoder is None and not frame_open:
                frame = stack.enter_context(owner.segmenter.frame(image))
                instance_source = frame if callable(getattr(frame, 'iter_instances', None)) else None
                frame_open = True
            for i, available, foreign in active:
                record = records[i]
                attempt = record.get('mask_attempts', 0) + 1
                count = min(settings['max_prompt_points'], record.get('prompt_minimum', minimum) + attempt - 1)
                indices = available[:count]
                missing_region = TORSO_REGIONS.get(record.get('reason'))
                if attempt > 1 and missing_region:
                    # Try the previously unused side of the omitted region first.
                    regional = [k for k in available if k in missing_region]
                    regional.sort(key=lambda k: k in record['prompt_indices'])
                    if regional:
                        target = regional[0]
                        indices = np.asarray(([target] + [k for k in available if k != target])[:count],
                                             dtype=np.intp)
                elif adaptive and attempt > 1 and record.get('reason') in (
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
                reliable = np.flatnonzero(valid[i]) if valid is not None else cores[i]
                limbs = [(points[i,p],points[i,k]) for p,k in ((7,9),(8,10),(13,15),(14,16))
                         if p in reliable and k in reliable]
                def assess(variant, observe=None):
                    return _assessment(*variant, i, indices, positive, foreign, bounds, peers, points,
                                       reliable, evidence, record, settings, image.size, limbs, observe)
                if instance_source is not None:
                    if supplied is not None:
                        next(supplied)
                    if attempt > 1:
                        from .refinement_budget import budget
                        if not budget(image, settings).admit('sam_decoder', record=record):
                            exhausted.add(i)
                            continue
                    from .instance_mask_candidates import select_instances
                    variants, assessed, count_evaluated = select_instances(instance_source, assess, positive, negative)
                    record['mask_candidates_evaluated'] = count_evaluated
                    if attempt > 1:
                        record['retry_mask_choices'] = len(variants)
                elif supplied is not None:
                    variants = [next(supplied)]
                else:
                    if attempt>1:
                        from .refinement_budget import budget
                        if not budget(image,settings).admit('sam_decoder',record=record):
                            exhausted.add(i)
                            continue
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
                if instance_source is None:
                    assessed = [assess(item) for item in variants]
                order = sorted(range(len(variants)), key=lambda v: assessed[v][0], reverse=True)
                best = order[0]
                raw, score = variants[best]
                history = phases.active(image)
                if history is not None:
                    from .tune_mask_candidate import record as record_sample
                    record_sample(history, raw, score, f'P{i} · attempt {attempt} · candidate {best}',
                                  bounds[i], positive, evidence[i], foreign, limbs, (i, attempt))
                observer = mask_observer(image, record)
                rank, mask, box, reason, edits = assessed[best]
                if observer is not None:
                    mask_history(image,record).append(dict(attempt=attempt,
                        positive=positive.tolist(), negative=np.asarray(negative).tolist(), chosen=best,
                        variants=[dict(score=float(item[1]), reason=assessment[3],
                                       **({'instance': assessment[4]['mask_instance']}
                                          if 'mask_instance' in assessment[4] else {}))
                                  for item, assessment in zip(variants, assessed)]))
                if adaptive:
                    leaks[i] = {j: set(cores[j][inside(raw, points[j, cores[j]])])
                                for j in range(len(bounds)) if j != i}
                record.update(status='rejected', mask_score=float(score), mask_attempts=attempt,
                              prompt_indices=indices.tolist(), prompt_points=positive.tolist(),
                              mask_variant=best)
                record.pop('component_edits', None)
                record.pop('bridge_cuts', None)
                record.pop('mask_instance', None)
                record.update(edits)
                if peers:
                    record['negative_points'] = np.asarray(negative).tolist()
                record['reason'] = reason
                record.pop('mask_pixels', None)
                record['separated'] = False
                if observer is not None or (history is not None and not history.samples_only):
                    def observe(stage, pixels):
                        if observer is not None:
                            observer(stage, pixels)
                        title = f'raw candidate {best}' if stage == 'raw' else 'after cleanup / validation'
                        phases.mask(image, f'Mask attempt {attempt}: {title}', i, pixels, record,
                                    positive=positive, negative=negative, rejected=bool(reason))
                    # Reuse the audit observer: rejected candidates can still
                    # have a cleaned mask even though assessment returns None.
                    assess(variants[best], observe)
                if reason:
                    rejected.add(i)
                    # A previously conflicting candidate still blocks its peer until the
                    # peer moves clear. It is never submitted after this failed retry.
                    continue
                rejected.discard(i)
                from .mask_uncertainty import boundary_fraction
                record['mask_boundary_fraction'] = boundary_fraction(mask)
                pixels = record['mask_pixels'] = int(mask.sum())
                if len(bounds) == 1:
                    phases.mask(image, 'Selected mask for pose', i, mask, record, rejected=False)
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
            conflicts = resolve_options(candidates, options_by_person, evidence, settings.get('max_mask_combinations',256)) | blocked
            active = [(i, available, foreign) for i, available, foreign in jobs
                      if i not in blocked and i not in exhausted and (i in rejected or i in conflicts)
                      and ((adaptive and records[i]['mask_attempts'] < settings.get('max_decoder_attempts',3)) or
                           (not adaptive and len(records[i]['prompt_indices']) < len(available)))]

    for i in conflicts:
        records[i].update(status='rejected', reason='shared_mask_pixels', mask_conflict=True, separated=False)
    if candidates or conflicts:
        phases.notes(image, 'Cross-person mask ownership / conflicts', records)
    for i, candidate in candidates.items():
        if not records[i]['reason']:
            x,y,r,b = candidate.extent
            mask = np.zeros((image.height,image.width), bool)
            mask[y:b,x:r] = candidate.crop()
            from .mask_uncertainty import boundary_fraction
            records[i]['mask_boundary_fraction']=boundary_fraction(mask)
            if records[i]['mask_variant'] != candidate.variant:
                records[i].update(mask_variant=candidate.variant, mask_score=candidate.quality,
                                  mask_pixels=candidate.pixels, separated=False)
                records[i].pop('component_edits', None)
                records[i].pop('bridge_cuts', None)
                records[i].update(candidate.edits)
                observer = mask_observer(image, records[i])
                if observer is not None:
                    observer('cleaned', mask)
                    mask_history(image,records[i])[-1]['paired_choice'] = candidate.variant
            peers = [j for j in range(len(bounds)) if j != i and
                     bounds[j][0] < candidate.box[2] and candidate.box[0] < bounds[j][2]
                     and bounds[j][1] < candidate.box[3] and candidate.box[1] < bounds[j][3]]
            torso = [k for k in cores[i] if k in (5,6,11,12)]
            records[i]['separated'] = bool(peers and len(torso) >= 2
                and inside(mask, points[i,torso]).all()
                and all(len(cores[j]) >= 2 and not inside(mask, points[j,cores[j]]).any() for j in peers))
            phases.mask(image, 'Selected mask after cross-person selection', i, mask, records[i], rejected=False)
            yield i, candidate.box, mask, candidate.indices
