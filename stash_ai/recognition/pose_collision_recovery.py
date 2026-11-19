"""Opt-in recovery of copied pose fragments before duplicate suppression.

Retries preserve detector coordinates. A collision is a reason to inspect a
candidate, never evidence that either detection is a duplicate person.
"""
import numpy as np
from PIL import Image, ImageDraw

CORE = np.array([0, 1, 2, 3, 4, 5, 6, 11, 12])
CHAINS = ((5, 7, 9), (6, 8, 10), (11, 13, 15), (12, 14, 16))
TORSO = (5, 6, 11, 12)
NEUTRAL = np.array([123.675, 116.28, 103.53], dtype=np.float32)


def inside(points, box):
    return ((points >= box[..., :2]) & (points < box[..., 2:])).all(axis=-1)


def reliable(prediction, bounds, settings):
    xy, score, prob = prediction
    valid = (inside(xy, np.asarray(bounds)[:, None, :]) &
             (score > settings['min_keypoint_score']) &
             (prob['presence'] > settings['min_presence']) &
             (prob['visibility'] > settings['min_visibility']))
    if 'flip_agreement' in prob:
        valid &= prob['flip_agreement'] >= .5
    return valid


def collisions(bounds, prediction, settings):
    """Require anatomical chain/torso agreement, not one touching joint."""
    xy = prediction[0]
    boxes = np.asarray(bounds, dtype=float)
    valid = reliable(prediction, boxes, settings)
    for a in range(len(boxes)):
        for b in range(a + 1, len(boxes)):
            lo, hi = np.maximum(boxes[a, :2], boxes[b, :2]), np.minimum(boxes[a, 2:], boxes[b, 2:])
            area = np.maximum(0, hi-lo).prod()
            if area <= 0:
                continue
            areas = np.prod(boxes[[a, b], 2:]-boxes[[a, b], :2], axis=1)
            overlap = area / areas.min()
            iou = area / (areas.sum()-area)
            if overlap < settings['partial_min_overlap'] or iou >= settings['contact_max_iou']:
                continue
            tolerance = max(2., settings['partial_point_distance'] *
                            np.linalg.norm(boxes[[a, b], 2:]-boxes[[a, b], :2], axis=1).min())
            shared = valid[a] & valid[b] & (np.linalg.norm(xy[a]-xy[b], axis=1) <= tolerance)
            chains = [chain for chain in (*CHAINS, TORSO)
                      if shared[list(chain)].sum() >= settings['partial_min_shared_points']]
            full = shared[CORE].sum() >= 4
            if not full and not chains:
                continue
            contested = np.flatnonzero(shared)
            # Stronger independent core coverage gets to remain the reference.
            owners = sorted((a, b), key=lambda i: int((valid[i] & ~shared)[CORE].sum()))
            for owner in owners:
                peer = b if owner == a else a
                own = CORE[valid[owner, CORE] & ~shared[CORE] &
                           (np.linalg.norm(xy[owner, CORE]-xy[peer, CORE], axis=1) > 2*tolerance)]
                yield owner, peer, contested, own, tolerance, valid, bool(chains)


def conditioned(image, box, retention, *, prediction=None, peer=None, own=(), owner=None, valid=None):
    """Return a fresh full-size RGB image; never crop/stretch the source."""
    with image.convert('RGB') as rgb:
        pixels = np.asarray(rgb).copy()
    yy, xx = np.ogrid[:image.height, :image.width]
    outside = (xx < box[0]) | (xx >= box[2]) | (yy < box[1]) | (yy >= box[3])
    if retention < 1:
        pixels[outside] = np.rint(NEUTRAL + retention*(pixels[outside].astype(float)-NEUTRAL)).astype(np.uint8)
    if peer is not None:
        # No instance mask is assumed here: suppress only measured rival bones,
        # protecting the target's independent core. Never blank a peer box.
        with Image.new('L', image.size) as stencil:
            draw = ImageDraw.Draw(stencil)
            xy = prediction[0]
            width = max(3, round(.025*np.linalg.norm(np.asarray(box[2:])-box[:2])))
            for chain in (*CHAINS, (5, 6, 12, 11, 5)):
                for root, tip in zip(chain, chain[1:]):
                    if valid[peer, root] and valid[peer, tip]:
                        draw.line([tuple(xy[peer, root]), tuple(xy[peer, tip])], fill=255, width=width)
            for k in own:
                x, y = xy[owner, k]
                draw.ellipse((x-width, y-width, x+width, y+width), fill=0)
            rival = np.asarray(stencil) > 0
        pixels[rival] = np.rint(NEUTRAL + .25*(pixels[rival].astype(float)-NEUTRAL)).astype(np.uint8)
    return Image.fromarray(pixels)


def acceptable(candidate, prediction, bounds, owner, peer, contested, own, tolerance, settings):
    """A higher confidence alone cannot establish a different person's pose."""
    xy, score, prob = candidate
    if (xy.shape != (1, 17, 2) or score.shape != (1, 17) or
            set(prob) != set(prediction[2]) or any(v.shape != (1, 17) for v in prob.values()) or
            not np.isfinite(xy).all() or any(not np.isfinite(v).all() or
            np.any((v < 0) | (v > 1)) for v in (score, *prob.values()))):
        raise ValueError('Pose collision recovery returned malformed COCO17 evidence.')
    valid = reliable(candidate, [bounds[owner]], settings)[0]
    old_valid = reliable(prediction, bounds, settings)
    # Independent landmarks must survive a partial-pose repair.
    if len(own) >= 2:
        if not np.all(valid[own] & (np.linalg.norm(xy[0, own]-prediction[0][owner, own], axis=1) <= tolerance)):
            return 'own_core_drift'
    else:
        # A fully copied pose needs new head/torso support in the target's
        # exclusive box region; shifted confidence peaks alone are insufficient.
        exclusive = valid[CORE] & ~inside(xy[0, CORE], np.asarray(bounds[peer]))
        if exclusive.sum() < 2:
            return 'no_independent_core'
    distinct = valid.copy()
    for rival in range(len(bounds)):
        if rival != owner:
            distance = np.linalg.norm(xy[0]-prediction[0][rival], axis=1)
            copied = valid & old_valid[rival] & (distance <= tolerance)
            if any(copied[list(chain)].sum() >= settings['partial_min_shared_points']
                   for chain in (*CHAINS, TORSO)):
                return 'copied_fragment_remains'
            distinct &= ~old_valid[rival] | (distance > 2*tolerance)
    if distinct[contested].sum() < settings['partial_min_shared_points']:
        return 'copied_fragment_remains'
    if valid.sum() < old_valid[owner].sum():
        return 'lost_reliable_points'
    return None


def recover(pose, image, bounds, prediction, settings):
    """Return updated predictions, audit attempts and charged model jobs."""
    if len(bounds) < 2 or not (settings.get('box_context_retry') or settings.get('partial_pose_recovery')):
        return prediction, [], 0
    from .refinement_budget import budget
    attempts, used, changed = [], 0, set()
    output = (prediction[0].copy(), prediction[1].copy(), {k: v.copy() for k, v in prediction[2].items()})
    for owner, peer, contested, own, tolerance, valid, partial in collisions(bounds, prediction, settings):
        if owner in changed or peer in changed:
            continue
        modes = (['box_context'] if settings['box_context_retry'] else [])
        if settings['partial_pose_recovery'] and partial and len(own) >= 2 and valid[peer, CORE].sum() >= 2:
            modes.append('partial_pose')
        for mode in modes:
            if used >= settings['max_contact_attempts']:
                return output, attempts, used
            record = dict(owner=int(owner), peer=int(peer), mode=mode, shared_indices=contested.tolist())
            attempts.append(record)
            if not budget(image, settings).admit(mode, record=record):
                record['reason'] = 'shared_retry_budget'
                return output, attempts, used
            used += 1
            extra = (dict(prediction=prediction, peer=peer, own=own, owner=owner, valid=valid)
                     if mode == 'partial_pose' else {})
            retention = settings['box_context_retention'] if settings['box_context_retry'] else 1.
            with conditioned(image, bounds[owner], retention, **extra) as retry:
                values = pose.predict_many([(retry, [bounds[owner]])])
            if len(values) != 1:
                raise ValueError('Pose collision recovery returned an invalid frame count.')
            candidate = values[0]
            reason = acceptable(candidate, output, bounds, owner, peer, contested, own, tolerance, settings)
            record.update(accepted=reason is None, reason=reason)
            if reason is None:
                output[0][owner], output[1][owner] = candidate[0][0], candidate[1][0]
                for key in output[2]:
                    output[2][key][owner] = candidate[2][key][0]
                changed.add(owner)
                break
    return output, attempts, used
