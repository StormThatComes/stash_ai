"""Small, retained same-cue ownership proofs, measured before face fallback.

Masks corroborate measured anatomy; disjoint silhouettes alone never establish
ownership. No face landmark is copied into the pose by this module.
"""
import math
import numpy as np

VERSION = 'same-cue-head-torso-v3-single-shoulder'
LIMITS = dict(head_error=.20, rival_margin=.12, min_torso_pixels=6.,
              head_projection_min=-.25, head_projection_max=1.25, head_lateral_ratio=.75,
              core_gap_pixels=4., mask_support=.8, foreign_mask_support=.1, mask_overlap=.1)


def _points(person):
    details = person.get('pose_refinement', {})
    borrowed = {i for i, _ in details.get('temporal_prompts', ())}
    if details.get('face_prompt'):
        borrowed.add(details['face_prompt'][0])
    visibility = person.get('keypoint_visibility')
    left, top, right, bottom = person['bounds']
    return {i: np.asarray(p, dtype=float) for i, p in enumerate(person.get('keypoints', ()))
            if i not in borrowed and len(p) == 2 and all(math.isfinite(v) for v in p)
            and left <= p[0] < right and top <= p[1] < bottom and min(p) >= 0
            and (visibility is None or visibility[i] >= .5)}


def _head(face, points):
    x, y, w, h = face.box
    aligned = []
    errors = []
    for index, landmark in enumerate((2, 1, 0)):
        if index not in points or landmark >= len(face.landmarks):
            continue
        target = face.landmarks[landmark]
        if not (all(math.isfinite(v) for v in target) and x <= target[0] <= x+w and y <= target[1] <= y+h):
            continue
        error = math.dist(points[index], target) / min(w, h)
        errors.append(error)
        if error <= LIMITS['head_error']:
            aligned.append((index, points[index], target))
    independent = any(math.dist(a[1], b[1]) >= 1 and math.dist(a[2], b[2]) >= 1
                      for i, a in enumerate(aligned) for b in aligned[i+1:])
    return aligned, max(errors, default=None), independent


def _torso(points, head):
    indices = [i for i in (5, 6, 11, 12) if i in points]
    if 5 not in points or 6 not in points or len(indices) < 3:
        return None
    shoulder = (points[5]+points[6])/2
    hip = np.mean([points[i] for i in (11, 12) if i in points], axis=0)
    axis = shoulder-hip
    length = float(np.linalg.norm(axis))
    if length < LIMITS['min_torso_pixels']:
        return None
    delta = head-shoulder
    projection = float(delta @ axis / length**2)
    lateral = float(np.linalg.norm(delta-projection*axis))
    width = float(np.linalg.norm(points[5]-points[6]))
    lateral_limit = LIMITS['head_lateral_ratio']*max(width, length*.5)
    connected = LIMITS['head_projection_min'] <= projection <= LIMITS['head_projection_max'] and lateral <= lateral_limit
    return dict(points=[points[i] for i in indices], shoulder=shoulder, hip=hip,
                connected=connected, head_projection=projection, head_lateral=lateral/length,
                head_lateral_limit=lateral_limit/length)


def _separated(first, second):
    a, b = np.asarray(first), np.asarray(second)
    # A small gap between core rectangles is independent of oversized leg boxes.
    gap = LIMITS['core_gap_pixels']
    return bool(np.any(a.max(axis=0)+gap < b.min(axis=0)) or np.any(b.max(axis=0)+gap < a.min(axis=0)))


def _mask(person, cache):
    key = id(person)
    if key not in cache:
        from .mask_usage import uses
        from .body_overlap import foreground_mask
        value = None
        if uses(person, 'overlap') and uses(person, 'body_admission'):
            try:
                value = foreground_mask(person)
            except (ValueError, TypeError):
                pass
        cache[key] = value
    return cache[key]


def _coverage(person, mask, points):
    left, top, right, bottom = person['bounds']
    return sum(bool(mask[int(y-top), int(x-left)]) if left <= x < right and top <= y < bottom else False
               for x, y in points)/len(points)


def _full_body(face, person, people, *, face_id, faces=(), mask_cache=None):
    """Use all detected peers, including rejected/maskless poses, before storage."""
    result = dict(version=VERSION, face=face_id, confirmed=False, reason='insufficient_head_points',
                  head_points=[], peers=[], thresholds=dict(LIMITS))
    points = _points(person)
    aligned, error, independent = _head(face, points)
    result.update(head_points=[i for i, _, _ in aligned], head_error=error)
    if not independent:
        return result
    if error > LIMITS['head_error']:
        result['reason'] = 'head_landmark_mismatch'
        return result
    head = np.mean([p for _, p, _ in aligned], axis=0)
    torso = _torso(points, head)
    result['torso_points'] = [i for i in (5,6,11,12) if i in points]
    result['reason'] = 'insufficient_torso_points'
    if torso is None:
        return result
    result.update(head_projection=torso['head_projection'], head_lateral=torso['head_lateral'],
                  head_lateral_limit=torso['head_lateral_limit'])
    result['reason'] = 'head_torso_mismatch'
    if not torso['connected']:
        return result
    for other in faces:
        if other is face:
            continue
        _, rival_error, strong = _head(other, points)
        if strong and rival_error-error < LIMITS['rival_margin']:
            result['reason'] = 'competing_face_claim'
            return result
    if person.get('pose_refinement', {}).get('mask_conflict'):
        result['reason'] = 'mask_conflict'
        return result
    core = [p for _, p, _ in aligned]+torso['points']
    probes = core+[a+(b-a)*t for a, b in ((head, torso['shoulder']), (torso['shoulder'], torso['hip']))
                   for t in (.25, .5, .75)]
    cache = {} if mask_cache is None else mask_cache
    for index, peer in enumerate(people):
        if peer is person:
            continue
        measured = dict(person_index=index, reason='insufficient_peer_torso')
        result['peers'].append(measured)
        left, top, right, bottom = peer['bounds']
        if _separated(core, [(left,top),(right,bottom)]):
            measured['reason'] = 'separated_peer_box'
            continue
        other = _points(peer)
        peer_torso = _torso(other, head)
        if peer_torso is None:
            continue
        _, rival_error, strong = _head(face, other)
        measured['head_error'] = rival_error
        if strong and peer_torso['connected'] and rival_error-error < LIMITS['rival_margin']:
            measured['reason'] = 'competing_head_claim'
            continue
        peer_core = peer_torso['points']+[other[i] for i in (0, 1, 2) if i in other]
        if _separated(core, peer_core):
            measured['reason'] = 'separated_head_torso'
            continue
        own_mask, peer_mask = _mask(person, cache), _mask(peer, cache)
        from .mask_usage import uses
        measured.update(own_mask_available=own_mask is not None, peer_mask_available=peer_mask is not None)
        measured['reason'] = ('masks_disabled' if any(not uses(p,k) for p in (person,peer)
                              for k in ('overlap','body_admission')) else 'masks_unavailable_or_invalid')
        if own_mask is None or peer_mask is None:
            continue
        # Both masks must support their own torso, and exclude the linked head/core.
        own_support = _coverage(person, own_mask[0], probes)
        peer_support = _coverage(peer, peer_mask[0], peer_torso['points'])
        foreign = _coverage(peer, peer_mask[0], probes)
        reverse = _coverage(person, own_mask[0], peer_torso['points'])
        a, b = person['bounds'], peer['bounds']
        x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
        overlap = (int(np.count_nonzero(own_mask[0][y0-a[1]:y1-a[1], x0-a[0]:x1-a[0]] &
                                      peer_mask[0][y0-b[1]:y1-b[1], x0-b[0]:x1-b[0]]))/min(own_mask[1], peer_mask[1])
                   if x0 < x1 and y0 < y1 else 0.)
        measured.update(own_mask_support=own_support, peer_mask_support=peer_support,
                        foreign_mask_support=max(foreign, reverse), mask_overlap=overlap)
        measured['reason'] = ('separated_masks_and_anatomy' if min(own_support, peer_support) >= LIMITS['mask_support']
                              and max(foreign, reverse) <= LIMITS['foreign_mask_support']
                              and overlap <= LIMITS['mask_overlap'] else 'mask_core_conflict')
    blockers = [p['reason'] for p in result['peers'] if not p['reason'].startswith('separated_')]
    result.update(confirmed=not blockers, reason=blockers[0] if blockers else 'same_frame_head_torso')
    return result


def measure(face, person, people, *, face_id, faces=(), mask_cache=None):
    """Keep the full-body proof and independently evaluate a masked upper body."""
    cache = {} if mask_cache is None else mask_cache
    result = _full_body(face, person, people, face_id=face_id, faces=faces, mask_cache=cache)
    result['full_body_reason'] = result['reason']
    if result['confirmed']:
        return result
    from .contact_upper_body import measure as upper_body
    upper = upper_body(face, person, people, faces=faces, mask_cache=cache)
    result['upper_body'] = upper
    # Real contradictions in the original anatomy remain vetoes. Whole-torso
    # overlap alone is not a contradiction in the head/shoulder ownership proof.
    vetoes = {'head_landmark_mismatch', 'head_torso_mismatch', 'competing_face_claim',
              'mask_conflict', 'competing_head_claim'}
    veto = result['reason'] in vetoes or any(p['reason'] in vetoes for p in result['peers'])
    if upper['confirmed'] and not veto:
        result.update(confirmed=True, reason=upper['reason'])
    return result


def retained(row, quality):
    proof = quality.get('contact_face_evidence')
    if not isinstance(proof, dict) or proof.get('version') != VERSION:
        return dict(confirmed=False, reason='same_frame_evidence_unavailable')
    if proof.get('face') != row.get('face') or row.get('_contact_correction'):
        return dict(confirmed=False, reason='linked_face_changed')
    if quality.get('query_only') or row.get('face_rejected'):
        return dict(confirmed=False, reason='query_only_or_rejected_face')
    return proof
