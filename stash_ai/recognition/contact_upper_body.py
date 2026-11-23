"""Masked, rotation-independent ownership evidence for a partial/profile body.

The mask corroborates original anatomy; it never supplies a missing keypoint.
Only the head/shoulder region participates in competition, so contact at a knee
or hip does not make an otherwise exclusive head ambiguous.
"""
import math
import numpy as np

from .contact_face_evidence import _points, _head, _torso, _mask, _coverage, _separated, LIMITS as FULL_LIMITS

LIMITS = dict(ear_distance_min=.10, ear_distance_max=1.25, min_shoulder_pixels=6.,
              shoulder_face_ratio_min=.25, shoulder_face_ratio_max=4.,
              neck_distance_min=.10, neck_distance_max=1.5, head_lateral_max=.85,
              neck_normal_min=.10, mask_support=.90, foreign_mask_support=.10,
              corridor_radius=.06, corridor_step=.04, mask_gap=.08,
              single_shoulder_distance_min=.25, single_shoulder_distance_max=2.25,
              single_shoulder_projection_min=-.25, single_shoulder_projection_max=2.,
              single_shoulder_lateral_max=1.75, face_axis_min=.10)


def _excluded(person, accepted):
    details = person.get('pose_refinement', {})
    borrowed = {i:'temporal_prompt' for i, _ in details.get('temporal_prompts', ())}
    if details.get('face_prompt'):
        borrowed[details['face_prompt'][0]] = 'face_prompt'
    raw = person.get('keypoints', ())
    visibility = person.get('keypoint_visibility')
    conflicts = _conflicts(person)
    result = []
    for i in range(7):
        if i in accepted:
            continue
        reason = ('missing_or_invalid' if i >= len(raw) or len(raw[i]) != 2
                  or not all(math.isfinite(v) and v >= 0 for v in raw[i]) else
                  borrowed[i] if i in borrowed else
                  conflicts[i] if i in conflicts else
                  'occluded' if visibility is not None and (not math.isfinite(visibility[i]) or visibility[i] < .5)
                  else 'outside_box')
        result.append(dict(point=i, reason=reason))
    return result


def _conflicts(person):
    details = person.get('pose_refinement', {})
    result = {i:state for i,state in enumerate(person.get('keypoint_states', ()))
              if state in ('occluded','conflicted')}
    for key in ('joint_ownership_withheld','withheld_rival_joints','mask_ownership_withheld'):
        result.update((i,'point_ownership_conflict') for i in details.get(key, ()))
    contested = details.get('point_exclusions', {}).get('near_peer', 0)
    result.update((i,'point_ownership_conflict') for i in range(17) if contested & (1 << i))
    return result


def _original_points(person):
    conflicts = _conflicts(person)
    return {i:p for i,p in _points(person).items() if i not in conflicts}


def _face_landmarks(face):
    """All five landmarks corroborate a mask; they never add pose votes."""
    x, y, w, h = face.box
    if len(face.landmarks) != 5:
        return None
    if any(len(p) != 2 or not all(math.isfinite(v) for v in p)
           or not (x <= p[0] <= x+w and y <= p[1] <= y+h) for p in face.landmarks):
        return None
    return np.asarray(face.landmarks, dtype=float)


def _single_shoulder(points, head, scale, landmarks):
    if landmarks is None:
        return None, 'single_shoulder_landmarks_unavailable'
    index = next(i for i in (5, 6) if i in points)
    shoulder = points[index]
    # Use measured face orientation instead of screen-up or an invented shoulder
    # midpoint. This keeps the test invariant under camera/whole-body rotation.
    axis = np.mean(landmarks[3:5], axis=0)-np.mean(landmarks[:2], axis=0)
    length = float(np.linalg.norm(axis))
    if length < max(2., scale*LIMITS['face_axis_min']):
        return None, 'single_shoulder_face_axis_unavailable'
    delta = shoulder-head
    projection = float(delta @ axis/length/scale)
    lateral = float(np.linalg.norm(delta-projection*scale*axis/length))/scale
    distance = float(np.linalg.norm(delta))/scale
    face_base = np.mean(landmarks[3:5], axis=0)
    # Probe down the face axis, then across to the measured shoulder. A direct
    # diagonal can cut through background beside a narrow neck. This bend is
    # only a mask sampling location, never a synthesized pose keypoint.
    neck_probe = face_base+axis/length*max(0., float((shoulder-face_base) @ axis/length))
    result = dict(head=head, shoulder=shoulder, face_base=face_base, neck_probe=neck_probe,
                  scale=scale, shoulder_mode='single',
                  neck_distance=distance, head_lateral=lateral, head_projection=projection)
    if not (LIMITS['single_shoulder_distance_min'] <= distance <= LIMITS['single_shoulder_distance_max']
            and LIMITS['single_shoulder_projection_min'] <= projection <= LIMITS['single_shoulder_projection_max']
            and lateral <= LIMITS['single_shoulder_lateral_max']):
        return result, 'head_shoulder_mismatch'
    hips = [points[i] for i in (11, 12) if i in points]
    if hips:
        hip = points.get(index+6, np.mean(hips, axis=0))
        torso_axis = shoulder-hip
        torso_length = float(np.linalg.norm(torso_axis))
        if torso_length < FULL_LIMITS['min_torso_pixels']:
            return result, 'head_torso_mismatch'
        above = head-shoulder
        forward = float(above @ torso_axis/torso_length**2)
        offset = float(np.linalg.norm(above-forward*torso_axis))
        if not (FULL_LIMITS['head_projection_min'] <= forward <= FULL_LIMITS['head_projection_max']
                and offset <= FULL_LIMITS['head_lateral_ratio']*max(torso_length, scale)):
            return result, 'head_torso_mismatch'
    return result, 'confirmed'


def _anatomy(points, head, scale, *, single_landmarks=None, allow_single=False):
    if head is not None and allow_single and sum(i in points for i in (5,6)) == 1:
        return _single_shoulder(points, head, scale, single_landmarks)
    if head is None or not all(i in points for i in (5, 6)):
        return None, 'insufficient_shoulder_points'
    across = points[6]-points[5]
    width = float(np.linalg.norm(across))
    if width < LIMITS['min_shoulder_pixels'] or not (
            LIMITS['shoulder_face_ratio_min'] <= width/scale <= LIMITS['shoulder_face_ratio_max']):
        return None, 'invalid_shoulder_span'
    shoulder = (points[5]+points[6])/2
    delta = head-shoulder
    unit = max(width, scale)
    lateral = float(delta @ across/width)
    normal = float(np.linalg.norm(delta-lateral*across/width))/unit
    distance = float(np.linalg.norm(delta))/unit
    result = dict(head=head, shoulder=shoulder, width=width, scale=scale,
                  neck_distance=distance, head_lateral=abs(lateral)/unit, neck_normal=normal)
    connected = (LIMITS['neck_distance_min'] <= distance <= LIMITS['neck_distance_max']
                 and abs(lateral)/unit <= LIMITS['head_lateral_max']
                 and normal >= LIMITS['neck_normal_min'])
    if not connected:
        return result, 'head_shoulder_mismatch'
    # Visible hips may contradict the head/shoulder hypothesis; missing hips
    # are allowed, but a measured incompatible torso cannot be ignored.
    if any(i in points for i in (11, 12)):
        torso = _torso(points, head)
        if torso is None or not torso['connected']:
            return result, 'head_torso_mismatch'
    return result, 'confirmed'


def _corridor(a, b, scale):
    delta = b-a
    length = float(np.linalg.norm(delta))
    perpendicular = np.array([-delta[1], delta[0]])/max(length, 1.)
    steps = max(2, math.ceil(length/max(1., scale*LIMITS['corridor_step'])))
    center = np.linspace(a, b, steps+1)
    # Three thin lines test a neck/shoulder corridor, not isolated prompt dots.
    radius = max(1., scale*LIMITS['corridor_radius'])
    return [center+offset*perpendicular for offset in (-radius, 0., radius)]


def _samples(points, head_points, anatomy):
    shoulders = [points[i] for i in (5, 6) if i in points]
    core = [points[i] for i in head_points]+shoulders
    head, shoulder, scale = anatomy['head'], anatomy['shoulder'], anatomy['scale']
    links = ([(head, anatomy['face_base']), (anatomy['face_base'], anatomy['neck_probe']),
              (anatomy['neck_probe'], shoulder)]
             if anatomy.get('shoulder_mode') == 'single' else [(head, shoulder)])
    if anatomy.get('shoulder_mode') == 'single':
        core.extend((anatomy['face_base'], anatomy['neck_probe']))
    if len(shoulders) == 2:
        links.extend((point, shoulder) for point in shoulders)
    links.extend((points[i], head) for i in head_points)
    corridors = [line for a, b in links for line in _corridor(a, b, scale)]
    return core, corridors


def _membership(person, mask, points):
    left, top, right, bottom = person['bounds']
    return [bool(mask[int(y-top), int(x-left)]) if left <= x < right and top <= y < bottom else False
            for x, y in points]


def _support(person, mask, core, corridors, scale):
    support = _coverage(person, mask, core)
    coverage = 1.
    gap = 0.
    for line in corridors:
        membership = _membership(person, mask, line)
        coverage = min(coverage, sum(membership)/len(membership))
        step = float(np.linalg.norm(line[1]-line[0]))
        run = 0
        for inside in membership:
            run = 0 if inside else run+1
            gap = max(gap, run*step/scale)
    return dict(anatomy_support=support, corridor_support=coverage, max_gap=gap,
                confirmed=support >= LIMITS['mask_support'] and coverage >= LIMITS['mask_support']
                and gap <= LIMITS['mask_gap'])


def measure(face, person, people, *, faces=(), mask_cache):
    points = _original_points(person)
    aligned, error, independent = _head(face, points)
    indices = [i for i, _, _ in aligned]
    result = dict(confirmed=False, reason='insufficient_head_points', head_points=indices,
                  ear_points=[], shoulder_points=[i for i in (5, 6) if i in points],
                  excluded_points=_excluded(person, points), thresholds=dict(LIMITS), peers=[], checks={})
    checks = result['checks']
    scale = float(min(face.box[2:]))
    # A corresponding original ear supplements one original eye. It has no
    # face-detector landmark and is checked anatomically, never copied.
    ears = []
    for eye, ear in ((1, 3), (2, 4)):
        if eye not in indices or ear not in points:
            continue
        distance = float(np.linalg.norm(points[ear]-points[eye]))/scale
        x, y, w, h = face.box
        center = np.array([x+w/2, y+h/2])
        valid = (LIMITS['ear_distance_min'] <= distance <= LIMITS['ear_distance_max']
                 and float(np.linalg.norm(points[ear]-center))/scale <= LIMITS['ear_distance_max'])
        result.setdefault('ears', []).append(dict(point=ear, distance=distance, eligible=valid))
        if valid:
            ears.append(ear)
    head_ok = independent or len(indices) == 1 and bool(ears)
    # Contradictory original nose/eye measurements veto either head route.
    checks['head'] = ('head_landmark_mismatch' if error is not None and error > FULL_LIMITS['head_error']
                      else 'confirmed' if head_ok else 'insufficient_head_points')
    result.update(ear_points=ears, head_mode='nose_eyes' if independent else 'eye_ear', head_error=error)
    head_indices = indices + ([] if independent else ears)
    available = head_indices or [i for i in range(5) if i in points]
    head = np.mean([points[i] for i in available], axis=0) if available else None
    single = len(result['shoulder_points']) == 1
    landmarks = _face_landmarks(face) if single else None
    result['shoulder_mode'] = 'single' if single else 'paired'
    anatomy, checks['shoulders'] = _anatomy(points, head, scale, allow_single=single, single_landmarks=landmarks)
    if anatomy is not None:
        result.update({k:anatomy[k] for k in ('neck_distance', 'head_lateral', 'neck_normal', 'head_projection')
                       if k in anatomy})
    own_mask = _mask(person, mask_cache)
    result['mask_available'] = own_mask is not None
    from .mask_usage import uses
    checks['mask'] = ('mask_conflict' if person.get('pose_refinement', {}).get('mask_conflict') else
                      'masks_disabled' if not all(uses(person,k) for k in ('overlap','body_admission')) else
                      'masks_unavailable_or_invalid' if own_mask is None else 'geometry_unavailable')
    core = corridors = None
    if anatomy is not None and available:
        core, corridors = _samples(points, available, anatomy)
        # Landmark coordinates test the mask but add no independent pose vote.
        core.extend(target for _, _, target in aligned)
        if own_mask is not None:
            result['mask'] = _support(person, own_mask[0], core, corridors, scale)
            checks['mask'] = 'confirmed' if result['mask']['confirmed'] else 'mask_upper_body_gap'
            if single:
                support = _coverage(person, own_mask[0], landmarks)
                result['face_landmark_mask_support'] = support
                checks['face_landmark_mask'] = 'confirmed' if support == 1. else 'face_landmarks_outside_mask'
        if single:
            # Include every face landmark in local competition as well as in
            # own-mask coverage. They remain corroboration, not original points.
            core.extend(landmarks)
    checks['faces'] = 'confirmed'
    for other_face in faces:
        if other_face is face:
            continue
        rival, rival_error, _ = _head(other_face, points)
        if rival and error is not None and rival_error-error < FULL_LIMITS['rival_margin']:
            checks['faces'] = 'competing_face_claim'
            break
    # Continue measuring peers even if a head route failed: the inspector must
    # distinguish missing anatomy from mask ownership and actual competition.
    for index, peer in enumerate(people):
        if peer is person:
            continue
        measured = dict(person_index=index, reason='geometry_unavailable')
        result['peers'].append(measured)
        if core is None:
            continue
        other = _original_points(peer)
        rival, rival_error, _ = _head(face, other)
        measured['head_error'] = rival_error
        _, rival_connection = _anatomy(other, head, scale)
        if rival and error is not None and rival_error-error < FULL_LIMITS['rival_margin'] and rival_connection == 'confirmed':
            measured['reason'] = 'competing_head_claim'
            continue
        x, y, w, h = face.box
        # Include the full face extent in the coarse exclusion. A remote lower
        # body intersection is harmless only after our own mask/anatomy passes.
        local = core + [np.array([x,y]), np.array([x+w,y+h])]
        left, top, right, bottom = peer['bounds']
        measured['local_box_separated'] = _separated(local, [(left,top),(right,bottom)])
        if measured['local_box_separated']:
            measured['reason'] = 'separated_upper_body_box'
            continue
        other_head_indices = [i for i in range(5) if i in other]
        other_head = np.mean([other[i] for i in other_head_indices], axis=0) if other_head_indices else None
        other_scale = float(np.linalg.norm(other[6]-other[5])) if all(i in other for i in (5,6)) else scale
        other_anatomy, other_reason = _anatomy(other, other_head, max(other_scale, 1.))
        measured['anatomy'] = other_reason
        if other_anatomy is None or other_reason != 'confirmed':
            measured['reason'] = 'insufficient_peer_upper_body'
            continue
        other_core, other_corridors = _samples(other, other_head_indices, other_anatomy)
        peer_mask = _mask(peer, mask_cache)
        measured['mask_available'] = peer_mask is not None
        if own_mask is None or peer_mask is None:
            measured['reason'] = 'masks_disabled' if any(not uses(p,k) for p in (person,peer)
                for k in ('overlap','body_admission')) else 'masks_unavailable_or_invalid'
            continue
        peer_support = _support(peer, peer_mask[0], other_core, other_corridors, other_anatomy['scale'])
        probes = np.concatenate([np.asarray(core), *corridors])
        reverse_probes = np.concatenate([np.asarray(other_core), *other_corridors])
        foreign = max(_coverage(peer, peer_mask[0], probes), _coverage(person, own_mask[0], reverse_probes))
        head_probes = [points[i] for i in available]+(list(landmarks) if single else [])
        foreign_head = _coverage(peer, peer_mask[0], head_probes)
        measured.update(mask=peer_support, foreign_mask_support=foreign, foreign_head_support=foreign_head)
        measured['reason'] = ('separated_upper_body_masks' if peer_support['confirmed']
                              and foreign <= LIMITS['foreign_mask_support'] and foreign_head == 0.
                              else 'mask_upper_body_conflict')
    failures = [value for value in checks.values() if value != 'confirmed']
    failures.extend(p['reason'] for p in result['peers'] if not p['reason'].startswith('separated_'))
    result.update(confirmed=not failures, reason=failures[0] if failures else
                  'same_frame_single_shoulder' if single else 'same_frame_upper_body')
    return result
