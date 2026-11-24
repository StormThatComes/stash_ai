"""Bounded five-landmark detector retries before face/body evidence is encoded."""
import math
from types import SimpleNamespace

import numpy as np

from .models import checked_detection


BOX_TOLERANCE = .05
FACEBOX_CONTRACT = 'facebox-landmarks-v2-crop-only'


def box_consistent(face):
    """Allow subpixel/edge noise, but not a translated landmark constellation.

    This checks location, not frontal pose or eye ordering. A profile and a
    rolled face have the same spatial contract as an upright face.
    """
    x, y, w, h = face.box
    margin = BOX_TOLERANCE * min(w, h)
    return all(x-margin <= px <= x+w+margin and y-margin <= py <= y+h+margin
               for px, py in face.landmarks)


def quarter_turn(face):
    """Rotation that makes the measured anatomical horizontal axis upright."""
    from .face_frame import axis
    from .visibility import VisibilityGeometryError
    try:
        horizontal, _ = axis(face.landmarks, *face.box[2:])
    except VisibilityGeometryError:
        return None
    angle = math.atan2(horizontal[1], horizontal[0])
    return int(math.floor(angle / (math.pi / 2) + .5)) % 4


def overlap(first, second):
    x, y, w, h = first.box
    a, b, c, d = second.box
    intersection = max(0, min(x+w, a+c)-max(x, a))*max(0, min(y+h, b+d)-max(y, b))
    return intersection / max(1., w*h+c*d-intersection)


def same_box(first, second):
    x, y, w, h = first.box
    a, b, c, d = second.box
    return (overlap(first, second) >= .5 and
            math.hypot(a+c/2-x-w/2, b+d/2-y-h/2) <= .2*min(w, h))


def restore(face, box, turns):
    from .rotated_faces import restore as restore_crop
    return restore_crop(face, box, (0, True, 180, False)[turns])


def candidate_rejection(other, candidate, original, peers, turn, error, spatial, rotate, image,
                        *, from_facebox=False):
    from .quality import alignment_measurements
    # A rotated retry must actually find an upright head.
    if turn and quarter_turn(other) != 0:
        return 'retry_not_upright', None
    if not same_box(original, candidate):
        return 'different_face_box', None
    # Reserve ownership using boxes, not the defective landmark positions.
    if any(overlap(candidate, peer) >= .25 for peer in peers):
        return 'competing_face', None
    if candidate.score < original.score-.05:
        return 'lower_detection_score', None
    quality = alignment_measurements(candidate, image)
    if not quality['alignment_valid']:
        return 'invalid_alignment', quality
    new_error = quality.get('alignment_error', 1.)
    if from_facebox:
        # Raw full-frame points must not set the crop's acceptance threshold.
        if new_error > .20:
            return 'alignment_error', quality
    elif spatial or rotate:
        if new_error > max(.20, error):
            return 'alignment_error', quality
    elif new_error >= error:
        return 'alignment_not_improved', quality
    return None, quality


def repair(detector, images, found):
    from .quality import alignment_measurements
    from .rotated_faces import enabled
    from . import landmark_repair_audit as audit
    threshold = detector.config.data['detection'].get('alignment_retry_error', 0.)
    rotation = enabled(detector.config)
    from_facebox = detector.config.data['detection'].get('landmarks_from_facebox', False)
    traces = audit.begin(images, found, rotation=rotation, threshold=threshold, from_facebox=from_facebox)
    if not threshold and not rotation and not from_facebox:
        audit.finish(traces, found)
        return found
    requests = []
    # In crop-only mode the full-frame pass supplies boxes and scores, never
    # fallback landmarks. Empty/failed/ambiguous retries retain this sentinel.
    result = [[face.without_landmarks() if from_facebox else face for face in faces]
              for faces in found]
    for frame, (pixels, faces) in enumerate(zip(images, found, strict=True)):
        for index, face in enumerate(faces):
            if from_facebox:
                # Do not let the discarded points choose or exclude rotations.
                turn, error, spatial, rotate = None, 0., False, rotation
                turns = (0, 1, 2, 3) if rotation else (0,)
            else:
                turn = quarter_turn(face) if rotation else 0
                spatial = not box_consistent(face)
                rotate = rotation and (turn != 0 or spatial)
                if not rotate and not threshold:
                    continue
                quality = alignment_measurements(face)
                error = quality.get('alignment_error', 1.)
                if not rotate and not (threshold and (error > threshold or spatial)):
                    continue
                # Out-of-box points cannot reliably choose their own repair
                # orientation. Keep the measured turn as a diagnostic hint.
                turns = (0, 1, 2, 3) if rotation and (spatial or turn is None) else (turn,)
            x, y, w, h = face.box
            box = (max(0, math.floor(x-.25*w)), max(0, math.floor(y-.25*h)),
                   min(pixels.shape[1], math.ceil(x+1.25*w)), min(pixels.shape[0], math.ceil(y+1.25*h)))
            if box[2] <= box[0] or box[3] <= box[1]:
                if traces is not None:
                    traces[frame][index]['status'] = 'empty_crop'
                continue
            if traces is not None:
                traces[frame][index].update(status='no_eligible_candidate',
                    estimated_turn_degrees=None if turn is None else turn*90,
                    crop_xyxy=list(box), tried_turn_degrees=[n*90 for n in turns])
            for degrees in turns:
                requests.append((frame, index, box, degrees, error, spatial, rotate))
    candidates = {}
    infer = (getattr(detector, '_raw_facebox_many', detector._raw_many)
             if from_facebox else detector._raw_many)
    maximum = max(1, min(8, detector.config.data['execution']['detector_batch_size']))
    for start in range(0, len(requests), maximum):
        batch = requests[start:start+maximum]
        crops = [np.rot90(images[f][box[1]:box[3], box[0]:box[2]], turn)
                 for f, _, box, turn, *_ in batch]
        repairs = infer(crops)
        for request, alternatives in zip(batch, repairs, strict=True):
            frame, index, box, turn, error, spatial, rotate = request
            original = found[frame][index]
            image = SimpleNamespace(width=images[frame].shape[1], height=images[frame].shape[0])
            eligible = []
            attempt = None
            if traces is not None:
                attempt = dict(turn_degrees=turn*90, candidates=[], status='no_eligible_candidate')
                traces[frame][index]['attempts'].append(attempt)
            peers = [peer for j, peer in enumerate(found[frame]) if j != index]
            for other in alternatives:
                other = checked_detection(getattr(other, 'row', other))
                candidate = restore(other, box, turn)
                reason, quality = candidate_rejection(other, candidate, original, peers,
                    turn, error, spatial, rotate, image, from_facebox=from_facebox)
                replacement = candidate
                if from_facebox:
                    # The full-frame detector owns the box/score. The crop only
                    # contributes points, which must fit that original geometry.
                    replacement = checked_detection([*original.row[:4], *candidate.row[4:14], original.score])
                    if reason is None:
                        quality = alignment_measurements(replacement, image)
                        if not quality['alignment_valid']:
                            reason = 'invalid_original_box_alignment'
                if attempt is not None:
                    attempt['candidates'].append(dict(crop_detection=audit.geometry(other),
                        source_detection=audit.geometry(candidate), rejection=reason, quality=quality))
                if reason is not None:
                    continue
                new_error = quality.get('alignment_error', 1.)
                eligible.append((new_error, -candidate.score, replacement.row, turn, replacement))
            # Multiple heads in the same crop are not resolved by confidence.
            if attempt is not None:
                attempt['status'] = ('eligible' if len(eligible) == 1 else
                                     'ambiguous_candidates' if eligible else 'no_eligible_candidate')
            if len(eligible) == 1:
                candidates.setdefault((frame, index), []).extend(eligible)
    for (frame, index), options in candidates.items():
        # Different orientations must agree on both the head and its landmarks.
        from .rotated_faces import duplicate_face
        if any(not duplicate_face(a[-1], b[-1]) for i, a in enumerate(options) for b in options[i+1:]):
            if traces is not None:
                traces[frame][index]['status'] = 'conflicting_orientations'
            continue
        selected = min(options, key=lambda row: row[:3])
        result[frame][index] = selected[-1]
        if traces is not None:
            traces[frame][index].update(status='refined_from_facebox' if from_facebox else 'repaired',
                                       selected_turn_degrees=selected[3]*90)
    audit.finish(traces, result)
    return result
