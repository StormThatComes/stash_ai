"""Bounded SCRFD retries before any face/body evidence is encoded."""
import math
from types import SimpleNamespace

import numpy as np

from .models import checked_detection


BOX_TOLERANCE = .05


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


def repair(detector, images, found):
    from .quality import alignment_measurements
    from .rotated_faces import enabled
    threshold = detector.config.data['detection'].get('alignment_retry_error', 0.)
    rotation = enabled(detector.config)
    if not threshold and not rotation:
        return found
    requests = []
    for frame, (pixels, faces) in enumerate(zip(images, found, strict=True)):
        for index, face in enumerate(faces):
            turn = quarter_turn(face) if rotation else 0
            spatial = not box_consistent(face)
            rotate = rotation and (turn != 0 or spatial)
            if not rotate and not threshold:
                continue
            quality = alignment_measurements(face)
            error = quality.get('alignment_error', 1.)
            if not rotate and not (threshold and (error > threshold or spatial)):
                continue
            x, y, w, h = face.box
            box = (max(0, math.floor(x-.25*w)), max(0, math.floor(y-.25*h)),
                   min(pixels.shape[1], math.ceil(x+1.25*w)), min(pixels.shape[0], math.ceil(y+1.25*h)))
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            turns = (turn,) if turn is not None else (0, 1, 2, 3)
            for degrees in turns:
                requests.append((frame, index, box, degrees, error, spatial, rotate))
    result = [list(faces) for faces in found]
    candidates = {}
    maximum = max(1, min(8, detector.config.data['execution']['detector_batch_size']))
    for start in range(0, len(requests), maximum):
        batch = requests[start:start+maximum]
        crops = [np.rot90(images[f][box[1]:box[3], box[0]:box[2]], turn)
                 for f, _, box, turn, *_ in batch]
        repairs = detector._raw_many(crops)
        for request, alternatives in zip(batch, repairs, strict=True):
            frame, index, box, turn, error, spatial, rotate = request
            original = found[frame][index]
            image = SimpleNamespace(width=images[frame].shape[1], height=images[frame].shape[0])
            eligible = []
            for other in alternatives:
                other = checked_detection(getattr(other, 'row', other))
                # A rotated retry must actually find an upright head.
                if turn and quarter_turn(other) != 0:
                    continue
                candidate = restore(other, box, turn)
                if not same_box(original, candidate):
                    continue
                # Landmark displacement is the defect being repaired: use boxes
                # to reserve ownership, not proximity to the defective points.
                if any(j != index and overlap(candidate, peer) >= .25
                       for j, peer in enumerate(found[frame])):
                    continue
                if candidate.score < original.score-.05:
                    continue
                quality = alignment_measurements(candidate, image)
                if not quality['alignment_valid']:
                    continue
                new_error = quality.get('alignment_error', 1.)
                if spatial or rotate:
                    if new_error > max(.20, error):
                        continue
                elif new_error >= error:
                    continue
                eligible.append((new_error, -candidate.score, candidate.row, candidate))
            # Multiple heads in the same crop are not resolved by confidence.
            if len(eligible) == 1:
                candidates.setdefault((frame, index), []).extend(eligible)
    for (frame, index), options in candidates.items():
        # Different orientations must agree on both the head and its landmarks.
        from .rotated_faces import duplicate_face
        if any(not duplicate_face(a[-1], b[-1]) for i, a in enumerate(options) for b in options[i+1:]):
            continue
        result[frame][index] = min(options, key=lambda row: row[:3])[-1]
    return result
