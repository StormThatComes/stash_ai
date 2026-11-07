"""Geometry-only face/pose association, shared by inference and audit reasons."""
import math


def policy():
    """Version the gates in retained evidence as well as model provenance."""
    return dict(version='visible-landmark-track-retry-v4', containment_tolerance=.05,
                min_partial_face_coverage=.5, min_head_landmarks=2, max_head_distance=.15)


def _inside(point, bounds):
    x, y = point
    left, top, right, bottom = bounds
    return math.isfinite(x) and math.isfinite(y) and x >= 0 and y >= 0 and left <= x <= right and top <= y <= bottom


def _visibility(person, index):
    values = person.get('keypoint_visibility')
    return 1. if values is None else float(values[index])


def association_rejection(face, person):
    """Keep the contained-face path; rescue partial boxes with aligned landmarks.

    Distances use original cue coordinates and the shorter face-box side.
    Pose points have already passed confidence/presence/visibility filtering.
    Never swap eyes or pick the nearest point: COCO nose/left-eye/right-eye
    map to YuNet/SCRFD landmark indices 2/1/0, respectively.
    """
    settings = policy()
    x, y, w, h = face.box
    left, top, right, bottom = person['bounds']
    tolerance = settings['containment_tolerance'] * min(w, h)
    contained = (left-tolerance <= x and top-tolerance <= y and
                 right+tolerance >= x+w and bottom+tolerance >= y+h)
    if not contained:
        overlap = max(0, min(x+w, right)-max(x, left)) * max(0, min(y+h, bottom)-max(y, top))
        if overlap < settings['min_partial_face_coverage'] * w * h:
            return 'face_outside_person_box'
    head = [(px,py) for i,(px,py) in enumerate(person['keypoints'][:3])
            if px>=0 and py>=0 and _visibility(person,i)>=.5]
    if not head:
        return 'no_valid_head_keypoints'
    if not any(x <= px <= x+w and y <= py <= y+h for px, py in head):
        return 'head_keypoints_outside_face'
    landmarks = getattr(face, 'landmarks', ())
    pairs = []
    for pose_index, face_index in enumerate((2, 1, 0)):
        if pose_index >= len(person['keypoints']) or face_index >= len(landmarks):
            continue
        pose_point, face_point = person['keypoints'][pose_index], landmarks[face_index]
        if (_visibility(person,pose_index)>=.5 and _inside(pose_point, (left, top, right, bottom)) and
                _inside(face_point, (x, y, x+w, y+h))):
            pairs.append((pose_point, face_point))
    if contained:
        # A missing far eye is common in profile. Measured contradictions still veto.
        if any(math.dist(pose, reference) > .20 * min(w, h) for pose, reference in pairs):
            return 'head_landmark_mismatch'
        return None
    if len(pairs) < settings['min_head_landmarks']:
        return 'insufficient_head_landmarks'
    distance = settings['max_head_distance'] * min(w, h)
    if any(math.dist(pose, measured) > distance for pose, measured in pairs):
        return 'head_landmark_mismatch'
    # Coincident predictions do not provide two independent anatomical points.
    if any(math.dist(a[0], b[0]) < 1 or math.dist(a[1], b[1]) < 1
           for index, a in enumerate(pairs) for b in pairs[index+1:]):
        return 'insufficient_head_landmarks'
    return None


def resolve(faces, people, choices):
    """Resolve competing head claims using a global assignment with abstention."""
    from collections import Counter
    usage = Counter(p for candidates in choices for p in candidates)
    matches = [c[0] if len(c) == 1 and usage[c[0]] == 1 else None for c in choices]
    contested = [i for i, c in enumerate(choices) if c and matches[i] is None]
    if not contested:
        return matches
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    owners = sorted({p for i in contested for p in choices[i]})
    columns = {p: j for j, p in enumerate(owners)}
    weights = np.zeros((len(contested), len(owners) + len(contested)))
    for row, i in enumerate(contested):
        face = faces[i]
        for p in choices[i]:
            distances = []
            landmarks = getattr(face, 'landmarks', ())
            weights_visible = []
            for k,(point,j) in enumerate(zip(people[p]['keypoints'], (2, 1, 0))):
                if min(point) >= 0 and j < len(landmarks) and _visibility(people[p],k)>=.5:
                    weights_visible.append(_visibility(people[p],k))
                    distances.append(math.dist(point, landmarks[j]) / min(face.box[2:]))
            if len(distances) >= 2 and max(distances) <= .20:
                weights[row, columns[p]] = 1 - np.average(distances,weights=weights_visible)
    rows, cols = linear_sum_assignment(weights, maximize=True)
    total = weights[rows, cols].sum()
    for row, col in zip(rows, cols):
        score = weights[row, col]
        if score <= 0:
            continue
        weights[row, col] = 0
        other_rows, other_cols = linear_sum_assignment(weights, maximize=True)
        if total - weights[other_rows, other_cols].sum() >= .12:
            matches[contested[row]] = owners[col]
        weights[row, col] = score
    return matches
