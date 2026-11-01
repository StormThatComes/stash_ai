"""Geometry-only face/pose association, shared by inference and audit reasons."""
import math


def policy():
    """Version the gates in retained evidence as well as model provenance."""
    return dict(version='landmark-supported-partial-face-v1', containment_tolerance=.05,
                min_partial_face_coverage=.5, min_head_landmarks=2, max_head_distance=.15)


def _inside(point, bounds):
    x, y = point
    left, top, right, bottom = bounds
    return math.isfinite(x) and math.isfinite(y) and x >= 0 and y >= 0 and left <= x <= right and top <= y <= bottom


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
    head = [(px, py) for px, py in person['keypoints'][:3] if px >= 0 and py >= 0]
    if not head:
        return 'no_valid_head_keypoints'
    if not any(x <= px <= x+w and y <= py <= y+h for px, py in head):
        return 'head_keypoints_outside_face'
    if contained:
        return None
    landmarks = getattr(face, 'landmarks', ())
    pairs = []
    for pose_index, face_index in enumerate((2, 1, 0)):
        if pose_index >= len(person['keypoints']) or face_index >= len(landmarks):
            continue
        pose_point, face_point = person['keypoints'][pose_index], landmarks[face_index]
        if (_inside(pose_point, (left, top, right, bottom)) and
                _inside(face_point, (x, y, x+w, y+h))):
            pairs.append((pose_point, face_point))
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
