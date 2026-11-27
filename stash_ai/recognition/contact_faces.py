"""Disambiguate touching RetinaFace detections with independently measured pose heads."""
import math
from itertools import combinations

from .body_observations import enabled as body_enabled
from .person_association import association_rejection


CONTRACT = 'retina-contact-heads-v1'
MIXED = 'face_landmarks_mixed_people'
BETWEEN = 'face_between_confirmed_heads'
# Pose nose / anatomical left and right eyes -> RetinaFace landmark indices.
HEAD = (2, 1, 0)


def enabled(config):
    return config.data['models']['detector'] == 'retinaface_r50' and body_enabled(config)


def measured_people(people, size):
    from .rotated_faces import reliable_head_points
    result = []
    for person in people:
        points = reliable_head_points(person, size)
        head = [points.get(i, (-1, -1)) for i in range(5)]
        result.append(dict(person, keypoints=head + list(person.get('keypoints', ())[5:])))
    return result


def nearby_heads(face, people):
    """Two distinct measured features in or immediately beside the face box."""
    x, y, w, h = face.box
    pad = .15 * min(w, h)
    result = {}
    for owner, person in enumerate(people):
        points = {i: p for i, p in enumerate(person['keypoints'][:3]) if min(p) >= 0
                  and x-pad <= p[0] <= x+w+pad and y-pad <= p[1] <= y+h+pad}
        if any(math.dist(a, b) >= 2 for a, b in combinations(points.values(), 2)):
            result[owner] = points
    return result


def ownership(face, people, heads):
    """Return a unique coherent owner, or the two heads contributing distinct features."""
    if not face.landmarks_available:
        return None, ()
    side = min(face.box[2:])
    owners = []
    votes = {}
    for owner, points in heads.items():
        errors = {i: math.dist(p, face.landmarks[HEAD[i]]) / side for i, p in points.items()}
        if max(errors.values()) <= .20 and association_rejection(face, people[owner]) is None:
            owners.append(owner)
        for i, distance in errors.items():
            votes.setdefault(i, []).append((distance, owner))
    if owners:
        return (owners[0] if len(owners) == 1 else None), ()
    sources = set()
    for distances in votes.values():
        distances.sort()
        # Require the same anatomical point measured on both competing heads.
        if len(distances) >= 2 and distances[0][0] <= .20 and distances[1][0]-distances[0][0] >= .12:
            sources.add(distances[0][1])
    return None, tuple(sorted(sources)) if len(sources) >= 2 else ()


def conflicting_duplicate(face, previous, people, owner):
    from .face_landmark_repair import same_box
    from .rotated_faces import duplicate_face
    if not (same_box(face, previous) or duplicate_face(face, previous)):
        return False
    neighbour, _ = ownership(previous, people, nearby_heads(previous, people))
    return neighbour is None or neighbour == owner


def intermediate_pair(face, heads):
    """Require all three points partway along their respective head-to-head segments."""
    if len(heads) != 2 or not face.landmarks_available:
        return ()
    (first, a), (second, b) = heads.items()
    scale = min(face.box[2:])
    for i, landmark in enumerate(HEAD):
        if i not in a or i not in b:
            return ()
        dx, dy = b[i][0]-a[i][0], b[i][1]-a[i][1]
        length = math.hypot(dx, dy)
        if length < .4 * scale:
            return ()
        x, y = face.landmarks[landmark]
        t = ((x-a[i][0])*dx + (y-a[i][1])*dy) / length**2
        if not .2 <= t <= .8 or math.dist((x, y), (a[i][0]+t*dx, a[i][1]+t*dy)) > .12*scale:
            return ()
    return first, second


def guard_body_inputs(faces, people, matches, size):
    """Keep contradicted face points out of the SapiensID input before collection."""
    measured = measured_people(people, size)
    for i, (face, match) in enumerate(zip(faces, matches, strict=True)):
        if match is None:
            continue
        heads = nearby_heads(face, measured)
        if len(heads) >= 2:
            owner, sources = ownership(face, measured, heads)
            if sources or (owner is None and intermediate_pair(face, heads)):
                matches[i] = None


def cue_work(rows, faces, people, strict):
    """Geometry-only plan. Missing pose support alone never rejects a face."""
    from .face_landmark_repair import overlap
    mixed, retry_boxes, search, centers = {}, [], set(), {}
    for i, (row, face) in enumerate(zip(rows, faces, strict=True)):
        heads = nearby_heads(face, people)
        if not heads or row['rejected'] in (MIXED, BETWEEN):
            continue
        owner, sources = ownership(face, people, heads)
        if sources:
            mixed[i] = sources
        elif owner is None and not row['rejected'] and (pair := intermediate_pair(face, heads)):
            centers[i] = pair
        if (strict and row['rejected'] == 'face_landmarks_invalid' and not face.landmarks_available
                and any(overlap(face, peer) >= .25 for j, peer in enumerate(faces) if i != j)):
            retry_boxes.append(i)
        if owner is None and len(heads) >= 2:
            search.update(heads)
    return mixed, retry_boxes, search, centers
