"""Readable audit reasons derived from the production face/person pair gates."""
from .sapiensid_backend import association_choices, association_rejection


REASON_LABELS = {
    'no_people_detected': 'no detected pose',
    'face_outside_person_box': 'face outside pose box',
    'no_valid_head_keypoints': 'no valid nose/eye points',
    'head_keypoints_outside_face': 'nose/eyes outside face',
    'head_keypoints_missing_or_outside': 'nose/eyes missing or outside face',
    'multiple_person_candidates': 'multiple poses',
    'competing_faces': 'competing faces',
    'unmatched': 'association unavailable',
}


def explain_associations(faces, people, matches):
    """Respect returned matches; diagnose geometry on original, filtered points.

    Missing keypoints may have been suppressed or outside the crop. Do not
    claim a particular threshold rejected them without that evidence.
    """
    choices = association_choices(faces, people)
    owners = {}
    for index, candidates in enumerate(choices):
        for person in candidates:
            owners.setdefault(person, []).append(index)
    results = []
    for index, (face, match, candidates) in enumerate(zip(faces, matches, choices)):
        competing = sorted({other for person in candidates for other in owners[person] if other != index})
        reason = None
        if match is None:
            if not people:
                reason = 'no_people_detected'
            elif len(candidates) > 1:
                reason = 'multiple_person_candidates'
            elif candidates:
                reason = 'competing_faces' if competing else 'unmatched'
            else:
                rejections = {association_rejection(face, person) for person in people}
                rejections.discard('face_outside_person_box')
                if not rejections:
                    reason = 'face_outside_person_box'
                elif len(rejections) == 1:
                    reason = next(iter(rejections))
                else:
                    reason = 'head_keypoints_missing_or_outside'
        results.append(dict(reason=reason, candidates=candidates, competing_faces=competing))
    return results


def _references(prefix, indices):
    text = ', '.join(f'{prefix}{index}' for index in indices[:3])
    return text + (f' (+{len(indices) - 3})' if len(indices) > 3 else '')


def face_label(index, face, match, explanation):
    title = f'F{index} {face.score:.3f} -> ' + (f'P{match}' if match is not None else 'unmatched')
    reason = explanation['reason']
    if reason is None:
        return title
    detail = REASON_LABELS[reason]
    if reason == 'multiple_person_candidates':
        detail += ': ' + _references('P', explanation['candidates'])
    elif reason == 'competing_faces':
        detail += ': ' + _references('F', explanation['competing_faces'])
    return title + '\n' + detail
