"""Hard contradictions never consume an ambiguity margin; unknowns still do."""
HARD_BODY_BLOCKERS = frozenset({'person_copresence', 'face_contradiction'})


def body(proof):
    return not HARD_BODY_BLOCKERS.intersection(proof.get('blockers', ()))


def hybrid(proof, face_mean, face_floor):
    return (not proof or 'face_contradiction' not in proof['blockers']) and (
        face_mean is None or face_mean + 1e-6 >= face_floor)
