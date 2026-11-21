"""Independently confirm a contextual face owner for scene-local body comparisons."""
from collections import Counter, defaultdict

from .body_identity import copresent, independent, quality
from .independent_evidence import body_vote, unit
from .person_anchor import body_gate


def _distinct(row, bank, settings):
    """A repeated crop cannot confirm itself, even when retained at another cue."""
    pixels = unit(body_vote(row))
    appearance = quality(row).get('appearance_hash')
    distance = settings['min_appearance_distance']
    for ref in bank:
        if ref['cue'] == row['cue'] or unit(body_vote(ref)) == pixels:
            continue
        other = quality(ref).get('appearance_hash')
        if distance and appearance and other and (int(appearance, 16) ^ int(other, 16)).bit_count() < distance:
            continue
        yield ref


def _check(row, owner, frozen, guard, occupied, proof):
    settings = frozen.settings
    gate = body_gate(row, settings)
    if not gate['eligible']:
        return gate['reason']
    if not row['face_quality_ok'] or row.get('_face_vector') is None:
        return 'linked_face_quality'
    if owner not in frozen.seeds:
        return 'no_strict_body_references'
    if occupied[owner, row['cue']] > 1 or copresent([row], frozen.seeds[owner]):
        return 'same_cue_owner_conflict'
    if not guard.allows(row, frozen.seeds[owner], owner):
        return 'face_contradiction'

    # Score the body against every physically possible owner. The contextual
    # face must not eliminate a rival from its own independent confirmation.
    ranking = frozen.rank(dict(row, _face_vector=None), guard)
    mean, minimum, _, bank, admissible, _ = next(entry for entry in ranking if entry[2] == owner)
    runner, rival = max(((entry[5], entry[2]) for entry in ranking if entry[2] != owner), default=(-1., None))
    witnesses = independent(_distinct(row, bank, settings), settings)
    proof.update(score=mean, minimum=minimum, threshold=settings['match_threshold'],
                 cohesion_threshold=settings['cohesion_threshold'], required_margin=settings['ambiguity_margin'],
                 independent_support=len(witnesses), required_support=settings['attach_min_support'],
                 witness_ids=[ref['id'] for ref in witnesses[:8]], runner=runner, rival=rival)
    if len(witnesses) < settings['attach_min_support']:
        return 'insufficient_independent_body_support'
    # Recheck the mean using only independent witnesses. Duplicate pixels and
    # near-identical appearances must not pull a borderline match over the gate.
    if len(witnesses) < len(bank):
        values = (frozen.scores.face_similarity(row['_vector'], ref['_vector']) for ref in witnesses)
        proof['score'] = min(mean, sum(values) / len(witnesses))
    if not admissible or proof['score'] < settings['match_threshold'] or minimum < settings['cohesion_threshold']:
        return 'body_match_or_cohesion'
    if rival is not None and (proof['score'] <= runner or proof['score'] - runner < settings['ambiguity_margin']):
        return 'competing_body_identity'
    return 'confirmed'


def corroborate(contextual, rows, decisions, frozen, guard):
    occupied = Counter((d['person'], row['cue']) for row in rows
                       if (d := decisions.get(row['id'])) and d.get('person') is not None)
    additions = defaultdict(list)
    for row in contextual:
        proof = decisions[row['id']]
        if proof['reason'] != 'measured_face_association':
            continue
        owner = proof['person']
        outcome = _check(row, owner, frozen, guard, occupied, proof)
        proof.update(face_assignment='same_scene_identity', scene_anchor_check=outcome)
        if outcome == 'confirmed':
            proof.update(role='anchor', reference_scope='scene')
            additions[owner].append(row)

    # Keep strict face roots separate for track and view evidence. Only the
    # local SapiensID bank gains modes, after every promotion was decided.
    if additions:
        frozen.add_scene_anchors(additions)
