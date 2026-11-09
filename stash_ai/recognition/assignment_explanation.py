"""Project retained evidence into observation-level explanations without replay."""
from .decision_evidence import REASONS

BODY_REASONS = {
    'measured_face_association': 'This body was associated with an assigned face in the same cue.',
    'face_rooted_body_view': 'The body matched a view supported by tracks with identified face endpoints.',
    'face_rooted_crossing': 'The body was recovered through a crossing between identified face endpoints.',
    'fragment_face_endpoints': 'Face evidence at track endpoints linked this fragment to the owner.',
    'external_rear_view': 'The body matched a certified view from another scan.',
    'frozen_face_body_bank': 'The body matched references established by face associations.',
    'body_scene_core': 'Independent body observations established a scene core.',
    'body_core_attachment': 'The body matched a scene core as an auxiliary observation.',
    'body_confirmed_core_attachment': 'The body matched a confirmed core during recovery.',
    'local_body_track': 'The body was linked through a scene track to its recorded root.',
    'joint_body_claim': 'Joint resolution assigned the body among competing claims in the same cue.',
    'anonymous_body_track': 'A provisional track groups body observations without an identified owner.',
}

# Select only the proof for the recorded route. A candidate can also retain
# scores and unsuccessful probes from earlier stages.
FACE_PROOF = {
    'anchor_view_identity': 'anchor_view', 'view_identity': 'view_prototype',
    'auxiliary_identity': 'strict_confirmation', 'same_scene_identity': 'same_scene_context',
    'global_view_identity': 'global_view', 'cross_scene_identity': 'cross_scene',
    'cross_scene_provisional': 'cross_scene', 'historical_identity': 'historical_reference',
    'historical_view_identity': 'historical_reference', 'pooled_view_identity': 'provisional_pool',
    'global_reference_attachment': 'reference_registry', 'shared_view_identity': 'shared_reference',
    'final_recovery_identity': 'final_recovery', 'strict_appearance_identity': 'strict_appearance',
    'pose_uncertain_identity': 'pose_recovery', 'selected_view_identity': 'cohort_resolution',
    'selected_cohort_identity': 'cohort_resolution', 'historical_cohort_identity': 'cohort_resolution',
    'late_strict_identity': 'historical_reference',
}
MEASUREMENTS = ('score', 'face_score', 'similarity', 'match_score', 'mean_score', 'minimum',
                'track_floor', 'threshold', 'match_threshold', 'cohesion_threshold', 'ambiguity_margin',
                'required_margin', 'independent_support', 'anchor_support', 'rival_score', 'runner', 'rival',
                'rival_scope', 'perspective')


def references(proof, kind, linked_face=None):
    result = []
    seen = set()

    def add(address, label, source_kind=kind):
        if isinstance(address, str) and (address, label, source_kind) not in seen:
            result.append(dict(kind=source_kind, observation=address, label=label))
            seen.add((address, label, source_kind))

    add(linked_face, 'Linked face', 'face')
    add(proof.get('root'), 'Track root')
    add(proof.get('medoid'), 'View medoid')
    for key in ('witness_ids', 'support_ids', 'support_examples', 'source_reference_ids',
                'owner_reference_ids', 'medoids', 'local_roots'):
        for address in proof.get(key) or ():
            add(address, key.replace('_', ' ').capitalize())
    support = proof.get('support')
    if isinstance(support, list):
        for item in support:
            if isinstance(item, dict):
                add(item.get('observation'), 'Supporting cue')
    for item in proof.get('view_evidence', ()):
        add(item.get('medoid'), 'View medoid')
        for address in item.get('support_ids', ()):
            add(address, 'View support')
    for index, track in enumerate(proof.get('source_tracks') or (), 1):
        if isinstance(track, (list, tuple)) and len(track) == 3:
            start, end, witness = track
            for address, label in ((start, 'start'), (witness, 'witness'), (end, 'end')):
                add(address, f'Track {index} {label}')
    return result


def merge_path(source, merges, target):
    """Only include a chain that actually reaches the reported identity."""
    result, seen = [], set()
    while source in merges and source not in seen:
        seen.add(source)
        row = merges[source]
        result.append(dict(source=source, target=row['target'], route=row['evidence'].get('route'),
                           proof_location='identity_merges'))
        source = row['target']
    return result if source == target else []


def face(decisions, assignment, merges, target):
    variants = []
    for item in decisions:
        if not item['supports_membership'] or (
                assignment['evidence_status'] == 'recorded_current_policy' and not item['matches_current_policy']):
            continue
        payload = item['recorded']
        reason = payload.get('reason')
        origin = payload.get('body_merge_origin') or {}
        original = origin.get('reason') or (payload.get('prior_reason') if reason == 'hybrid_identity' else None)
        route = original or reason
        proof = payload.get(FACE_PROOF.get(route)) or {}
        data = {k: proof[k] for k in MEASUREMENTS if proof.get(k) is not None}
        if (route in ('known_identity', 'candidate_group', 'same_scene_identity', 'auxiliary_identity', 'view_identity')
                and payload.get('scores')):
            data['score'] = payload['scores'][0]
            if len(payload['scores']) > 1:
                data['runner'] = payload['scores'][1]
        source = origin.get('person') or payload.get('merged_from') or item['person_id']
        path = merge_path(source, merges, target)
        for field in ('same_scene_merge', 'global_view_merge', 'final_subcomponent_merge'):
            if payload.get(field):
                path.insert(0, dict(source=payload.get('merged_from'), target=None, route=field,
                                    proof_location=field))
        variants.append(dict(reason=route, summary=REASONS.get(route, ('unknown', item['explanation']))[1],
                             final_reason=reason, policy=item['policy'], metrics=data,
                             references=references(proof, 'face'), merge_source=source if source != target else None,
                             merges=path))
    return dict(evidence_status=assignment['evidence_status'], status='accepted',
                role=assignment['role'], decisions=variants)


def body(proof, *, linked_face, role, status, policy, current, merges, target):
    reason = proof.get('reason')
    metrics = {k: proof[k] for k in MEASUREMENTS if proof.get(k) is not None}
    # Body matchers use -1 as a sentinel when there was no competing owner.
    if metrics.get('runner') == -1 and proof.get('rival') is None:
        metrics.pop('runner')
    summary = BODY_REASONS.get(proof.get('route'), BODY_REASONS.get(reason))
    merge = proof.get('global_merge') or proof.get('hybrid_merge') or {}
    source = merge.get('source')
    return dict(evidence_status='recorded_current_policy' if current else 'recorded_other_policy',
                status=status, role=role, decisions=[dict(reason=reason, summary=summary or
                'See the retained body decision for this assignment.', policy=policy, metrics=metrics,
                references=references(proof, 'body', linked_face), merge_source=source,
                merges=merge_path(source, merges, target))])
