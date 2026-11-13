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
    'scene_body_attachment': 'The body matched an owner with independently supported evidence in this scene.',
    'anonymous_body_track': 'A provisional track groups body observations without an identified owner.',
}

# Select only the proof for the recorded route. A candidate can also retain
# scores and unsuccessful probes from earlier stages.
FACE_PROOF = {
    'prior_reference_identity': 'historical_reference', 'hybrid_query': 'hybrid_query',
    'known_identity': 'face_comparison', 'candidate_group': 'face_comparison',
    'anchor_view_identity': 'anchor_view', 'view_identity': 'view_prototype',
    'auxiliary_identity': 'strict_confirmation', 'same_scene_identity': 'same_scene_context',
    'global_view_identity': 'global_view', 'cross_scene_identity': 'cross_scene',
    'cross_scene_provisional': 'cross_scene', 'historical_identity': 'historical_reference',
    'historical_view_identity': 'historical_reference', 'pooled_view_identity': 'provisional_pool',
    'global_reference_attachment': 'reference_registry', 'shared_view_identity': 'shared_reference',
    'final_recovery_identity': 'final_recovery', 'strict_appearance_identity': 'strict_appearance',
    'pose_uncertain_identity': 'pose_recovery', 'selected_view_identity': 'shared_reference',
    'selected_cohort_identity': 'shared_reference', 'historical_cohort_identity': 'historical_reference',
    'late_strict_identity': 'shared_reference',
}
MEASUREMENTS = ('score', 'face_score', 'similarity', 'match_score', 'mean_score', 'minimum',
                'track_floor', 'threshold', 'match_threshold', 'cohesion_threshold', 'ambiguity_margin',
                'required_margin', 'independent_support', 'anchor_support', 'rival_score', 'runner', 'rival',
                'signal', 'rival_scope', 'rival_cluster', 'perspective', 'required_support', 'reference_count', 'certified_floor', 'face_threshold')


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
                'owner_reference_ids', 'medoids', 'local_roots', 'scoring_references', 'core_observations'):
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
        original = origin.get('reason') or (payload.get('prior_reason') if reason in ('hybrid_identity', 'body_global_identity') else None)
        route = original or reason
        proof = payload.get(FACE_PROOF.get(route)) or {}
        if route == 'late_strict_identity' and not proof:
            proof = payload.get('provisional_pool') or payload.get('historical_reference') or {}
        if route in ('candidate_group', 'known_identity') and payload.get('compared_references'):
            proof = dict(proof, scoring_references=payload['compared_references'])
        data = {k: proof[k] for k in MEASUREMENTS if proof.get(k) is not None}
        for field, gate in (('threshold','match_gate_range'), ('cohesion_threshold','cohesion_gate_range')):
            if gate in proof:
                low, high = proof[gate]
                data[field] = low if low == high else f'{low:g} to {high:g} across this cohort'
        # Historical strict and relaxed-pose decisions stored a base view threshold;
        # its exact effective gate was not recorded for each scoring route.
        if 'match_gate_range' not in proof and proof.get('mode') in ('view','strict') and 'ranking_score' in proof:
            for field in ('threshold','cohesion_threshold'):
                if field in data:
                    data['base_view_' + field] = data.pop(field)
        if (route in ('known_identity', 'candidate_group', 'same_scene_identity', 'auxiliary_identity', 'view_identity')
                and payload.get('scores')):
            data['score'] = payload['scores'][0]
            if len(payload['scores']) > 1:
                data['runner'] = payload['scores'][1]
        source = origin.get('person') or payload.get('merged_from') or item['person_id']
        path = merge_path(source, merges, target)
        local_edges = payload.get('replay_merge_chain') or ()
        if local_edges:
            path = [dict(source=edge['source'], target=edge['target'], route=edge['route'],
                         proof_location='replay_merge_chain', proof_index=index)
                    for index, edge in enumerate(local_edges)] + path
        else:
            fields = [k for k in ('same_scene_merge','global_view_merge','final_subcomponent_merge') if payload.get(k)]
            path = [dict(source=payload.get('merged_from') if len(fields)==1 else None,
                         target=None, route=k, proof_location=k) for k in fields] + path
        summary = REASONS.get(route, ('unknown', item['explanation']))[1]
        if route == 'hybrid_query' and proof.get('signal')=='face':
            summary = 'The face matched a frozen reference in the hybrid query stage.'
        if proof.get('stage') == 'seed':
            summary = 'This observation started the candidate group, which subsequently met the identity support requirement.'
        elif route == 'candidate_group' and proof.get('stage') == 'match':
            summary = 'This observation matched the existing candidate group before its support was resolved.'
        variants.append(dict(reason=route, summary=summary, stage=proof.get('stage'),
                             final_reason=reason, policy=item['policy'], reference_role=payload.get('reference_role'), metrics=data,
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
    sources = [link['source'] for key in ('global_merge','hybrid_merge')
               if isinstance(link := proof.get(key), dict) and link.get('source')]
    paths = [(src, merge_path(src, merges, target)) for src in sources]
    source, path = max(paths, key=lambda item: len(item[1]), default=(None, []))
    return dict(evidence_status='recorded_current_policy' if current else 'recorded_other_policy',
                status=status, role=role, decisions=[dict(reason=reason, summary=summary or
                'See the retained body decision for this assignment.', policy=policy, metrics=metrics,
                references=references(proof, 'body', linked_face), merge_source=source,
                merges=path)])
