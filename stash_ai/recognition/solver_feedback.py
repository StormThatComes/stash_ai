"""Trial-local feedback collection; no search coordinator state is mutated."""
from .config import digest
from .merge_goal_index import current as goal_index
from .merge_report_policy import parameter
from .merge_solver_diagnostics import diagnostic_record, latest_decisions
from .merge_solver_hybrid import public
from .merge_solver_targets import deficits
from .merge_solver_constraints import component_proposals

from .parallel_numeric import diagnostic_scope
from .dependencies import read_only

@diagnostic_scope
@read_only()
def collect(target, settings, goals, feasibility, query_feedback, initial_changes, excluded, targeted):
    from .merge_solver import measured_options, rebuilt_goals
    from .merge_solver_feedback import pair_separation, quality_feedback, face_fallbacks
    target_ids = goal_index().observation_ids
    from .database import encoded
    from .merge_scene_exclusions import constraints
    negatives = list(constraints(goals))
    ids = target_ids | feasibility.addresses | {oid for item in negatives for oid in item['observation_ids']}
    result = latest_decisions(target, ids)
    membership = dict(target.execute('SELECT o.id,m.person FROM observations o LEFT JOIN members m '
        'ON m.observation=o.id WHERE o.id IN (SELECT value FROM json_each(?))', (encoded(sorted(ids)),)))
    from .body_reports import members as body_members
    membership.update(body_members(target,settings,ids=ids))
    targets = deficits(goals, membership, membership)
    set_targets = {item['set']:item['target'] for item in targets['sets']}
    wrong = {}
    expected, forbidden_by_observation = {}, {}
    destination_sets = {}
    for group in goals['groups']:
        for member in group['members']:
            for row in member['observations']:
                owner = membership.get(row['id'])
                if owner:
                    destination_sets.setdefault(owner,set()).add(group['index'])
    for group in goals['groups']:
        wanted = set_targets.get(group['index'])
        for member in group['members']:
            for row in member['observations']:
                owner = membership.get(row['id'])
                if wanted and destination_sets.get(wanted)=={group['index']}:
                    expected[row['id']] = wanted
                if owner and len(destination_sets[owner])>1:
                    wrong[row['id']] = dict(expected=wanted,constraint='between_sets')
                elif owner and wanted and owner != wanted:
                    wrong[row['id']] = dict(expected=wanted, constraint='positive_set')
    from .positive_observations import reference_targets
    label_targets = reference_targets(goals,membership,membership) if goals.get('positive_observations') else {}
    for item in goals.get('positive_observations', ()):
        wanted,_ = label_targets[item['set']]
        for oid in item['observation_ids']:
            expected[oid] = wanted
            if membership.get(oid) != wanted or not wanted:
                wrong[oid] = dict(expected=wanted, constraint='positive_observation')
    for item in negatives:
        forbidden = {membership.get(oid) for oid in item['target_observation_ids']} - {None}
        for oid in item['observation_ids']:
            forbidden_by_observation.setdefault(oid,set()).update(forbidden)
            if membership.get(oid) in forbidden:
                wrong[oid] = dict(expected=None, constraint='exclusion')
    label_trials = []
    if goals.get('positive_observations'):
        from .positive_observations import measure, proposals
        measure(target, goals, expected, result)
        evidence_ids = {oid for row in result if 'label_probe' in row for oid in
                        [*row.get('witnesses',()),(row.get('track_blocker') or {}).get('peer')] if oid}
        evidence_owners = body_members(target,settings,ids=evidence_ids)
        label_trials = proposals(settings,goals,result,{**membership,**evidence_owners})
    # A fragmented positive set alone does not establish which branch was wrong.
    from .threshold_boundaries import strictly_above
    tighter = []
    for row in result:
        if wrong.get(row['observation'], {}).get('constraint') != 'exclusion':
            continue
        if row['reason'] not in ('face_rooted_body_view', 'external_rear_view', 'frozen_face_body_bank'):
            continue
        score = row['measurements'].get('score')
        if row['status'] == 'accepted' and score is not None:
            key = 'sapiensid.identities.match_threshold'
            threshold = strictly_above(score)
            if parameter(settings.data,key) < threshold <= 1:
                tighter.append({key:threshold})
    query_feedback.priorities = targets['priorities']
    current_goals = rebuilt_goals(target, goals)
    hybrid = {}
    updated = measured_options(target, settings, current_goals, hybrid=hybrid,
                               frozen_goals=goals if targeted else None)
    hybrid_changes = list({digest(p): p for p in hybrid['proposals'] + list(initial_changes)}.values())
    queries = query_feedback.finish(settings)
    separation = pair_separation(settings, hybrid)
    quality = quality_feedback(target, settings, goals,
        priorities=targets['priorities'] if goals.get('goal_policy', {}).get('name') == 'identity' else None)
    if not queries['positive_witnesses']:
        queries['face_routes'] = []
        queries['face_proposals'].extend(face_fallbacks(target, settings, goals, targets,
            evidence=queries['face_routes'] if targeted else None))
    separated = [dict(hybrid_changes[0] if hybrid_changes else {}, **p) for p in separation['proposals']]
    directed = [item['patch'] for item in label_trials] + tighter + list(queries['proposals']) + component_proposals(settings, hybrid, targets=targets) + separated + hybrid_changes
    # Clear simultaneous query and reference blockers under one configuration.
    if queries['proposals'] and hybrid_changes and any(p['relation'] == 'within_set' for p in hybrid.get('pairs', ())):
        joint = dict(hybrid_changes[0])
        for key,value in queries['proposals'][0].items():
            joint[key] = min(joint.get(key,value), value)
        directed.insert(0, joint)
    if directed and quality['proposals']:
        directed.append(dict(directed[0], **quality['proposals'][0]))
    latest_feedback = {}
    from .merge_solver_tighten import route_parameters, body_supported
    face_references = {r['id'] for g in goals['groups'] for m in g['members'] for r in m['observations']
                       if r.get('goal_role') == 'reference' and r.get('evidence_kind', 'face') == 'face'}
    from .scene_strategy_validation import measured_context
    if goals.get('positive_observations'):
        latest_feedback['label_trials'] = label_trials
        latest_feedback['label_probes'] = {r['observation']:r['label_probe'] for r in result if 'label_probe' in r}
    latest_feedback['scene_boundaries'] = measured_context(settings.data['sapiensid']['identities'],
        result,expected,forbidden_by_observation)
    latest_feedback.update(query={k:v for k,v in queries.items() if k not in ('proposals','options')},
        matching_parameters=route_parameters(r for r in result if r['observation'] in target_ids),
        fallback_references=sum(r['status'] == 'accepted' and r['observation'] in face_references and body_supported(r)
                                for r in result),
        separation_intervals=separation['intervals'], quality=quality['observations'],
        proposals=dict(hybrid=directed, face=
            [dict(p, **q) for p in queries['face_proposals'][:2] for q in quality['proposals'][:1]] +
            queries['face_proposals'] + quality['proposals']),
        targets=targets,
        options=updated, hybrid_proposals=list(hybrid['proposals']),
        provenance=diagnostic_record(settings, result, queries, public(hybrid), quality['observations'], wrong=wrong))
    if targeted:
        from .merge_solver_targeted import plan as targeted_plan
        latest_feedback['targeted'] = targeted_plan(target, settings, goals, feasibility, membership, latest_feedback, hybrid)
    if excluded:
        from .merge_scene_exclusions import KINDS
        for kind in KINDS:
            if goals.get(kind + '_exclusions'):
                detail = {k: v for k, v in excluded.items() if k not in ('proposals', 'options')}
                detail['evidence'] = [r for r in detail['evidence']
                    if ('requested_observation_ids' in r) == (kind == 'observation')]
                latest_feedback[kind + '_exclusions'] = detail
        for family in ('face', 'hybrid'):
            # Face-only runs still receive their negative boundary hints.
            hints = [p for p in excluded['proposals']
                     if any(k.startswith('sapiensid.') for k in p) == (family == 'hybrid')]
            latest_feedback['proposals'][family] = hints + latest_feedback['proposals'][family]
        for key, values in excluded['options'].items():
            updated.setdefault(key, []).extend(values)
    for extra in (queries['options'], separation['options']):
        for key, values in extra.items():
            updated[key] = list(updated.get(key, ())) + list(values)
    return result, latest_feedback, hybrid, hybrid_changes, updated
