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
    ids = target_ids | feasibility.addresses
    result = latest_decisions(target, ids)
    membership = dict(target.execute('SELECT o.id,m.person FROM observations o LEFT JOIN members m '
        'ON m.observation=o.id WHERE o.id IN (SELECT value FROM json_each(?))', (encoded(sorted(ids)),)))
    from .body_reports import members as body_members
    membership.update((k,v) for k,v in body_members(target,settings).items() if k in ids)
    targets = deficits(goals, membership, membership)
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
    directed = list(queries['proposals']) + component_proposals(settings, hybrid, targets=targets) + separated + hybrid_changes
    # Clear simultaneous query and reference blockers under one configuration.
    if queries['proposals'] and hybrid_changes and any(p['relation'] == 'within_set' for p in hybrid.get('pairs', ())):
        joint = dict(hybrid_changes[0])
        for key,value in queries['proposals'][0].items():
            joint[key] = min(joint.get(key,value), value)
        directed.insert(0, joint)
    if directed and quality['proposals']:
        directed.append(dict(directed[0], **quality['proposals'][0]))
    latest_feedback = {}
    latest_feedback.update(query={k:v for k,v in queries.items() if k not in ('proposals','options')},
        separation_intervals=separation['intervals'], quality=quality['observations'],
        proposals=dict(hybrid=directed, face=
            [dict(p, **q) for p in queries['face_proposals'][:2] for q in quality['proposals'][:1]] +
            queries['face_proposals'] + quality['proposals']),
        targets=targets,
        options=updated, hybrid_proposals=list(hybrid['proposals']),
        provenance=diagnostic_record(settings, result, queries, public(hybrid), quality['observations']))
    if targeted:
        from .merge_solver_targeted import plan as targeted_plan
        latest_feedback['targeted'] = targeted_plan(target, settings, goals, feasibility, membership, latest_feedback, hybrid)
    if excluded:
        latest_feedback['scene_exclusions'] = {k: v for k, v in excluded.items() if k not in ('proposals', 'options')}
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
