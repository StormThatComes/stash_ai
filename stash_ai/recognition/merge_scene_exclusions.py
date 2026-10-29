"""Explicit negative scene labels for offline goals, never matching overrides."""
from collections import defaultdict

from .database import encoded


def resolve(db, config, pairs, scope, owners):
    from .identity_report import _select
    from .models import namespace
    from .offline_evaluation import _selection
    from .body_reports import observations as body_observations
    constraints = {}
    for performer, scene in pairs:
        if any(not isinstance(v, str) or not v.isascii() or not v.isdecimal() or int(v) < 1
               for v in (performer, scene)):
            raise ValueError('Scene exclusions require positive Stash performer and scene IDs.')
        performer, scene = str(int(performer)), str(int(scene))
        _, person = _select(db, namespace(config), None, performer, scope)
        if person['id'] not in owners:
            raise ValueError(f'Excluded performer {performer} must belong to a merge goal set.')
        group = owners[person['id']]
        key = group, scene
        if key not in constraints:
            # A missing/incompatible scene cannot certify absence. This also
            # accepts a complete empty scan; no observation is invented for it.
            _, scans = _selection(db, config, [scene], scope)
            ids = {r[0] for r in db.execute('SELECT id FROM observations WHERE scan=?', (scans[0]['id'],))}
            ids.update(body_observations(db, config, [scans[0]['id']]))
            constraints[key] = dict(set=group, scene=scene, scan=scans[0]['id'],
                performer_ids=[], resolved_identity_ids=[], observation_ids=sorted(ids))
        item = constraints[key]
        item['performer_ids'] = sorted(set(item['performer_ids']) | {performer})
        item['resolved_identity_ids'] = sorted(set(item['resolved_identity_ids']) | {person['id']})
    return [constraints[k] for k in sorted(constraints)]


def freeze_targets(db, config, goals):
    """Track the declared person through splits, aliases and face/body handoffs.

    Linked accepted bodies are alternative addresses for negative checking,
    not extra votes in the positive consolidation objective.
    """
    from .body_identity import available, enabled
    from .body_observations import contract
    by_set = defaultdict(list)
    for constraint in goals.get('scene_exclusions', ()):
        by_set[constraint['set']].append(constraint)
    for group in goals['groups']:
        conditions = by_set[group['index']]
        if not conditions:
            continue
        ids = {r['id'] for m in group['members'] for r in m['observations']}
        if not ids:
            raise ValueError('Scene exclusions leave a goal set without positive evidence in another scene.')
        if enabled(config) and available(db):
            ids.update(r[0] for r in db.execute('SELECT b.id FROM body_observations b '
                'JOIN body_members m ON m.observation=b.id JOIN members f ON f.observation=b.face '
                "WHERE m.status='accepted' AND m.person=f.person AND b.rejected IS NULL AND b.contract=? "
                'AND b.face IN (SELECT value FROM json_each(?))', (contract(config), encoded(sorted(ids)))))
        for item in conditions:
            item.update(target_observation_ids=sorted(ids))


def apply(goals, status, observations, predicted):
    constraints = goals.get('scene_exclusions', ())
    if not constraints:
        return status
    wanted = {item['scene'] for item in constraints}
    scenes = defaultdict(list)
    for oid, scene in observations.items():
        if scene in wanted and predicted.get(oid) is not None:
            scenes[scene].append(oid)
    results, violations = [], []
    for item in constraints:
        destinations = {predicted[oid] for oid in item['target_observation_ids']
                        if oid in observations and predicted.get(oid) is not None}
        missing = sorted(set(item['observation_ids']) - observations.keys())
        # UUIDs can be reused by either fragment after a split. Only the frozen
        # positive evidence identifies the declared person, not UUID spelling.
        matches = sorted(oid for oid in scenes[item['scene']] if predicted[oid] in destinations)
        base = {k: item[k] for k in ('set', 'scene', 'performer_ids')}
        result = dict(base, satisfied=bool(destinations) and not missing and not matches,
            matching_count=len(matches), matching_observations=matches[:32],
            matching_identities=sorted({predicted[oid] for oid in matches}),
            missing_count=len(missing), missing_observations=missing[:32],
            observation_examples_truncated=len(matches) > 32 or len(missing) > 32,
            target_represented=bool(destinations))
        results.append(result)
        if matches:
            violations.append(dict(base, reason='excluded_performer_in_scene', count=len(matches),
                                   identities=result['matching_identities']))
        if missing:
            violations.append(dict(base, reason='excluded_scene_not_evaluated', count=len(missing)))
        if not destinations:
            violations.append(dict(base, reason='excluded_performer_unresolved'))
    satisfied = all(r['satisfied'] for r in results)
    status.update(scene_exclusions=results, scene_exclusions_satisfied=satisfied,
        scene_exclusion_violations=violations, admissible=status['admissible'] and satisfied,
        fully_solved=status['fully_solved'] and satisfied)
    return status


def summary_lines(status):
    return [f"  Exclusion: performers={','.join(r['performer_ids'])} scene={r['scene']} "
            f"satisfied={r['satisfied']} matches={r['matching_count']} "
            f"target_represented={r['target_represented']} missing_obs={r['missing_count']}"
            for r in status.get('scene_exclusions', ())]
