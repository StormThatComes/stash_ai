"""Frozen goal deficits used to order experiments, never to assign identities."""
from .merge_goal_sets import assess_set, ownership, set_destinations
from .merge_goal_index import evaluation_scope


@evaluation_scope
def deficits(goals, observations, predicted):
    if goals.get('goal_policy', {}).get('name') != 'identity':
        return dict(priorities={}, destinations={}, sets=[])
    owners, raw = ownership(goals, observations, predicted)
    priorities, destinations, sets = {}, {}, []
    for group in goals['groups']:
        index, evidence = group['index'], group['identity_evidence']
        owned = set_destinations(index, raw, owners)
        result = assess_set(evidence, observations, predicted, owned)
        target = result['set_target']
        missing = [s for s, support in result['scene_support'].items() if not support.get(target)]
        core_gap = max(0, evidence['core_required']-result['set_core_destinations'].get(target, 0))
        sets.append(dict(set=index, target=target, missing_target_scenes=missing, core_deficit=core_gap))
        for scene, frozen in evidence['scenes'].items():
            for unit in frozen['scoring_units']:
                for row in unit['observations']:
                    if predicted.get(row['id']) != target or target is None:
                        priorities[row['id']] = max(priorities.get(row['id'], 0),
                            4*int(scene in missing)+int(core_gap > 0)+int(predicted.get(row['id']) is None))
        for destination in owned:
            destinations[destination] = dict(set=index, target=target,
                scenes=[s for s, support in result['scene_support'].items() if support.get(destination)],
                required_scenes=result['required_scenes'], core_required=evidence['core_required'],
                core=result['set_core_destinations'].get(destination, 0))
    return dict(priorities=priorities, destinations=destinations, sets=sets)


def bridge_gain(left, right, targets):
    """Prefer complementary coverage and useful core, rather than dense repeats."""
    profiles = targets.get('destinations', {})
    a, b = profiles.get(left), profiles.get(right)
    if not a or not b or a['set'] != b['set']:
        return 0.
    scenes = len(set(a['scenes']) ^ set(b['scenes']))/max(1, a['required_scenes'])
    core = min(a['core'], b['core'], max(0, a['core_required']-max(a['core'], b['core'])))
    return scenes+core/max(1, a['core_required'])
