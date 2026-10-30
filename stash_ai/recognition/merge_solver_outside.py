"""Optional bounds on collateral partition changes relative to baseline replay."""

FIELDS = ('coverage_lost', 'split_pairs', 'newly_joined_pairs', 'new_outsider_observations')


def validate(limits):
    limits = {k: v for k, v in (limits or {}).items() if v is not None}
    if any(k not in FIELDS or type(v) is not int or v < 0 for k, v in limits.items()):
        raise ValueError('Outside limits must be nonnegative integer partition-change counts.')
    return limits


def apply(status, outside, baseline_status, limits):
    if not limits:
        return
    if 'new_outsider_observations' in limits and 'new_outsider_observations' not in outside:
        raise ValueError('New outsider limits require observation-level baseline differences.')
    values = outside
    violations = [dict(reason='outside_impact_limit', metric=k, actual=values[k], maximum=v)
                  for k, v in limits.items() if values[k] > v]
    status.update(outside_preserved=not violations, outside_limits=dict(limits), outside_violations=violations)
    status['admissible'] &= not violations
    status['fully_solved'] &= not violations


def outsiders(status, predicted, goal_ids):
    """Stable (goal set, observation) links, independent of destination UUIDs."""
    destinations={}
    for group in status['groups']:
        for person in group['resulting_identities']:
            destinations.setdefault(person,[]).append(group['index'])
    return {(index,oid) for oid,person in predicted.items() if oid not in goal_ids and person is not None
            for index in destinations.get(person,())}
