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
    values = dict(outside, new_outsider_observations=max(0,
        sum(g['outsider_observations'] for g in status['groups'])-
        sum(g['outsider_observations'] for g in baseline_status['groups'])))
    violations = [dict(reason='outside_impact_limit', metric=k, actual=values[k], maximum=v)
                  for k, v in limits.items() if values[k] > v]
    status.update(outside_preserved=not violations, outside_limits=dict(limits), outside_violations=violations)
    status['admissible'] &= not violations
    status['fully_solved'] &= not violations
