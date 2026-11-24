"""Finite, comparable construction milestones; never part of winner ranking."""


def measurements(status, mechanisms):
    result = {}
    # Fixed goal-set counts come from completed production assignments. A
    # different winning identity or a smaller numeric threshold is not progress.
    for group in status.get('construction', {}).get('groups', ()):
        prefix = f"set:{group['index']}:"
        result[prefix+'scenes'] = min(group['covered_scenes'], group['required_scenes'])
        result[prefix+'core'] = min(group['core_support'], group['core_required'])
    for cohort in (mechanisms or {}).get('cohorts', ()):
        prefix = 'cohort:'+cohort['key']+':'
        if 'independent_support' in cohort:
            result[prefix+'support'] = cohort['independent_support']
        # Cohort keys identify fixed retained witnesses. Compare only reasons
        # explicitly measured on those witnesses, never absence in a sampled log.
        for reason in ('anchor_admission', 'strict_reference_authority', 'independent_support', 'cohesion'):
            result[prefix+'cleared:'+reason] = int(reason not in cohort.get('blockers', ()))
    return result
