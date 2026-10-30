"""Bounded local-body witnesses for solver proposals; never matching decisions."""
from .hybrid_query_observer import _observer


def observe(rows, groups, seeds, scores, guard, config):
    observer = _observer.get()
    if observer is None or not hasattr(observer, 'local_body'):
        return
    from .body_identity import copresent, independent, Scores
    from .body_guards import FaceGuard
    from .solver_measurement_budget import settings as diagnostic_settings
    from .resource_errors import ResourceLimit
    settings = config.data['sapiensid']['identities']
    # Diagnostic comparisons cannot consume the production scorer's budget.
    scores = Scores(diagnostic_settings(config))
    scores.limit = max(0, 16384-observer.local_body_comparisons)
    guard = FaceGuard(guard.db, config, scores=scores)
    banks = [(g, None, 'min_support') for g in groups]
    banks += [(g, person, 'attach_min_support') for person, g in seeds.items()]
    for row in rows:
        if row['id'] not in observer.sets and row.get('face') not in observer.sets:
            continue
        # A complete rival pass is required; an unfinished row yields no hint.
        if len(observer.local_body) >= 128 or sum(len(g) for g, _, _ in banks) > 4096:
            observer.truncation_reasons.add('local_body_budget')
            break
        choices = []
        try:
            for group, person, gate in banks:
                if any(r['id'] == row['id'] for r in group):
                    continue
                if copresent([row], group) or not guard.allows(row, group, person):
                    continue
                mean, minimum = scores.compare([row], group)
                support = len(independent(group, settings))
                if support:
                    choices.append(dict(observation=row['id'], witnesses=[r['id'] for r in group],
                        mean=mean, minimum=minimum, support=support, support_gate=gate))
        except ResourceLimit:
            observer.truncation_reasons.add('local_body_budget')
            break
        choices.sort(key=lambda q: (-q['mean'], q['witnesses']))
        for index, choice in enumerate(choices[:4]):
            runner = max((q['mean'] for j, q in enumerate(choices) if j != index), default=-1.)
            if choice['minimum'] >= 0 and choice['mean'] >= runner:
                if len(observer.local_body) < 128:
                    observer.local_body.append(dict(choice, gap=min(1., choice['mean']-runner)))
    observer.local_body_comparisons += scores.count
