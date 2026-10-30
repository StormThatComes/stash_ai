"""Bounded local-body witnesses for solver proposals; never matching decisions."""
from .hybrid_query_observer import _observer

HINT_LIMIT = 128
COMPARISON_LIMIT = 16384


class Budget:
    """Reserve work per goal set and scene before the first scene is visited."""
    def __init__(self, goals):
        self.owners, self.buckets = {}, {}
        for group in goals['groups']:
            for member in group['members']:
                for row in member['observations']:
                    key = group['index'], row.get('scene', 'unknown')
                    addresses = [row['id'], *goals.get('physical_alternatives', {}).get(row['id'], ())]
                    self.buckets.setdefault(key, set()).update(addresses)
                    self.owners.update((oid,key) for oid in addresses)
        self.allocations = None

    def start(self, visits):
        if self.allocations is not None:return
        ordered = sorted(self.buckets, key=lambda k:(max(visits.get(o,0) for o in self.buckets[k]), k))[:HINT_LIMIT]
        count = max(1,len(ordered))
        self.allocations = {key:dict(hint_limit=HINT_LIMIT//count, comparison_limit=COMPARISON_LIMIT//count,
            hints=0, comparisons=0, queries=0) for key in ordered}

    def allocation(self, row):
        key = self.owners.get(row['id'], self.owners.get(row.get('face')))
        return key, (self.allocations or {}).get(key)

    def report(self):
        return [dict(set=key[0],scene=key[1],selected=key in (self.allocations or {}),
                     **(self.allocations or {}).get(key,dict(hints=0,comparisons=0,queries=0)))
                for key in sorted(self.buckets)]


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
    guard = FaceGuard(guard.db, config, scores=scores)
    banks = [(g, None, 'min_support') for g in groups]
    banks += [(g, person, 'attach_min_support') for person, g in seeds.items()]
    observer.local_budget.start(observer.visits)
    for row in rows:
        if row['id'] not in observer.sets and row.get('face') not in observer.sets:
            continue
        key, allocation = observer.local_budget.allocation(row)
        # Exhaustion in an early scene must leave later goals' reservations intact.
        if allocation is None or allocation['hints'] >= allocation['hint_limit']:
            observer.truncation_reasons.add('local_body_budget')
            continue
        if sum(len(g) for g, _, _ in banks) > 4096:
            observer.truncation_reasons.add('local_body_budget')
            continue
        start = scores.count
        scores.limit = start + max(0,allocation['comparison_limit']-allocation['comparisons'])
        allocation['queries'] += 1
        observer.visits[row['id']] = observer.visits.get(row['id'],0)+1
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
            continue
        finally:
            allocation['comparisons'] += scores.count-start
        choices.sort(key=lambda q: (-q['mean'], q['witnesses']))
        for index, choice in enumerate(choices[:4]):
            runner = max((q['mean'] for j, q in enumerate(choices) if j != index), default=-1.)
            if choice['minimum'] >= 0 and choice['mean'] >= runner:
                if allocation['hints'] < allocation['hint_limit']:
                    observer.local_body.append(dict(choice, gap=min(1., choice['mean']-runner)))
                    allocation['hints'] += 1
    observer.local_body_comparisons += scores.count
