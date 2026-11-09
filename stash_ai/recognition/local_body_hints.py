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
        self.spare_hints = self.spare_comparisons = 0

    def start(self, visits):
        if self.allocations is not None:return
        ordered = sorted(self.buckets, key=lambda k:(max(visits.get(o,0) for o in self.buckets[k]), k))[:HINT_LIMIT]
        count = max(1,len(ordered))
        self.spare_hints = HINT_LIMIT%count
        self.spare_comparisons = COMPARISON_LIMIT%count
        self.allocations = {key:dict(hint_limit=HINT_LIMIT//count, comparison_limit=COMPARISON_LIMIT//count,
            hints=0, comparisons=0, queries=0) for key in ordered}

    def borrow(self, allocation):
        # Only redistribute completed or unallocated reservations.
        hints = min(self.spare_hints,4)
        comparisons = min(self.spare_comparisons,2048)
        allocation['hint_limit'] += hints
        allocation['comparison_limit'] += comparisons
        self.spare_hints -= hints
        self.spare_comparisons -= comparisons

    def release(self, keys):
        for key in keys:
            allocation = self.allocations[key]
            self.spare_hints += allocation['hint_limit']-allocation['hints']
            self.spare_comparisons += allocation['comparison_limit']-allocation['comparisons']
            allocation['hint_limit'] = allocation['hints']
            allocation['comparison_limit'] = allocation['comparisons']

    def allocation(self, row):
        key = self.owners.get(row['id'], self.owners.get(row.get('face')))
        return key, (self.allocations or {}).get(key)

    def report(self):
        return [dict(set=key[0],scene=key[1],selected=key in (self.allocations or {}),
                     **(self.allocations or {}).get(key,dict(hints=0,comparisons=0,queries=0)))
                for key in sorted(self.buckets)]


def observe(rows, groups, seeds, scores, guard, config, *, frozen=None):
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
    if frozen is not None:
        banks += [(mode, person, 'attach_min_support') for person, modes in frozen.modes.items()
                  for mode in modes]
    support_counts = dict(frozen.support) if frozen is not None else {}
    witnesses = {id(g): (person or g[0]['id']) for g, person, _ in banks}
    observer.local_budget.start(observer.visits)
    completed_keys = set()
    for row in rows:
        if row['id'] not in observer.sets and row.get('face') not in observer.sets:
            continue
        key, allocation = observer.local_budget.allocation(row)
        # Exhaustion in an early scene must leave later goals' reservations intact.
        if allocation is not None:
            completed_keys.add(key)
            if len(banks)>1 or allocation['hints']>=allocation['hint_limit'] or allocation['comparisons']>=allocation['comparison_limit']:
                observer.local_budget.borrow(allocation)
        if allocation is None or allocation['hints'] >= allocation['hint_limit']:
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
                whole = seeds[person] if person is not None else group
                if copresent([row], whole) or not guard.allows(row, whole, person):
                    continue
                if len(group)>256:
                    mean,minimum = paged_score(row,group,scores)
                else:
                    mean,minimum = scores.compare([row], group)
                if id(group) not in support_counts:
                    support_counts[id(group)] = len(independent(group,settings))
                support = support_counts[id(group)]
                if support:
                    choices.append(dict(owner=witnesses[id(group)], observation=row['id'], witnesses=[r['id'] for r in group[:32]],
                        mean=mean, minimum=minimum, support=support, support_gate=gate))
        except ResourceLimit:
            allocation['incomplete_rankings'] = allocation.get('incomplete_rankings',0)+1
            observer.truncation_reasons.add('local_body_budget')
            continue
        finally:
            allocation['comparisons'] += scores.count-start
        choices.sort(key=lambda q: (-q['mean'], q['witnesses']))
        for index, choice in enumerate(choices[:4]):
            runner = max((q['mean'] for j, q in enumerate(choices) if q['owner'] != choice['owner']), default=-1.)
            if choice['minimum'] >= 0 and choice['mean'] >= runner:
                if allocation['hints'] < allocation['hint_limit']:
                    observer.local_body.append(dict(choice, gap=min(1., choice['mean']-runner)))
                    allocation['hints'] += 1
    observer.local_budget.release(completed_keys)
    observer.local_body_comparisons += scores.count


def paged_score(row,group,scores):
    from .body_identity import quality
    seen = set()
    total = 0.
    low = float('inf')
    for start in range(0,len(group),256):
        for witness in group[start:start+256]:
            score = scores.face_similarity(row['_vector'],witness['_vector'])
            low = min(low,score)
            pixel = quality(witness).get('pixel_digest') or ('observation',witness['id'])
            if pixel not in seen:
                total += score
                seen.add(pixel)
    return total/len(seen),low
