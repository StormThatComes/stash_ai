"""Feasible-first, face-first tightening on fully replayed configurations.

Bounds are local experiments, not monotonicity or optimality certificates.
Admission filters are restored only up to the input policy: discarding more
evidence is not a substitute for stricter matching.
"""
from collections import Counter
from math import isclose

from .config import digest
from .merge_report_policy import parameter
from .merge_solver_domains import registry, normalize
from .merge_solver_metrics import scale


FACE = ('recognition.match_threshold', 'clustering.cohesion_threshold',
        'recognition.ambiguity_margin', 'recognition.min_identity_support',
        'recognition.min_merge_support')
VIEW = tuple('clustering.'+k for k in ('view_match_threshold', 'view_cohesion_threshold',
        'view_ambiguity_margin', 'view_min_support', 'view_attach_min_support'))
AUXILIARY = ('clustering.auxiliary_min_support',)
CONTEXT = ('recognition.cooccurrence_merge_threshold', 'clustering.same_scene_min_support',
           'clustering.same_scene_match_bonus')
HYBRID = tuple('sapiensid.matching.'+k for k in
               ('face_floor', 'person_threshold', 'ambiguity_margin', 'min_support'))
BODY = tuple('sapiensid.'+section+'.'+key for section in ('identities', 'global_matching')
             for key in ('match_threshold', 'cohesion_threshold', 'ambiguity_margin', 'min_support')) + (
             'sapiensid.identities.attach_min_support', 'sapiensid.identities.min_appearance_distance')
MATCHING = frozenset(FACE+VIEW+AUXILIARY+CONTEXT+HYBRID+BODY)


def body_supported(row):
    return bool(row.get('hybrid_merge') or row.get('reason') == 'body_global_identity' or
        row.get('hybrid_query') and row['hybrid_query'].get('signal') != 'face')


def route_parameters(decisions):
    """Only production routes used by accepted target evidence supply rewards."""
    counts = Counter()
    from .reference_roles import STRICT_REASONS
    for row in decisions:
        if row['status'] != 'accepted':
            continue
        reason = row.get('reason') or ''
        keys = set()
        if row.get('evidence_kind') == 'person':
            if reason != 'measured_face_association':
                keys.update(k for k in BODY if '.identities.' in k)
            if row.get('body_global'):
                keys.update(k for k in BODY if '.global_matching.' in k)
        else:
            if reason in STRICT_REASONS or reason == 'auxiliary_identity':
                keys.update(FACE)
            if reason == 'auxiliary_identity':
                keys.update(AUXILIARY)
            if 'view' in reason or row.get('face_view'):
                keys.update(FACE+VIEW)
            if reason == 'same_scene_identity':
                keys.update(FACE+CONTEXT)
            if row.get('cooccurring_merge'):
                keys.add('recognition.cooccurrence_merge_threshold')
            if reason == 'body_global_identity':
                keys.update(k for k in BODY if '.global_matching.' in k)
            if row.get('hybrid_merge'):
                keys.update(HYBRID)
            if row.get('hybrid_query'):
                if row['hybrid_query'].get('signal') == 'face':
                    keys.update(FACE+AUXILIARY)
                else:
                    keys.update(HYBRID)
        counts.update(keys)
    return dict(sorted(counts.items()))


def tier(key):
    if key not in MATCHING:
        return 0  # Undo quality relaxation before raising matching requirements.
    if key in FACE:
        return 1
    if key in VIEW+AUXILIARY+CONTEXT:
        return 2
    if key == HYBRID[0]:
        return 3  # A body match must retain the strongest feasible face floor.
    return 4


class Tightener:
    def __init__(self, config):
        self.config, self.domains = config, registry(config)
        self.parameters = ()
        self.matching = frozenset()
        self.first_solution = None
        self.fallback_references = 0
        self.known = {}
        self.issued = set()
        self.executed = 0
        self.exhausted = False

    def begin(self, name, changes, feedback):
        if self.first_solution is not None:
            return
        self.first_solution = name
        self.fallback_references = feedback.get('fallback_references', 0)
        active = feedback.get('matching_parameters', {})
        self.matching = MATCHING.intersection(active)
        keys = set(active) | set(changes)
        self.parameters = tuple(sorted((k for k in keys if k in self.domains
            and self.domains[k].kind is not bool and parameter(self.config.data, k) is not None),
            key=lambda k: (tier(k) if k in self.matching else 0, k)))

    def value(self, changes, key):
        return changes.get(key, parameter(self.config.data, key))

    def face_preference(self, fallback_references):
        # Tight thresholds must not win by switching previously face-supported
        # reference evidence to body matching. Withholding auxiliary evidence
        # earns no route reward; necessary initial body support remains allowed.
        return -max(0, fallback_references-self.fallback_references)

    def score(self, changes):
        values = [0.] * 5
        for key in self.parameters:
            domain = self.domains[key]
            original, value = parameter(self.config.data, key), self.value(changes, key)
            if key not in self.matching:
                # No reward for filtering beyond the original admission policy.
                amount = -max(0., domain.relax*(value-original))/scale(key, original)
            elif domain.kind is int:
                amount = value/(value+scale(key, original))
            else:
                amount = -domain.relax*(value-domain.low)/max(1e-6, domain.high-domain.low)
            values[tier(key) if key in self.matching else 0] += amount
        return tuple(values)

    def observe(self, changes, accepted):
        self.known[digest(changes)] = (dict(changes), accepted)

    def limit(self, key):
        if key not in self.matching:
            return parameter(self.config.data, key)
        domain = self.domains[key]
        return domain.low if domain.relax > 0 else domain.high

    def tasks(self, source):
        from .merge_solver_constraints import learned_groups
        # Explicit effective max() gates and cohesion dependencies, plus blockers
        # observed together in the production pipeline. Singles alone can miss
        # a route switch after a stronger core has been rebuilt.
        groups = [('recognition.match_threshold', 'clustering.cohesion_threshold'),
                  ('recognition.match_threshold', 'clustering.view_match_threshold'),
                  ('recognition.ambiguity_margin', 'clustering.view_ambiguity_margin'),
                  ('recognition.min_merge_support', 'clustering.view_min_support'),
                  HYBRID[:2]]
        groups += [tuple('sapiensid.'+section+'.'+k for k in ('match_threshold', 'cohesion_threshold'))
                   for section in ('identities', 'global_matching')]
        groups += list(learned_groups(source))
        seen = set()
        groups = [tuple(k for k in keys if k in self.parameters) for keys in groups]
        for key in self.parameters:
            for keys in [(key,)] + [g for g in groups if key in g]:
                if keys and keys not in seen:
                    seen.add(keys)
                    yield keys

    def candidate(self, changes, keys, options):
        start = {k:self.value(changes,k) for k in keys}
        end = {k:self.limit(k) for k in keys}
        keys = tuple(k for k in keys if (end[k]-start[k])*(-self.domains[k].relax) > 0)
        if not keys:
            return None
        context = {k:v for k,v in changes.items() if k not in keys}
        upper = 1.
        failed = False
        # Reuse a boundary only on the same ray with every other setting equal.
        # Clustering is not globally monotone; a changed core invalidates bounds.
        for previous, accepted in self.known.values():
            if accepted or {k:v for k,v in previous.items() if k not in keys} != context:
                continue
            fractions = [(self.value(previous,k)-start[k])/(end[k]-start[k]) for k in keys]
            if (0 < fractions[0] <= upper and all(isclose(t, fractions[0], abs_tol=1e-6)
                                                for t in fractions)):
                upper, failed = fractions[0], True
        fraction = upper/2 if failed else 1.
        # Prefer measured boundaries when near the centre of the remaining
        # interval; do not get stuck walking adjacent six-decimal samples.
        if len(keys) == 1:
            key = keys[0]
            measured = [(v-start[key])/(end[key]-start[key]) for v in options.get(key, ())]
            useful = [t for t in measured if upper*.2 <= t <= upper*.8]
            if useful:
                fraction = min(useful, key=lambda t: (abs(t-upper/2), t))
        values = {}
        for key in keys:
            domain = self.domains[key]
            step = 1 if domain.kind is int else 1e-6
            distance = abs(end[key]-start[key]) * fraction
            if distance < step*.999999:
                # Integer search must still test the next count at a midpoint.
                distance = step
            value = domain.project(start[key] - domain.relax*distance)
            if failed and (value-start[key])/(end[key]-start[key]) >= upper-1e-12:
                return None
            values[key] = value
        patch = normalize(self.config, dict(changes, **values))
        # Normalization may couple gates, but may never silently relax another.
        if any(self.domains[k].relax*(self.value(patch,k)-self.value(changes,k)) > 0
               for k in patch.keys() | changes.keys()):
            return None
        if self.score(patch) <= self.score(changes):
            return None
        return patch

    def next(self, parent, changes, source):
        self.exhausted = False
        for keys in self.tasks(source):
            patch = self.candidate(changes, keys, source.get('options', {}))
            if patch is None:
                continue
            signature = digest(patch)
            if signature in self.known or signature in self.issued:
                continue
            self.issued.add(signature)
            return 'tightening', parent, 'tighten_joint' if len(keys)>1 else 'tighten_boundary', patch
        self.exhausted = True
        raise StopIteration

    def report(self, changes=None):
        boundaries = []
        if changes is not None:
            for key in self.parameters:
                current = self.value(changes, key)
                context = {k:v for k,v in changes.items() if k != key}
                rejected = [self.value(p,key) for p,accepted in self.known.values() if not accepted
                    and {k:v for k,v in p.items() if k != key} == context
                    and (self.value(p,key)-current)*self.domains[key].relax < 0]
                boundaries.append(dict(parameter=key, selected=current,
                    nearest_rejected=min(rejected,key=lambda v:abs(v-current)) if rejected else None,
                    direction='decrease' if self.domains[key].relax > 0 else 'increase'))
        return dict(first_solution=self.first_solution, executed=self.executed,
            parameters=list(self.parameters), local_proposals_exhausted=self.exhausted,
            matching_parameters=sorted(self.matching),
            initial_body_supported_face_references=self.fallback_references,
            selected_boundaries=boundaries,
            priority=['restore_admission', 'face_core', 'face_views_and_attachments', 'hybrid_face_floor', 'body'],
            interpretation='Completed production replays only; bounds depend on all other settings. '
                           'No global optimum or future false-merge guarantee.')
