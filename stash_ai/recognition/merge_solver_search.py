"""Budgeted exploration independent of frozen catalogue nominations."""
from collections import deque
from itertools import islice
import math

from .merge_report_policy import parameter

PREFIX = 'sapiensid.matching.'
GATES = ('face_floor', 'person_threshold', 'ambiguity_margin', 'min_support')
# One joint proposal plus three rounds over the four independent gates.
BASELINE_WARMUP = 1 + 3 * len(GATES)


def interleave(streams):
    pending = deque(iter(s) for s in streams)
    while pending:
        stream = pending.popleft()
        try:
            yield next(stream)
        except StopIteration:
            continue
        pending.append(stream)


def spread(values, current):
    """Keep every measured value, exploring beyond densely packed near-ties."""
    def side(ordered):
        if not ordered:
            return
        yield ordered[0]
        intervals = deque([(1, len(ordered))])
        while intervals:
            low, high = intervals.popleft()
            if low >= high:
                continue
            middle = (low + high) // 2
            yield ordered[middle]
            intervals.extend(((low, middle), (middle + 1, high)))
    values = set(values)
    yield from interleave((side(sorted((v for v in values if v < current), reverse=True)),
                           side(sorted(v for v in values if v > current))))


def gate_values(key, current, measured):
    if key.endswith('min_support'):
        local = (max(1, current-1), max(1, current//2), 1, current+1)
    elif key.endswith('ambiguity_margin'):
        local = (current/2, 0., current/4, min(1., current+.02), min(1., current*2))
    else:
        # Loosen independently before combining gates. Tightening remains useful
        # for separating requested sets and rejecting newly admitted competitors.
        local = [round(max(0., current-step), 6) for step in (.02, .05, .1, .2, .01)]
        local += [round(min(1., current+step), 6) for step in (.02, .05, .1, .2, .01)]
    seen = {current}
    for value in interleave((local, spread(measured, current))):
        if value not in seen:
            seen.add(value)
            yield value


def hybrid_neighbours(config, changes, options, measured):
    def joints():
        for patch in tuple(measured):
            if any(value != changes.get(key, parameter(config.data, key)) for key, value in patch.items()):
                yield dict(changes, **patch)
    def coordinate(key):
        current = changes.get(key, parameter(config.data, key))
        for value in gate_values(key, current, options.get(key, ())):
            yield dict(changes, **{key: value})
    joint = joints()
    yield from islice(joint, 1)
    coordinates = interleave(coordinate(PREFIX + gate) for gate in GATES)
    # Long pair-proposal lists cannot defer the initial single-gate probes.
    yield from islice(coordinates, BASELINE_WARMUP - 1)
    yield from interleave((coordinates, joint))


def refinement(config, changes, options):
    """Small local probes and reversions after useful coarse progress stops."""
    streams = []
    for key, current in sorted(changes.items()):
        original = parameter(config.data, key)
        values = [original]
        if type(current) is not bool:
            if key.endswith('support'):
                values += [max(1, current-1), current+1]
            else:
                lower = max((v for v in options.get(key, ()) if v < current), default=None)
                upper = min((v for v in options.get(key, ()) if v > current), default=None)
                values += [round((v+current)/2, 6) for v in (lower, upper) if v is not None]
                # Proportional probes also work for pixel/sharpness gates.
                step = .005 if current <= 1 else max(1., abs(current)*.01)
                values += [round(max(0., current+sign*step/divisor), 6)
                           for divisor in (1, 2, 5) for sign in (-1, 1)]
        streams.append([dict(changes, **{key:value}) for value in dict.fromkeys(values)
                        if value != current])
    yield from interleave(streams)


class BeamSearch:
    """Five diverse completed states, adaptive family budgets, bounded patience.

    Baseline exploration has its own reserved lane. Losing trials can supply
    measured thresholds, but only admitted beam states acquire descendants.
    """
    width = 5
    refinement_budget = 24

    def __init__(self, families, factory, maximum):
        self.families, self.factory, self.maximum = tuple(families), factory, maximum
        self.beam, self.streams, self.root = {}, {}, {}
        self.stats = {f: dict(executed=0, improvements=0, last=0) for f in families}
        self.executed = self.stale = self.explored = 0
        self.best_quality = None
        self.patience = max(24, min(64, maximum//4))
        self.refining = False
        self.refined = 0
        self.refinements = {}
        self.stop_reason = None
        self.peak_beam = 0
        self.admitted = self.pruned = 0
        self.parent_turn = 0

    def observe(self, name, changes, rank, outcome, *, family=None):
        # Patch-size tie-breaks must not turn a plateau into endless "progress".
        improved = self.best_quality is None or rank[:-2] > self.best_quality
        if improved:
            self.best_quality, self.stale = rank[:-2], 0
            self.refining = False
            self.refinements.clear()
        elif name != 'baseline':
            self.stale += 1
        if family is not None:
            self.executed += 1
            stat = self.stats[family]
            stat['executed'] += 1
            stat['improvements'] += int(improved)
            stat['last'] = self.executed
        candidate = dict(changes=changes, rank=rank, outcome=outcome)
        pool = dict(self.beam, **{name: candidate})
        ordered = sorted(pool, key=lambda n: pool[n]['rank'], reverse=True)
        selected, signatures = [], set()
        # Prefer distinct partitions; reserve two places for distinct parameter
        # routes on a plateau (e.g. face vs body), without retaining every trial.
        for key, limit in ((lambda c: c['outcome'], 3),
                           (lambda c: (c['outcome'], tuple(sorted(c['changes']))), self.width)):
            signatures = {key(pool[n]) for n in selected}
            for n in ordered:
                signature = key(pool[n])
                if n in selected or signature in signatures:
                    continue
                selected.append(n)
                signatures.add(signature)
                if len(selected) >= limit:
                    break
        self.beam = {n: pool[n] for n in selected[:self.width]}
        self.peak_beam = max(self.peak_beam, len(self.beam))
        if name in self.beam:
            self.admitted += 1
        else:
            self.pruned += 1
        self.streams = {k:v for k,v in self.streams.items() if k[0] in self.beam}
        self.refinements = {k:v for k,v in self.refinements.items() if k in self.beam}
        if self.refining and name in self.beam:
            self.refinements[name] = iter(self.factory(name, changes, None, True))
        if name == 'baseline':
            self.root = {f: iter(self.factory(name, {}, f, False)) for f in self.families}
        return improved

    def _family_order(self):
        def priority(family):
            stat = self.stats[family]
            # A useful family gets more trials; starvation is capped at four.
            overdue = self.executed-stat['last'] >= 4
            score = (stat['improvements']/(stat['executed']+1) +
                     math.sqrt(2*math.log(self.executed+2)/(stat['executed']+1)))
            return stat['executed'] == 0, overdue, score
        return sorted(self.families, key=priority, reverse=True)

    def next(self):
        if self.refining and self.refined >= self.refinement_budget:
            self.stop_reason = 'stagnation_after_refinement'
            raise StopIteration
        if not self.refining and self.stale >= self.patience:
            self.refining, self.refined = True, 0
            self.refinements = {n: iter(self.factory(n, c['changes'], None, True)) for n,c in self.beam.items()}
        if self.refining:
            while self.refinements:
                n = next(iter(self.refinements))
                stream = self.refinements.pop(n)
                try:
                    family, strategy, changes = next(stream)
                except StopIteration:
                    continue
                self.refinements[n] = stream
                return family, 'refinement', n, strategy, changes
            self.stop_reason = 'refinement_exhausted'
            raise StopIteration
        baseline_first = self.explored < BASELINE_WARMUP or self.executed % 4 == 0
        for root in (baseline_first, not baseline_first):
            for family in self._family_order():
                if root:
                    if family not in self.root:
                        continue
                    try:
                        strategy, changes = next(self.root[family])
                        return family, 'baseline_exploration', 'baseline', strategy, changes
                    except StopIteration:
                        del self.root[family]
                else:
                    parents = [n for n in self.beam if n != 'baseline']
                    if parents:
                        offset = self.parent_turn % len(parents)
                        parents = parents[offset:] + parents[:offset]
                    for n in parents:
                        key = n, family
                        if key not in self.streams:
                            self.streams[key] = iter(self.factory(n, self.beam[n]['changes'], family, False))
                        try:
                            strategy, changes = next(self.streams[key])
                            return family, 'adaptive', n, strategy, changes
                        except StopIteration:
                            continue
        self.stop_reason = 'frontier_exhausted'
        raise StopIteration

    def completed(self, phase):
        if phase == 'baseline_exploration':
            self.explored += 1
        if phase == 'refinement':
            self.refined += 1
        self.parent_turn += 1
