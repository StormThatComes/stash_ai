"""Budgeted exploration independent of frozen catalogue nominations."""
from collections import deque
from itertools import islice
import math

from .merge_report_policy import parameter
from .merge_solver_metrics import signature as parameter_signature

PREFIX = 'sapiensid.matching.'
GATES = ('face_floor', 'person_threshold', 'ambiguity_margin', 'min_support')
# One joint proposal plus three rounds over the four independent gates.
BASELINE_WARMUP = 1 + 3 * len(GATES)


def quality_gain(previous, current):
    """Scheduling reward only: the first changed objective keeps its precedence."""
    if previous is None or current <= previous:
        return 0.
    for index, (before, after) in enumerate(zip(previous, current)):
        if before != after:
            return min(1., (after-before)/max(1., abs(before), abs(after)))/(index+1)
    return 0.


def statistics():
    return dict(executed=0, improvements=0, last=0, seconds=0., rejected=0,
                gain=0., information=0., recent=[])


def cadence(stat, default):
    recent = stat.get('recent', ())[-4:]
    if len(recent) >= 2 and sum(g > 0 for g in recent)*2 >= len(recent):
        return max(1, default//2)
    if len(recent) == 4 and not any(recent):
        return min(8, default*2)
    return default


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
    body_keys=[k for k in options if k.startswith(('sapiensid.identities.','sapiensid.global_matching.'))]
    coordinates = interleave(coordinate(k) for k in body_keys+[PREFIX+gate for gate in GATES])
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
    """Three to eight diverse states, adaptive family budgets, bounded patience.

    Baseline exploration has its own reserved lane. Losing trials can supply
    measured thresholds, but only admitted beam states acquire descendants.
    """
    width = 5
    refinement_budget = 24

    def __init__(self, families, factory, maximum, config=None):
        self.config = config
        self.families, self.factory, self.maximum = tuple(families), factory, maximum
        self.beam, self.streams, self.root = {}, {}, {}
        self.stats = {f: statistics() for f in families}
        self.lanes = {}
        self.recent_states = deque(maxlen=16)
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
        self.normal_completed = self.adaptive_completed = 0
        self.last_root = 0
        self.incumbent = 'baseline'
        self.focus_due = False
        self.last_parent = None
        self.operators = {}
        self.restarts = 0
        self.restart_pending = None
        self.restart_executed = 0

    def observe(self, name, changes, rank, outcome, *, family=None, operator=None, elapsed=0., blockers=(), phase=None):
        # Patch-size tie-breaks must not turn a plateau into endless "progress".
        improved = self.best_quality is None or rank[:-2] > self.best_quality
        gain = quality_gain(self.best_quality, rank[:-2])
        state = outcome, tuple(blockers)
        information = float(state not in self.recent_states)
        self.recent_states.append(state)
        if self.config is not None and len(self.recent_states) >= 4:
            diversity = len(set(self.recent_states))
            self.width = max(3, min(8, 2+(diversity+1)//2))
        if improved:
            self.best_quality, self.stale = rank[:-2], 0
            self.refining = False
            self.refinements.clear()
            self.restart_pending = None
        elif name != 'baseline':
            self.stale += 1
        if family is not None:
            self.executed += 1
            records = [self.stats[family]]
            if operator:
                records.append(self.operators.setdefault(operator, statistics()))
            if phase:
                records.append(self.lanes.setdefault(phase, statistics()))
            for stat in records:
                stat['executed'] += 1
                stat['improvements'] += int(improved)
                stat['last'] = self.executed
                stat['seconds'] += elapsed
                stat['gain'] += gain
                stat['information'] += information
                stat['recent'] = (stat['recent']+[gain])[-4:]
        candidate = dict(changes=changes, rank=rank, outcome=outcome, blockers=tuple(blockers))
        pool = dict(self.beam, **{name: candidate})
        ordered = sorted(pool, key=lambda n: pool[n]['rank'], reverse=True)
        self.incumbent = ordered[0]
        if improved and self.incumbent != 'baseline':
            self.focus_due = True
        selected, signatures = [], set()
        # Prefer distinct partitions; reserve two places for distinct parameter
        # routes on a plateau (e.g. face vs body), without retaining every trial.
        for key, limit in ((lambda c: c['outcome'], max(1, self.width-2)),
                           (lambda c: (c['outcome'], c['blockers'], parameter_signature(c['changes'],self.config)), self.width)):
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
            score = ((stat['gain']+.05*stat['information'])/(stat['executed']+1) +
                     math.sqrt(2*math.log(self.executed+2)/(stat['executed']+1)))
            cost=max(1.,stat['seconds']/max(1,stat['executed']))
            return stat['executed'] == 0 and stat['rejected']<8, overdue, score/cost/(1+stat['rejected']*.02)
        return sorted(self.families, key=priority, reverse=True)

    def rejected(self, family, operator):
        self.stats[family]['rejected'] += 1
        op=self.operators.setdefault(operator,statistics())
        op['rejected']+=1

    def operations(self, streams):
        pending={k:iter(v) for k,v in streams.items()}
        while pending:
            def priority(k):
                stat=self.operators.get(k,{})
                count=stat.get('executed',0)
                reward = stat.get('gain',0)+.05*stat.get('information',0)
                return (reward+1)/(count+1)**.5 / max(1.,stat.get('seconds',0)/max(1,count)) / (1+stat.get('rejected',0)*.1)
            key=max(pending,key=priority)
            try:yield next(pending[key])
            except StopIteration:del pending[key]

    def _restart(self):
        if self.restarts>=1:return False
        self.restarts+=1;self.restart_executed=0;self.refining=False
        def tagged(n,c):
            for item in self.factory(n,c['changes'],None,'restart'):yield (n,*item)
        self.restart_pending=interleave(tagged(n,c) for n,c in self.beam.items())
        return True

    def next(self):
        if self.restart_pending is not None:
            if self.restart_executed<8:
                try:
                    parent,family,strategy,changes=next(self.restart_pending)
                    return family,'restart',parent,strategy,changes
                except StopIteration:pass
            self.restart_pending=None
            self.refining=True;self.refined=0
            self.refinements={n:iter(self.factory(n,c['changes'],None,True)) for n,c in self.beam.items()}
        if self.refining and self.refined >= self.refinement_budget:
            if self._restart():return self.next()
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
            if self._restart():return self.next()
            self.stop_reason = 'refinement_exhausted'
            raise StopIteration
        # Count only this lane's completed trials: interleaved even/odd lanes
        # otherwise alias with the modulo schedule and can starve the winner.
        baseline_first = (self.normal_completed - self.last_root >= 3 or
            not self.focus_due and self.explored < BASELINE_WARMUP and self.incumbent == 'baseline')
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
                    winner = [self.incumbent] if self.incumbent in parents else []
                    others = [n for n in parents if n != self.incumbent]
                    if others:
                        offset = self.parent_turn % len(others)
                        others = others[offset:] + others[:offset]
                    parents = (winner + others if self.focus_due or self.adaptive_completed % 2 == 0
                               else others + winner)
                    for n in parents:
                        key = n, family
                        if key not in self.streams:
                            self.streams[key] = iter(self.factory(n, self.beam[n]['changes'], family, False))
                        try:
                            strategy, changes = next(self.streams[key])
                            self.last_parent = n
                            return family, 'adaptive', n, strategy, changes
                        except StopIteration:
                            continue
        self.stop_reason = 'frontier_exhausted'
        raise StopIteration

    def completed(self, phase, *, parent=None):
        if phase in ('baseline_exploration', 'adaptive'):
            self.normal_completed += 1
        if phase == 'adaptive':
            self.adaptive_completed += 1
            self.focus_due = False
            if (parent if parent is not None else self.last_parent) != self.incumbent:
                self.parent_turn += 1
        if phase == 'baseline_exploration':
            self.explored += 1
            self.last_root = self.normal_completed
        if phase == 'refinement':
            self.refined += 1
        if phase == 'restart':
            self.restart_executed += 1


def restart_probes(config, changes, options):
    """Deterministic coverage of distant measured values; global replay cap wins."""
    choices=[]
    for key,values in sorted(options.items()):
        values=sorted(set(values))
        if not values:continue
        current=changes.get(key,parameter(config.data,key))
        selected=[values[0],values[len(values)//2],values[-1]]
        if key.startswith(PREFIX) and not key.endswith('min_support'):selected.extend((.1,.5,.9))
        for value in dict.fromkeys(selected):
            if value!=current:choices.append((key,value))
    for index,(key,value) in enumerate(choices):
        patch=dict(changes,**{key:value})
        if index+len(choices)//2<len(choices):
            other,v=choices[index+len(choices)//2]
            if other!=key:patch[other]=v
        yield patch
