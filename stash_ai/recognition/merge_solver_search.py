"""Budgeted exploration independent of frozen catalogue nominations."""
from collections import deque
from itertools import islice

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


class Frontier:
    """Reserve baseline exploration even while adaptive descendants improve."""
    def __init__(self, factory):
        self.root = deque([('baseline', iter(factory({})))])
        self.adaptive = deque()
        self.explored = 0
        self.adapt_next = False

    def next(self):
        lanes = [('baseline_exploration', self.root), ('adaptive', self.adaptive)]
        if self.explored >= BASELINE_WARMUP and self.adapt_next:
            lanes.reverse()
        for phase, queue in lanes:
            while queue:
                parent, iterator = queue.popleft()
                try:
                    changes = next(iterator)
                except StopIteration:
                    continue
                queue.append((parent, iterator))
                return phase, parent, changes
        raise StopIteration

    def completed(self, phase):
        if phase == 'baseline_exploration':
            self.explored += 1
        self.adapt_next = phase != 'adaptive'

    def add(self, parent, stream, *, preferred):
        item = (parent, iter(stream))
        if preferred:
            self.adaptive.appendleft(item)
        else:
            self.adaptive.append(item)
