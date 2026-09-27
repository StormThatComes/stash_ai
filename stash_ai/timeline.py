"""Indexed positive observations on a non-overlapping VTT timeline.

Range queries select posting offsets and clip precomputed runs, instead of
visiting every frame/source again. Overlapping legacy timelines use the caller's
reference scanner, since removing a long leading interval can split a union.
"""
from bisect import bisect_left, bisect_right


class Series:
    def __init__(self):
        self.positions, self.starts, self.ends, self.scores = [], [], [], []
        self.coverage, self.runs = [], []

    def append(self, position, start, end, score=None):
        offset = len(self.positions)
        if not offset or start > self.ends[-1]:
            self.coverage.append(offset)
        if not offset or position != self.positions[-1] + 1 or start > self.ends[-1] + .001:
            self.runs.append(offset)
        self.positions.append(position)
        self.starts.append(start)
        self.ends.append(end)
        self.scores.append(score)

    def bounds(self, first, last):
        lo = bisect_left(self.positions, first)
        return lo, bisect_left(self.positions, last, lo)

    def slices(self, boundaries, lo, hi):
        group = bisect_right(boundaries, lo) - 1
        while True:
            next_group = group + 1
            stop = boundaries[next_group] if next_group < len(boundaries) else hi
            yield lo, min(stop, hi)
            if stop >= hi:
                return
            lo, group = stop, next_group

    def intervals(self, lo, hi, end):
        return [[self.starts[a], min(self.ends[b-1], end)]
            for a, b in self.slices(self.coverage, lo, hi)]

    def evidence(self, lo, hi, end, limit, short, group):
        value = dict(intervals=self.intervals(lo, hi, end),
            sample_times=self.starts[lo:min(hi, lo+limit)], hits=hi-lo)
        if short:
            value.update(run_group=group, runs=[
                [self.positions[a], self.positions[b-1]+1, self.starts[a], min(self.ends[b-1], end)]
                for a, b in self.slices(self.runs, lo, hi)])
        return value


class Timeline:
    def __init__(self, events, context, allowed):
        self.tags, self.context = {}, context
        for position, event in enumerate(events):
            start, end = event['start'], event['end']
            for tag, values in event['extras'].items():
                sources = values if allowed is None else {s: score for s, score in values.items() if s in allowed}
                if not sources:
                    continue
                if tag not in self.tags:
                    self.tags[tag] = Series(), {}
                total, own = self.tags[tag]
                total.append(position, start, end)
                for source, score in sources.items():
                    if source is not None:
                        if source not in own:
                            own[source] = Series()
                        own[source].append(position, start, end, score)

    def support(self, first, last, end, policy, group):
        support = {}
        for tag, (total, sources) in self.tags.items():
            lo, hi = total.bounds(first, last)
            if lo == hi:
                continue
            item = total.evidence(lo, hi, end, policy.extra_min_hits, policy.short_marker_extras, group)
            own = {}
            for source, series in sources.items():
                a, b = series.bounds(first, last)
                if a < b:
                    own[source] = dict(intervals=series.intervals(a, b, end),
                        sample_times=series.starts[a:min(b, a+policy.extra_min_hits)],
                        score=max(series.scores[a:b]), signatures=self.context[source])
            if own:
                item['sources'] = own
            support[tag] = item
        return support


def support_size(support):
    """Conservative charge for this fixed schema, without recursively walking it.

    Counts shared objects again for each owner. Padding covers dict/list capacity
    and primitive objects on supported CPython builds; strings use real sizes.
    It is a cache eviction estimate, never a storage serialization or RSS limit.
    """
    import sys
    size = sys.getsizeof(support) + 512
    for tag, item in support.items():
        size += sys.getsizeof(tag) + sys.getsizeof(item) + 1024
        size += 192 * len(item['intervals']) + 64 * len(item['sample_times'])
        size += 256 * len(item.get('runs', ())) + sys.getsizeof(item.get('run_group'))
        size += sys.getsizeof(item.get('sources', {}))
        for source, own in item.get('sources', {}).items():
            size += sys.getsizeof(source) + sys.getsizeof(own) + 1024
            size += 192 * len(own['intervals']) + 64 * len(own['sample_times'])
            size += sum(sys.getsizeof(signature) + 32 for signature in own['signatures'])
    return size
