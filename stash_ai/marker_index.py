"""Per-scene indexes of immutable native markers and saved point extents."""
from bisect import bisect_left, bisect_right
from collections import defaultdict
from math import inf, isfinite, nextafter

from stash_ai import merge


def _start_bounds(starts, start, tolerance):
    # abs(old - start) rounds before comparison. Widen the search conservatively,
    # especially when start - tolerance cancels to zero; callers check it exactly.
    if tolerance == 0:
        return bisect_left(starts, start), bisect_right(starts, start)
    radius = nextafter(tolerance, inf)
    return (bisect_left(starts, nextafter(start - radius, -inf)),
            bisect_right(starts, nextafter(start + radius, inf)))


class PreviousMarkerRanges:
    """Recover a point marker's extent from published, pending and recovery ranges."""
    def __init__(self, ranges):
        grouped = defaultdict(dict)
        for item in ranges:
            if item['end'] is not None:
                name, start = merge.AI.prefix + item['tag'], round(item['start'], 3)
                grouped[name][start] = max(grouped[name].get(start, -inf), item['end'])
        self.by_name = {name: (sorted(values), values) for name, values in grouped.items()}

    def temporal(self, marker):
        saved = self.by_name.get(marker['primary_tag']['name'])
        if saved is None:
            return marker
        starts, ends = saved
        lo, hi = _start_bounds(starts, marker['seconds'], .0005)
        matches = [ends[start] for start in starts[lo:hi] if abs(start - marker['seconds']) <= .0005]
        return {**marker, 'end_seconds': max(matches)} if matches else marker


class _IntervalBucket:
    def __init__(self, rows):
        # Each row carries its original order, native marker and temporal view.
        self.rows = sorted(rows, key=lambda row: row[2]['seconds'])
        self.starts = [row[2]['seconds'] for row in self.rows]
        ends = [row[2].get('end_seconds') if row[2].get('end_seconds') is not None
                else row[2]['seconds'] for row in self.rows]
        self.finite = all(isfinite(value) for value in (*self.starts, *ends))
        self.size = 1 << (len(rows) - 1).bit_length()
        self.max_ends = [-inf] * (self.size * 2)
        self.max_ends[self.size:self.size + len(ends)] = ends
        for node in range(self.size - 1, 0, -1):
            self.max_ends[node] = max(self.max_ends[2*node], self.max_ends[2*node + 1])

    def overlapping(self, interval, tolerance):
        start = interval['seconds']
        end = interval.get('end_seconds')
        end = start if end is None else end
        if not self.finite or not all(isfinite(value) for value in (start, end, tolerance)):
            candidates = range(len(self.rows))
        else:
            limit = bisect_right(self.starts, end)
            candidates = set()
            pending = [(1, 0, self.size)]
            while pending:
                node, first, stop = pending.pop()
                if first >= limit or self.max_ends[node] < start:
                    continue
                if stop - first == 1:
                    candidates.add(first)
                else:
                    mid = (first + stop) // 2
                    pending.extend(((2*node, first, mid), (2*node + 1, mid, stop)))
            # Start tolerance is separate from interval overlap, including points
            # and reversed legacy ranges. It does not extend interval ends.
            lo, hi = _start_bounds(self.starts, start, tolerance)
            candidates.update(range(lo, hi))
        rows = [self.rows[offset] for offset in candidates
                if merge.overlaps(self.rows[offset][2], interval, tolerance)]
        return [row[1] for row in sorted(rows, key=lambda row: row[0])]


class MarkerIntervalIndex:
    """Query owned markers by primary ID (planning) or name (cleanup).

    Max-end subtrees prune even nested/overlapping ranges. Consumed matches remain
    in this immutable index: their manual extras must survive subsequent splits.
    """
    def __init__(self, markers, previous_ranges, *, by_name=False):
        if not isinstance(previous_ranges, PreviousMarkerRanges):
            previous_ranges = PreviousMarkerRanges(previous_ranges)
        grouped = defaultdict(list)
        self.temporal_views = {}
        for order, marker in enumerate(markers):
            temporal = previous_ranges.temporal(marker)
            self.temporal_views[id(marker)] = temporal
            primary = marker['primary_tag']['name'] if by_name else str(marker['primary_tag']['id'])
            grouped[primary].append((order, marker, temporal))
        self.buckets = {primary: _IntervalBucket(rows) for primary, rows in grouped.items()}

    def temporal(self, marker):
        return self.temporal_views[id(marker)]

    def overlapping(self, primary, interval, tolerance=0):
        bucket = self.buckets.get(primary)
        return bucket.overlapping(interval, tolerance) if bucket is not None else []
