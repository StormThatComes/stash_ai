"""Per-scene indexes of immutable native markers."""
from bisect import bisect_left, bisect_right
from collections import defaultdict
from math import inf, isfinite, nextafter

import stash_ai.evidence.interval_geometry as merge_intervals


def _start_bounds(starts, start, tolerance):
    # abs(old - start) rounds before comparison. Widen the search conservatively,
    # especially when start - tolerance cancels to zero; callers check it exactly.
    if tolerance == 0:
        return bisect_left(starts, start), bisect_right(starts, start)
    radius = nextafter(tolerance, inf)
    return (bisect_left(starts, nextafter(start - radius, -inf)),
            bisect_right(starts, nextafter(start + radius, inf)))


class _IntervalBucket:
    def __init__(self, rows):
        # Each row carries its original order and native marker.
        self.rows = sorted(rows, key=lambda row: row[1]['seconds'])
        self.starts = [row[1]['seconds'] for row in self.rows]
        ends = [row[1].get('end_seconds') if row[1].get('end_seconds') is not None
                else row[1]['seconds'] for row in self.rows]
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
                if merge_intervals.overlaps(self.rows[offset][1], interval, tolerance)]
        return [row[1] for row in sorted(rows, key=lambda row: row[0])]


class MarkerIntervalIndex:
    """Query owned markers by primary ID (planning) or name (cleanup).

    Max-end subtrees prune even nested/overlapping ranges. Consumed matches remain
    in this immutable index: their manual extras must survive subsequent splits.
    """
    def __init__(self, markers, *, by_name=False):
        grouped = defaultdict(list)
        for order, marker in enumerate(markers):
            primary = marker['primary_tag']['name'] if by_name else str(marker['primary_tag']['id'])
            grouped[primary].append((order, marker))
        self.buckets = {primary: _IntervalBucket(rows) for primary, rows in grouped.items()}

    def overlapping(self, primary, interval, tolerance=0):
        bucket = self.buckets.get(primary)
        return bucket.overlapping(interval, tolerance) if bucket is not None else []

