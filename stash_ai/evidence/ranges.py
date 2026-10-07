"""Logarithmic queries over sorted, disjoint merged intervals."""
from bisect import bisect_right


class Coverage:
    def __init__(self, intervals):
        self.starts = [start for start, _ in intervals]
        self.ends = [end for _, end in intervals]

    def contains(self, start, end):
        position = bisect_right(self.starts, start) - 1
        return position >= 0 and self.ends[position] >= end

    def overlaps(self, start, end):
        position = bisect_right(self.ends, start)
        return position < len(self.starts) and self.starts[position] < end
