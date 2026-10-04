"""Bounded integer distributions used by anonymous diagnostics."""


class Distribution:
    """Bounded-memory integer distribution with deterministic approximate percentiles."""

    # Small cardinalities stay exact. Larger values use a fixed logarithmic
    # histogram, preventing a diagnostics report from retaining one dictionary
    # entry for every distinct BLOB size in a very large database.
    EXACT_MAX = 255
    LOG_BUCKETS = 32
    MAX_EXPONENT = 62  # SQLite INTEGER / practical Python container sizes are <= 2**63-1.
    MAX_BUCKETS = (EXACT_MAX + 1) + (MAX_EXPONENT - EXACT_MAX.bit_length() + 1) * LOG_BUCKETS

    def __init__(self):
        # key -> [count, minimum observed value, maximum observed value].
        # The key space is bounded by MAX_BUCKETS regardless of sample count.
        self.buckets = {}
        self.count = 0
        self.total = 0
        self.minimum = None
        self.maximum = None

    @classmethod
    def _bucket(cls, value):
        if value <= cls.EXACT_MAX:
            return (0, value)
        exponent = min(value.bit_length() - 1, cls.MAX_EXPONENT)
        base = 1 << exponent
        width = max(1, base // cls.LOG_BUCKETS)
        slot = min(cls.LOG_BUCKETS - 1, max(0, (value - base) // width))
        return (1, exponent, slot)

    def add(self, value):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return
        key = self._bucket(value)
        bucket = self.buckets.get(key)
        if bucket is None:
            self.buckets[key] = [1, value, value]
        else:
            bucket[0] += 1
            bucket[1] = min(bucket[1], value)
            bucket[2] = max(bucket[2], value)
        self.count += 1
        self.total += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)

    def _percentile(self, numerator):
        if not self.count:
            return 0
        # Nearest-rank over a deterministic bounded histogram. Exact buckets
        # return the original value; larger buckets return the midpoint of the
        # observed range within that bucket.
        rank = max(1, (self.count * numerator + 99) // 100)
        seen = 0
        for key in sorted(self.buckets):
            occurrences, low, high = self.buckets[key]
            seen += occurrences
            if seen >= rank:
                return low if low == high else (low + high) // 2
        return self.maximum or 0

    def summary(self, *, total=False):
        result = {
            "count": self.count,
            "min": self.minimum or 0,
            "p50": self._percentile(50),
            "p95": self._percentile(95),
            "p99": self._percentile(99),
            "max": self.maximum or 0,
            "mean": round(self.total / self.count, 2) if self.count else 0,
        }
        if total:
            result["total"] = self.total
        return result
