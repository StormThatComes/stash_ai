"""Thread-safe LRU bounded by retained bytes and measured RSS headroom."""
from collections import Counter, OrderedDict
from contextlib import contextmanager
import time
import threading
from . import memory


class RamCache:
    """LRU charged in estimated retained bytes, not scene count or compressed size.

    Borrowed values must not be mutated. Accounting is conservative across
    entries that share objects; the budget does not include active calculations.
    """
    def __init__(self, limit=0, *, rss_budget=0):
        self.limit = max(0, int(limit))
        self.entries = OrderedDict()
        self.size = self.peak = 0
        self.stats = Counter()
        self.rss_budget = max(0, int(rss_budget))
        self.rss_peak = 0
        self._next_pressure_check = 0.0
        self._effective_limit = self.limit
        self._lock = threading.RLock()

    def usage(self):
        """Current retained bytes and configured cap; never changes cache state."""
        with self._lock:
            return self.size, self.limit

    def get(self, key):
        with self._lock:
            if key not in self.entries:
                self.stats['misses'] += 1
                return None
            self.entries.move_to_end(key)
            self.stats['hits'] += 1
            return self.entries[key][0]

    def put(self, key, value, *, retained_bytes=None):
        if not self.limit:
            return
        cost = (memory.retained_size((key, value)) if retained_bytes is None
                else memory.retained_size(key) + max(0, int(retained_bytes)) + 128) + 128
        with self._lock:
            self.discard(key)
            self.pressure()
            if cost > self._effective_limit:
                self.stats['oversized'] += 1
                return
            self.trim(max(0, self._effective_limit - cost))
            self.entries[key] = value, cost
            self.size += cost
            self.peak = max(self.peak, self.size)

    def trim(self, target):
        with self._lock:
            while self.entries and self.size > target:
                _, (_, size) = self.entries.popitem(last=False)
                self.size -= size
                self.stats['evictions'] += 1

    def pressure(self, force=False):
        """Keep working/model headroom. This is not an OS hard memory limit."""
        if not self.rss_budget:
            return
        with self._lock:
            now = time.monotonic()
            if not force and now < self._next_pressure_check:
                return
            self._next_pressure_check = now + .25
            rss = memory.process_rss()
            if rss is None:
                return
            self.rss_peak = max(self.rss_peak, rss)
            threshold = self.rss_budget * .875
            self._effective_limit = min(self.limit, max(0, int(threshold - max(0, rss - self.size))))
            if rss > threshold:
                excess = int(rss - threshold)
                self.trim(min(self._effective_limit, max(0, self.size - max(excess, self.size // 4))))
                self.stats['pressure_trims'] += 1

    def discard(self, key):
        with self._lock:
            entry = self.entries.pop(key, None)
            if entry is not None:
                self.size -= entry[1]

    def discard_matching(self, predicate):
        """Release a token scope without retaining an unbounded list of keys."""
        with self._lock:
            for key in tuple(self.entries):
                if predicate(key):
                    self.discard(key)

    def clear(self):
        with self._lock:
            self.entries.clear()
            self.size = 0


    @contextmanager
    def temporary_budget(self, *, limit, rss_budget):
        """Reserve this process's share while parallel readers own other shares."""
        with self._lock:
            previous = self.limit, self.rss_budget, self._effective_limit
            self.limit = max(0, int(limit))
            self.rss_budget = max(0, int(rss_budget))
            self._effective_limit = self.limit
            self.trim(self.limit)
            self.pressure(force=True)
        try:
            yield
        finally:
            with self._lock:
                self.limit, self.rss_budget, self._effective_limit = previous
                self._next_pressure_check = 0.0
                self.trim(self._effective_limit)
                self.pressure(force=True)
