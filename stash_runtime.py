"""Byte-bounded reusable results and collection-sized progress reporting."""
from collections import Counter, OrderedDict
from contextlib import contextmanager, nullcontext
from functools import wraps
import sys
import time
import threading


def stage(report, name):
    return report.measure(name) if report is not None else nullcontext()


def measured(stage):
    """Report a method's exclusive time when a run report is attached."""
    def decorate(method):
        @wraps(method)
        def wrapped(self, *args, **kwargs):
            report = getattr(self, 'report', None)
            if report is None:
                return method(self, *args, **kwargs)
            with report.measure(stage):
                return method(self, *args, **kwargs)
        return wrapped
    return decorate


def process_rss():
    """Current resident memory on Windows/Linux; no peak-RSS approximation."""
    try:
        if sys.platform == 'win32':
            import ctypes
            from ctypes import wintypes
            class Counters(ctypes.Structure):
                _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
                    (name, ctypes.c_size_t) for name in ('peak', 'working', 'paged_peak',
                    'paged', 'nonpaged_peak', 'nonpaged', 'pagefile', 'pagefile_peak')]
            result = Counters()
            result.cb = ctypes.sizeof(result)
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            psapi = ctypes.WinDLL('psapi', use_last_error=True)
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(result), result.cb):
                return result.working
        elif sys.platform.startswith('linux'):
            import os
            with open('/proc/self/statm') as stream:
                return int(stream.read().split()[1]) * os.sysconf('SC_PAGE_SIZE')
    except (OSError, ValueError, IndexError, AttributeError):
        pass
    return None


def retained_size(value):
    """Count retained Python objects once, without serializing a second copy."""
    pending, seen, total = [value], set(), 0
    while pending:
        item = pending.pop()
        identity = id(item)
        if identity in seen:
            continue
        seen.add(identity)
        total += sys.getsizeof(item)
        if isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, (list, tuple, set, frozenset)):
            pending.extend(item)
        if hasattr(item, '__dict__'):
            pending.append(vars(item))
    return total


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
        cost = (retained_size((key, value)) if retained_bytes is None
                else retained_size(key) + max(0, int(retained_bytes)) + 128) + 128
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
            rss = process_rss()
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

    def clear(self):
        with self._lock:
            self.entries.clear()
            self.size = 0


class RunReport:
    """Fixed-size counters and bounded error examples, even for many scenes."""
    def __init__(self, verbose=False, write=print, max_groups=6, examples=2):
        self.verbose, self.write = verbose, write
        self.max_groups, self.examples = max_groups, examples
        self.groups = {}
        self.overflow = 0
        self.counts = Counter()
        self.timings, self.timing_calls = Counter(), Counter()
        self._timers = threading.local()
        self._timing_lock = threading.RLock()
        self.diagnostics = None

    @contextmanager
    def measure(self, stage):
        stack = getattr(self._timers, 'stack', None)
        if stack is None:
            stack = self._timers.stack = []
        entry = [time.perf_counter(), 0.0]
        stack.append(entry)
        failed = True
        try:
            yield
            failed = False
        finally:
            elapsed = time.perf_counter() - entry[0]
            stack.pop()
            if stack:
                stack[-1][1] += elapsed
            with self._timing_lock:
                self.timings[stage] += max(0.0, elapsed-entry[1])
                self.timing_calls[stage] += 1
            if self.diagnostics is not None:
                self.diagnostics.record_stage(stage, elapsed, max(0.0, elapsed-entry[1]), failed)

    @contextmanager
    def request(self, operation, read):
        start, failed = time.perf_counter(), True
        details = {'response_bytes': 0}
        previous = getattr(self._timers, 'request', None)
        self._timers.request = details
        try:
            with self.measure('stash_read' if read else 'stash_write'):
                yield
            failed = False
        finally:
            self._timers.request = previous
            if self.diagnostics is not None:
                self.diagnostics.record_api(operation, time.perf_counter()-start, failed, details['response_bytes'])

    def response_size(self, size):
        current = getattr(self._timers, 'request', None)
        if current is not None:
            current['response_bytes'] += size

    def record_counts(self, **values):
        if self.diagnostics is not None:
            self.diagnostics.record_counts(values)

    def detail(self, message):
        if self.verbose:
            self.write(message)

    def issue(self, stage, source, scene, exc, message):
        key = (stage, source, type(exc).__name__)
        if key not in self.groups:
            if len(self.groups) >= self.max_groups:
                self.overflow += 1
                if self.verbose:
                    self.write(f'ERROR {stage} {source} scene {scene}: {message}')
                return
            self.groups[key] = dict(count=0, examples=[])
        group = self.groups[key]
        group['count'] += 1
        example = f"scene {scene}: {message}" if scene is not None else message
        example = example.replace('\n', ' ')[:350]
        if len(group['examples']) < self.examples:
            group['examples'].append(example)
        if self.verbose or group['count'] == 1:
            self.write(f"ERROR {stage} {source}: {example}")

    def progress(self, bar, counts):
        bar.set_postfix(processed=counts['processed'], reused=counts['reused'],
                        skipped=counts['skipped'], errors=counts['errors'], refresh=False)

    def finish(self, title, counts, cache=None):
        groups = (
            ('Scenes/models', ('processed', 'reused', 'skipped', 'errors', 'scenes_synced', 'merge_reused', 'bootstrap_reused', 'checkpoint_resumes')),
            ('Scene tags', ('merged_scene_tags', 'scene_tags_updated', 'shared_scene_tags', 'scene_cache_partial', 'scene_cache_missing')),
            ('Markers', ('merged_markers', 'markers_created', 'markers_updated', 'markers_deleted', 'markers_unchanged')),
            ('Occurrence points', ('occurrence_markers', 'occurrence_missing_timestamps', 'occurrence_limited',
                                   'occurrence_manual_locations')),
            ('Cleanup', ('cleanup_fast_path', 'cleanup_resumed', 'cleanup_candidate_scenes', 'blacklisted_tags_deleted',
                         'cleanup_markers_changed_or_gone', 'cleanup_tags_changed_or_gone')),
            ('Shared evidence', ('shared_marker_runs', 'shared_marker_extensions', 'short_extras', 'cross_model_extras')),
        )
        combined = Counter(counts)
        combined.update(self.counts)
        self.write(title)
        printed = set()
        for label, keys in groups:
            present = [key for key in keys if key in combined]
            if present:
                self.write('  ' + label + ': ' + ', '.join(f'{key}={combined[key]}' for key in present))
                printed.update(present)
        rest = sorted(set(combined) - printed)
        if rest:
            self.write('  Other: ' + ', '.join(f'{key}={combined[key]}' for key in rest))
        if cache is not None:
            self.write(f"  RAM cache: peak={cache.peak / 2**30:.2f} GiB, hits={cache.stats['hits']}, "
                       f"misses={cache.stats['misses']}, evictions={cache.stats['evictions']}, "
                       f"RSS sampled peak={cache.rss_peak / 2**30:.2f} GiB, pressure trims={cache.stats['pressure_trims']}")
        if self.timings:
            self.write('  Timings (exclusive): ' + ', '.join(
                f'{key}={value:.3f}s/{self.timing_calls[key]} calls' for key, value in sorted(self.timings.items())))
        for (stage, source, kind), group in self.groups.items():
            self.write(f"  Errors {stage}/{source}/{kind}: {group['count']}; " + ' | '.join(group['examples']))
        if self.overflow:
            self.write(f'  Other errors: {self.overflow}; use --verbose for full diagnostics.')
