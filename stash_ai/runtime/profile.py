"""Thread-safe bounded latency aggregates, independent of merge JSONL tracing."""
from collections import Counter
import math
import threading


def empty_sample():
    return dict(calls=0, errors=0, seconds=0.0, inclusive_seconds=0.0,
                min_seconds=None, max_seconds=0.0, units=0, histogram=[0] * 64)


def combine(target, source):
    for name in ('calls', 'errors', 'seconds', 'inclusive_seconds', 'units'):
        target[name] += source[name]
    if source['min_seconds'] is not None:
        target['min_seconds'] = (source['min_seconds'] if target['min_seconds'] is None
                                 else min(target['min_seconds'], source['min_seconds']))
    target['max_seconds'] = max(target['max_seconds'], source['max_seconds'])
    target['histogram'] = [a + b for a, b in zip(target['histogram'], source['histogram'])]


def describe(sample):
    result = {key: value for key, value in sample.items() if key != 'histogram'}
    count = sample['calls']
    result['mean_seconds'] = sample['inclusive_seconds'] / count if count else 0.0
    result['mean_exclusive_seconds'] = sample['seconds'] / count if count else 0.0
    result['units_per_second'] = (sample['units'] / sample['inclusive_seconds']
                                  if sample['inclusive_seconds'] else 0.0)
    for name, fraction in (('p50_seconds', .5), ('p95_seconds', .95)):
        cumulative, value = 0, 0.0
        for bucket, frequency in enumerate(sample['histogram']):
            cumulative += frequency
            if count and cumulative >= math.ceil(count * fraction):
                value = min(sample['max_seconds'], 0.0 if bucket == 0 else 2.0 ** (bucket - 1) / 1e6)
                break
        result[name] = value
    return result


class ProfileMetrics:
    """64 log buckets per operation/model/outcome; no per-call sample list."""
    def __init__(self):
        self.rows, self.outcomes, self.models = {}, {}, {}
        self.lock = threading.RLock()

    def record(self, kind, name, elapsed, exclusive, failed=False, *, model='', work='', units=0, unit=''):
        elapsed, exclusive = max(0.0, elapsed), max(0.0, exclusive)
        bucket = 0 if elapsed == 0 else min(63, max(1, math.ceil(math.log2(max(1.0, elapsed * 1e6))) + 1))
        key = (kind, name, model or '', work, 'error' if failed else 'ok', unit)
        with self.lock:
            row = self.rows.get(key)
            if row is None:
                row = self.rows[key] = empty_sample()
            row['calls'] += 1
            row['errors'] += int(failed)
            row['seconds'] += exclusive
            row['inclusive_seconds'] += elapsed
            row['min_seconds'] = elapsed if row['min_seconds'] is None else min(row['min_seconds'], elapsed)
            row['max_seconds'] = max(row['max_seconds'], elapsed)
            row['histogram'][bucket] += 1
            if not failed:
                row['units'] += units

    def outcome(self, model, name):
        with self.lock:
            self.outcomes.setdefault(model, Counter())[name] += 1

    def ingest(self, rows):
        with self.lock:
            for row in rows:
                key = tuple(row[name] for name in ('kind', 'name', 'model', 'work', 'outcome', 'unit'))
                target = self.rows.get(key)
                if target is None:
                    target = self.rows[key] = empty_sample()
                combine(target, row)

    def export(self):
        with self.lock:
            return [dict(kind=key[0], name=key[1], model=key[2], work=key[3], outcome=key[4], unit=key[5],
                         **{name: list(value) if name == 'histogram' else value for name, value in row.items()})
                    for key, row in sorted(self.rows.items())]


def execute_worker(scene, store, callback, name, *, profile=False):
    """Capture exceptions after measurement; the parent alone writes history."""
    import time
    from contextlib import nullcontext
    from stash_ai.runtime.report import RunReport
    from stash_ai.runtime.worker_result import WorkerResult
    report = RunReport(write=lambda _: None) if profile else None
    previous = store.report if store is not None else None
    if profile:
        report.profile = ProfileMetrics()
        if store is not None:
            store.report = report
    cpu = time.process_time() if profile else None
    payload, error, failed = None, None, True
    try:
        with report.measure(name, units=1, unit='scenes') if profile else nullcontext():
            payload = callback()
        failed = False
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        if profile:
            cpu_elapsed = time.process_time() - cpu
            report.profile.record('worker_cpu', name, cpu_elapsed, cpu_elapsed, failed,
                                  units=1, unit='scenes')
            if store is not None:
                store.report = previous
    return WorkerResult(str(scene), payload, error, report.profile.export() if profile else None)
