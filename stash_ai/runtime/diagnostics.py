"""Bounded merge diagnostics; JSONL contains timings and counts, never payloads."""
from collections import Counter
from contextlib import contextmanager
import heapq
import json
from pathlib import Path
import threading
import time
from .status import completion_status


def add_sample(target, name, elapsed, exclusive, failed=False, response_bytes=0):
    row = target.setdefault(name, dict(calls=0, seconds=0.0, inclusive_seconds=0.0,
                                       max_seconds=0.0, errors=0))
    row['calls'] += 1
    row['seconds'] += exclusive
    row['inclusive_seconds'] += elapsed
    row['max_seconds'] = max(row['max_seconds'], elapsed)
    row['errors'] += int(failed)
    if response_bytes:
        row['response_bytes'] = row.get('response_bytes', 0) + response_bytes


class MergeDiagnostics:
    """Keep only one active scene and three slow examples; stream everything else."""
    def __init__(self, report, path=None, *, dry_run=False, merge_only=False, force=False):
        self.report, self.path = report, Path(path) if path else None
        self.metadata = dict(dry_run=dry_run, merge_only=merge_only, force_merge=force)
        self.stream = None
        self.stages, self.api, self.counts = {}, {}, Counter()
        self.scenes, self.scene_wall, self.scene_cpu = 0, 0.0, 0.0
        self.slowest, self.statuses = [], Counter()
        self.local = threading.local()
        self.lock = threading.RLock()
        self.active_scenes = set()

    def bind(self, operation):
        """Carry the captured scene to HTTP workers without leaking thread state."""
        captured = getattr(self.local, 'scene', None)
        def bound():
            previous = getattr(self.local, 'scene', None)
            self.local.scene = captured
            try:
                return operation()
            finally:
                self.local.scene = previous
        return bound

    def __enter__(self):
        if self.path:
            if self.path.suffix.lower() != '.jsonl':
                raise ValueError('--profile-merge requires a new .jsonl file.')
            self.stream = self.path.open('x', encoding='utf-8')
        self.start, self.cpu_start = time.perf_counter(), time.process_time()
        self.report.diagnostics = self
        self.emit(dict(type='start', format=1, **self.metadata))
        return self

    def emit(self, record):
        if self.stream:
            try:
                self.stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
                self.stream.flush()
            except OSError as exc:
                # Diagnostic IO must not turn successful Stash writes into a
                # failed transaction or obscure an actual API failure.
                self.report.write('Merge profile write failed: ' + type(exc).__name__)
                self.close_stream()

    def close_stream(self):
        stream, self.stream = self.stream, None
        if stream is not None:
            try:
                stream.close()
            except OSError:
                self.report.write('Merge profile close failed: OSError')

    def record_stage(self, name, elapsed, exclusive, failed):
        with self.lock:
            add_sample(self.stages, name, elapsed, exclusive, failed)
            scene = getattr(self.local, 'scene', None)
            if scene is not None and id(scene) in self.active_scenes:
                add_sample(scene['stages'], name, elapsed, exclusive, failed)

    def record_api(self, name, elapsed, failed, response_bytes):
        with self.lock:
            add_sample(self.api, name, elapsed, elapsed, failed, response_bytes)
            scene = getattr(self.local, 'scene', None)
            if scene is not None and id(scene) in self.active_scenes:
                add_sample(scene['api'], name, elapsed, elapsed, failed, response_bytes)

    def record_counts(self, values):
        with self.lock:
            self.counts.update(values)
            scene = getattr(self.local, 'scene', None)
            if scene is not None and id(scene) in self.active_scenes:
                scene['counts'].update(values)

    @contextmanager
    def scene(self, identifier):
        record = dict(type='scene', scene_id=str(identifier), status='ok', reason='unknown',
                      stages={}, api={}, counts=Counter(), statistics={})
        previous = getattr(self.local, 'scene', None)
        self.local.scene = record
        with self.lock:
            self.active_scenes.add(id(record))
        start, cpu = time.perf_counter(), time.process_time()
        try:
            yield record
        except BaseException as exc:
            record['status'] = completion_status(type(exc))
            raise
        finally:
            wall, used = time.perf_counter()-start, time.process_time()-cpu
            with self.lock:
                self.active_scenes.discard(id(record))
                record.update(wall_seconds=wall, cpu_seconds=used,
                              unmeasured_seconds=max(0.0, wall-sum(v['seconds'] for v in record['stages'].values())))
            self.local.scene = previous
            self.scenes += 1
            self.scene_wall += wall
            self.scene_cpu += used
            self.statuses[record['status']] += 1
            if record['status'] != 'reused':
                heapq.heappush(self.slowest, (wall, self.scenes, record))
                if len(self.slowest) > 3:
                    heapq.heappop(self.slowest)
            self.emit(record)
            if self.report.verbose:
                stages = ', '.join(f'{key}={value["seconds"]:.3f}s/{value["calls"]}'
                                   for key, value in sorted(record['stages'].items()) if value['seconds'] >= .0005)
                self.report.detail(f'Merge scene {identifier}: wall={wall:.3f}s cpu={used:.3f}s '
                                   f'reason={record["reason"]}; ' + stages)

    def __exit__(self, exc_type, exc, tb):
        self.report.diagnostics = None
        summary = dict(type='summary', status=completion_status(exc_type, errors=self.statuses['error']),
                       **self.metadata, scenes=self.scenes, scene_statuses=dict(self.statuses),
                       wall_seconds=time.perf_counter()-self.start, cpu_seconds=time.process_time()-self.cpu_start,
                       scene_wall_seconds=self.scene_wall, scene_cpu_seconds=self.scene_cpu,
                       stages=self.stages, api=self.api, counts=dict(self.counts))
        self.emit(summary)
        self.close_stream()
        if self.scenes:
            self.report.write(f'  Merge profile: scenes={self.scenes}, wall={self.scene_wall:.3f}s, '
                              f'cpu={self.scene_cpu:.3f}s, avg={self.scene_wall/self.scenes:.3f}s/scene')
        for wall, _, row in sorted(self.slowest, reverse=True):
            if not self.path and wall < 2:
                continue
            costly = sorted(row['stages'].items(), key=lambda item: -item[1]['seconds'])[:3]
            self.report.write(f'  Slow scene {row["scene_id"]}: wall={wall:.3f}s cpu={row["cpu_seconds"]:.3f}s; '
                              + ', '.join(f'{name}={value["seconds"]:.3f}s' for name, value in costly))
        if self.path:
            for name, values in sorted(self.api.items(), key=lambda item: -item[1]['seconds'])[:3]:
                self.report.write(f'  API {name}: calls={values["calls"]}, total={values["seconds"]:.3f}s, '
                    f'avg={values["seconds"]/values["calls"]:.3f}s, max={values["max_seconds"]:.3f}s, errors={values["errors"]}')
