"""Controlled telemetry allocations, IPC serialization and complete offline merge."""
import argparse
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
import json
import pickle
from pathlib import Path
import statistics
import tempfile
import time
from types import SimpleNamespace

from benchmarks.fixtures import record
from stash_ai.runtime.profile import ProfileMetrics, empty_sample, combine
from stash_ai.runtime.worker_result import WorkerResult, pack_chunk


def timed(callback):
    start = time.perf_counter()
    value = callback()
    return time.perf_counter()-start, value


def metrics(iterations):
    source = ProfileMetrics()
    source.record('stage', 'compute', .001, .0005, units=1, unit='scenes')
    row = source.export()[0]
    class LegacyMetrics(ProfileMetrics):
        def ingest(self, rows):
            with self.lock:
                for item in rows:
                    key = tuple(item[name] for name in ('kind', 'name', 'model', 'work', 'outcome', 'unit'))
                    combine(self.rows.setdefault(key, empty_sample()), item)
    def reference():
        result = LegacyMetrics()
        for _ in range(iterations): result.ingest([row])
        return result.export()[0]
    def optimized():
        result = ProfileMetrics()
        for _ in range(iterations): result.ingest([row])
        return result.export()[0]
    old, a = timed(reference); new, b = timed(optimized)
    assert a == b
    return dict(updates=iterations, reference_seconds=old, optimized_seconds=new,
                reference_initial_samples=iterations, optimized_initial_samples=1)


def progress(iterations):
    # Same unsaved counter contents, exercising the production throttle path.
    from stash_ai.history.lifecycle import RunHistory
    from stash_ai.runtime.report import RunReport
    from threading import get_ident
    report = RunReport(write=lambda _: None)
    report.counts.update({f'counter_{i}': i for i in range(100)})
    report.count_sources = [Counter({f'watched_{i}': i for i in range(100)})]
    history = object.__new__(RunHistory)
    history.store = SimpleNamespace(progress=lambda *a, **kw: None)
    history.args, history.owner_thread = SimpleNamespace(report=report), get_ident()
    history.start = history.last_progress = time.perf_counter()
    history.cpu_start, history.last_units = time.process_time(), 0
    history.progress_seconds = float('inf')
    def reference():
        for _ in range(iterations):
            if get_ident() != history.owner_thread: continue
            moment = time.perf_counter()
            counts = Counter(report.counts)
            for source in report.count_sources: counts.update(source)
            units = sum(counts[name] for name in ('processed', 'reused', 'threshold_cache_reused', 'scenes_synced'))
            assert moment-history.last_progress < history.progress_seconds and units-history.last_units < history.progress_units
    old, _ = timed(reference)
    new, _ = timed(lambda: [history.progress() for _ in range(iterations)])
    return dict(checks=iterations, reference_seconds=old, optimized_seconds=new,
                reference_counter_copies=iterations, optimized_counter_copies=0)


def _echo(connection):
    connection.send_bytes(b'ready')
    try:
        while payload := connection.recv_bytes():
            connection.send_bytes(payload)
    finally:
        connection.close()


def ipc(iterations):
    metrics = ProfileMetrics()
    metrics.record('stage', 'merge_worker_compute', .001, .0005, units=1, unit='scenes')
    metrics.record('worker_cpu', 'merge_worker_compute', .0007, .0007, units=1, unit='scenes')
    rows = [WorkerResult(str(i), {'hits': ['tag'], 'spans': []}, metrics=metrics.export()) for i in range(8)]
    chunk = pack_chunk(rows)
    def transfer(value):
        for _ in range(iterations): pickle.loads(pickle.dumps(value, protocol=5))
    old, _ = timed(lambda: transfer(rows)); new, _ = timed(lambda: transfer(chunk))
    from multiprocessing import get_context
    context = get_context('spawn')
    parent, child = context.Pipe()
    worker = context.Process(target=_echo, args=(child,))
    worker.start(); child.close()
    assert parent.recv_bytes() == b'ready'
    pipe_times = [[], []]
    try:
        for repeat in range(3):
            for index in ([0, 1] if repeat % 2 == 0 else [1, 0]):
                payload = pickle.dumps((rows, chunk)[index], protocol=5)
                def roundtrip():
                    for _ in range(iterations):
                        parent.send_bytes(payload)
                        assert parent.recv_bytes() == payload
                seconds, _ = timed(roundtrip)
                pipe_times[index].append(seconds)
    finally:
        parent.send_bytes(b'')
        parent.close()
        worker.join(5)
        if worker.is_alive():
            worker.kill(); worker.join()
    assert worker.exitcode == 0
    return dict(scenes=8, transfers=iterations, reference_bytes=len(pickle.dumps(rows, protocol=5)),
                optimized_bytes=len(pickle.dumps(chunk, protocol=5)), reference_roundtrip_seconds=old,
                optimized_roundtrip_seconds=new, reference_pipe_seconds=statistics.median(pipe_times[0]),
                optimized_pipe_seconds=statistics.median(pipe_times[1]), pipe_note='Echo of pre-serialized eight-scene payloads; process startup excluded')


def merge_trial(directory, scene_count, use_chunks):
    from stash_ai.runtime.context import RuntimeContext
    from stash_ai.runtime.report import RunReport
    from stash_ai.storage.store import StateStore
    from stash_ai.publication.publisher import MergePublisher
    from stash_ai.publication import parallel
    from stash_ai.publication.fingerprints import scene_fingerprint
    from tests.helpers.stash import Stash
    from stash_ai.evidence.labels import AI
    from stash_ai.models import SOURCES
    app = RuntimeContext()
    policies = {source: frozenset() for source in SOURCES}
    class Scenes(Stash):
        def __init__(self):
            super().__init__()
            self.scenes = {str(i): dict(id=str(i), tags=[], scene_markers=[], paths={}) for i in range(scene_count)}
        def find_scene(self, identifier):
            self.scene = self.scenes[str(identifier)]
            return deepcopy(self.scene)
        def update_scene_tags(self, identifier, ids):
            self.scene = self.scenes[str(identifier)]
            super().update_scene_tags(identifier, ids)
        def create_marker(self, payload):
            self.scene = self.scenes[str(payload['scene_id'])]
            return super().create_marker(payload)
    stash = Scenes()
    current = {i: dict(**stash.find_scene(i), fingerprint=scene_fingerprint(stash.find_scene(i))) for i in stash.scenes}
    path = Path(directory)/'state.db'
    report = RunReport(write=lambda _: None); report.profile = ProfileMetrics()
    args = SimpleNamespace(state_db=path, merge_workers=2, profile_merge=None, force_merge=True,
                           ram_cache_gb=0, ram_budget_gb=.5)
    frames = [dict(start=float(i), end=float(i+1), tags=[['tag', .9]]) for i in range(8)]
    original = parallel.chunk_size
    if not use_chunks: parallel.chunk_size = lambda total, workers: 1
    try:
        with StateStore(path, 'benchmark') as store:
            for scene in stash.scenes:
                store.put(scene, 'wd', {kind: record('wd', frames, scene=kind=='scene') for kind in ('scene', 'markers')})
            stats = Counter()
            publisher = MergePublisher(app, args, store, stash,
                app.services.TagResolver(stash, source=AI), policies, policies,
                publication_key='benchmark', stats=stats, report=report)
            elapsed, _ = timed(lambda: publisher.run(current))
            assert not stats['errors'] and stats['scenes_synced'] == scene_count
            snapshots = [(scene, kind, store.unpack_snapshot(blob)) for scene, kind, blob in store.connection.execute(
                'SELECT scene,kind,published FROM snapshots ORDER BY scene,kind')]
            assert all(len(scene['scene_markers']) == 1 and scene['tags'] for scene in stash.scenes.values())
            worker_cpu = sum(row['inclusive_seconds'] for row in report.profile.export() if row['kind']=='worker_cpu')
            return dict(seconds=elapsed, worker_cpu_seconds=worker_cpu, counts=dict(stats)), (stash.scenes, snapshots)
    finally:
        parallel.chunk_size = original


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--iterations', type=int, default=10000)
    p.add_argument('--scenes', type=int, default=64)
    p.add_argument('--repeats', type=int, default=3)
    args = p.parse_args()
    if args.iterations < 1 or args.scenes < 2 or args.repeats < 1: p.error('invalid benchmark size')
    trials = {False: [], True: []}
    output = None
    for repeat in range(args.repeats):
        for enabled in ([False, True] if repeat % 2 == 0 else [True, False]):
            with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                sample, observed = merge_trial(directory, args.scenes, enabled)
            if output is None: output = observed
            assert output == observed
            trials[enabled].append(sample)
    result = dict(metrics=metrics(args.iterations), history=progress(args.iterations), ipc=ipc(args.iterations),
                  complete_offline_merge=dict(scenes=args.scenes, repeats=args.repeats,
                    reference_trials=trials[False], optimized_trials=trials[True],
                    reference_median_seconds=statistics.median(row['seconds'] for row in trials[False]),
                    optimized_median_seconds=statistics.median(row['seconds'] for row in trials[True])))
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
