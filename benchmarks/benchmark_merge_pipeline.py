"""Offline full merge including durable snapshots; mock Stash has zero latency."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time
import zlib

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Bind the current harness packages before --project selects application code.
# Their package initializers do not import any application modules.
import tests.helpers
from benchmarks.benchmark_performance import synthetic_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--frames', type=int, default=400)
    parser.add_argument('--labels', type=int, default=28)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if min(args.frames, args.labels, args.repeats) < 1:
        parser.error('frames, labels and repeats must be positive')
    sys.path.insert(0, str(args.project.resolve()))
    from stash_ai import cli as app
    from stash_ai.publication import merged as merge_sync
    from stash_ai import merge
    from stash_ai.diagnostics import MergeDiagnostics
    from stash_ai.runtime import RunReport
    from stash_ai.models import SOURCES
    from stash_ai.evidence.temporal.policy import TemporalPolicy
    from tests.helpers.stash import Stash
    app.TEMPORAL_POLICY = TemporalPolicy()
    app.MAX_MARKERS_PER_SCENE = app.MARKER_MAX_TAGS_PER_FRAME = 0
    app.MARKER_MIN_HITS = 2
    app.MARKER_MERGE_GAP_SECONDS = 0
    app.APPLY_SCENE_TAGS = app.CREATE_MARKERS = app.CREATE_MISSING = True
    source = synthetic_records(SOURCES, args.frames, args.labels)
    input_sha256 = hashlib.sha256(json.dumps(source, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    policies = {s: frozenset() for s in SOURCES}
    rows = []
    for _ in range(args.repeats):
        gc.collect()
        report = RunReport(write=lambda _: None)
        with tempfile.TemporaryDirectory() as directory, merge.StateStore(Path(directory)/'state.sqlite3',
                'offline-benchmark', cache_bytes=128*2**20) as state:
            for model, segments in source.items(): state.put('1', model, {'markers': segments[0]})
            state.report = report
            stash = Stash()
            with MergeDiagnostics(report):
                start, cpu = time.perf_counter(), time.process_time()
                with report.measure('merge_compute'):
                    hits, spans = merge.evidence(app, state, '1', policies, policies)
                with report.measure('merge_plan'):
                    counts = merge_sync.sync_merged(app, stash, app.TagResolver(stash, source=merge.AI), state,
                                              stash.find_scene('1'), hits, spans, report=report)
                elapsed, cpu_elapsed = time.perf_counter()-start, time.process_time()-cpu
            stages = dict(report.timings)  # Verification below is outside the measured region.
            blobs = state.connection.execute('SELECT kind,published FROM snapshots ORDER BY kind').fetchall()
            sizes = {kind: dict(json_bytes=len(zlib.decompress(blob)), compressed_bytes=len(blob)) for kind, blob in blobs}
            logical = {kind: state.unpack(blob) for kind, blob in blobs}
            canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
            rows.append(dict(wall_seconds=elapsed, cpu_seconds=cpu_elapsed, stages=stages,
                             counts=dict(counts), snapshots=sizes,
                             logical_snapshot_sha256=hashlib.sha256(canonical(logical)).hexdigest(),
                             stash_sha256=hashlib.sha256(canonical(stash.scene)).hexdigest(),
                             snapshot_sha256=hashlib.sha256(b''.join(blob for _, blob in blobs)).hexdigest()))
            # Keep verification's expanded snapshot out of the next trial's working set.
            del logical, blobs, hits, spans, stash
    if len({row['snapshot_sha256'] for row in rows}) != 1:
        raise RuntimeError('Non-deterministic snapshots')
    if len({row['logical_snapshot_sha256'] for row in rows}) != 1 or len({row['stash_sha256'] for row in rows}) != 1:
        raise RuntimeError('Non-deterministic logical output')
    result = dict(frames=args.frames, labels=args.labels, models=len(SOURCES), input_sha256=input_sha256,
                  median_seconds=statistics.median(row['wall_seconds'] for row in rows), trials=rows)
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(rendered+'\n', encoding='utf-8')
    print(rendered)


if __name__ == '__main__':
    main()
