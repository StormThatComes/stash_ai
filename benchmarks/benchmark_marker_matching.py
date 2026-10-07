"""Offline repeat merge with many existing markers; mock Stash has zero latency."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Keep this checkout's harness/mocks when --project selects application code.
import tests.helpers


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--markers', type=int, default=1000)
    parser.add_argument('--labels', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--point-only', action='store_true')
    parser.add_argument('--cleanup', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if min(args.markers, args.labels, args.repeats) < 1 or args.labels > args.markers:
        parser.error('markers/labels/repeats must be positive; labels cannot exceed markers')
    sys.path.insert(0, str(args.project.resolve()))
    from stash_ai.runtime.context import RuntimeContext
    app = RuntimeContext()
    import stash_ai.evidence.labels as merge_labels
    import stash_ai.storage.store as merge_store
    from stash_ai.publication import merged as merge_sync
    from stash_ai.runtime.report import RunReport
    from stash_ai.evidence.temporal.policy import TemporalPolicy
    from tests.helpers.stash import Stash
    from tests.helpers.evidence import span, empty_policies

    app.TEMPORAL_POLICY = TemporalPolicy(shared_marker_frames=False, shared_marker_extras=False)
    app.APPLY_SCENE_TAGS = False
    app.CREATE_MARKERS = app.CREATE_MISSING = True
    app.MARKER_USE_END_TIME = not args.point_only
    app.MARKER_MAX_TAGS_PER_FRAME = 0
    app.MARKER_MERGE_GAP_SECONDS = 0
    spans = [span('wd', float(i*3), float(i*3+1), tag=f'core_{i % args.labels}')
             for i in range(args.markers)]
    policies = empty_policies()
    trials = []
    with tempfile.TemporaryDirectory() as directory, merge_store.StateStore(
            Path(directory)/'state.sqlite3', 'marker-matching-benchmark', cache_bytes=128*2**20) as store:
        stash = Stash()
        resolver = app.services.TagResolver(stash, source=merge_labels.AI)
        merge_sync.sync_merged(app, stash, resolver, store, stash.find_scene('1'), [], spans, tag_policies=policies)
        for _ in range(args.repeats):
            gc.collect()
            report = RunReport(write=lambda _: None)
            store.report = report
            current = stash.find_scene('1')  # Same input; API latency is excluded.
            start, cpu = time.perf_counter(), time.process_time()
            counts = merge_sync.sync_merged(app, stash, resolver, store, current, [], spans,
                                       tag_policies=policies, cleanup=args.cleanup, report=report)
            elapsed, cpu_elapsed = time.perf_counter()-start, time.process_time()-cpu
            stages = dict(report.timings)
            snapshot = store.snapshot('1', 'markers')[0]
            trials.append(dict(wall_seconds=elapsed, cpu_seconds=cpu_elapsed, stages=stages,
                               counts=dict(counts), logical_snapshot_sha256=digest(snapshot),
                               stash_sha256=digest(stash.scene)))
            if counts['markers_unchanged'] != args.markers or any(counts[k] for k in (
                    'markers_created', 'markers_updated', 'markers_deleted')):
                raise RuntimeError('Repeat merge changed native markers')
            del snapshot, current
    for key in ('logical_snapshot_sha256', 'stash_sha256'):
        if len({trial[key] for trial in trials}) != 1:
            raise RuntimeError('Non-deterministic output: ' + key)
    result = dict(markers=args.markers, labels=args.labels, point_only=args.point_only,
                  cleanup=args.cleanup, input_sha256=digest(spans),
                  median_seconds=statistics.median(t['wall_seconds'] for t in trials),
                  median_matching_seconds=statistics.median(t['stages']['marker_matching'] for t in trials),
                  trials=trials)
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(rendered + '\n', encoding='utf-8')
    print(rendered)


if __name__ == '__main__':
    main()

