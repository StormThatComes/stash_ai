"""Offline merge CPU benchmark for a cache retaining scores below publication.

Run the same script against both revisions with --project; compare stage hashes
and timings. No SQLite, Stash API, inference or GPU time is included.
"""

import argparse
from dataclasses import asdict, is_dataclass
import gc
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import sys
import time


def json_value(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(type(value).__name__)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False, default=json_value).encode()).hexdigest()


def measure(operation, repeats):
    operation()  # Warm up imports and derived caches outside the timed trials.
    seconds, hashes = [], set()
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        result = operation()
        seconds.append(time.perf_counter() - start)
        hashes.add(fingerprint(result))
        del result
    if len(hashes) != 1:
        raise RuntimeError('Non-deterministic benchmark result')
    return dict(seconds=seconds, median_seconds=statistics.median(seconds), sha256=hashes.pop())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--frames', type=int, default=400)
    parser.add_argument('--labels', type=int, default=12)
    parser.add_argument('--weak-labels', type=int, default=108)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if min(args.frames, args.labels, args.repeats) < 1 or args.weak_labels < 0:
        parser.error('frames, labels and repeats must be positive; weak-labels must be nonnegative')
    sys.path.insert(0, str(args.project.resolve()))
    from stash_ai import cli as app, fusion, merge, temporal
    from stash_ai.domain import Cue, FramePrediction
    from stash_ai.models import SOURCES
    app.TEMPORAL_POLICY = temporal.TemporalPolicy()
    app.MARKER_TAG_BLOCKLIST = set()
    app.MARKER_MIN_HITS = 2
    app.MARKER_MERGE_GAP_SECONDS = 0
    app.MARKER_MAX_TAGS_PER_FRAME = app.MAX_MARKERS_PER_SCENE = 0
    records = {source: [dict(mode='frames', meta=dict(origin='inference',
        marker_threshold=.8, marker_score_floor=.1, marker_tag_blocklist=[],
        frame_input_signature='retained-benchmark', inference_signature=source + '-benchmark'),
        items=[dict(start=i, end=i+1, tags=[
            (f'tag_{j}', .9 if (i+j) % 4 == 0 else .7) for j in range(args.labels)] + [
            (f'weak_{j}', .2 + ((i+j) % 20) / 100) for j in range(args.weak_labels)])
            for i in range(args.frames)])] for source in SOURCES}
    policies = {source: frozenset() for source in SOURCES}
    groups = list(fusion.make_groups(records).values())
    first = next(iter(records.values()))[0]
    frames = [FramePrediction(Cue(row['start'], row['end'], '', (0, 0, 1, 1)),
                                  tuple(row['tags'])) for row in first['items']]
    borrow = {'borrow': True} if 'borrow' in inspect.signature(merge.union_intervals).parameters else {}
    result = dict(frames=args.frames, labels=args.labels, weak_labels=args.weak_labels,
        sources=len(SOURCES), repeats=args.repeats, input_sha256=fingerprint(records),
        observations=measure(lambda: [fusion.observations(app, group, policies) for group in groups], args.repeats),
        temporal_markers=measure(lambda: temporal.build_markers(app, frames, frozenset(), threshold=.8,
            identity='retained-benchmark'), args.repeats),
        merge_cpu=measure(lambda: merge.union_intervals(merge.marker_contributions(app, records, policies),
            **borrow), args.repeats))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
