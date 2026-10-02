"""Offline CPU benchmark; no Stash connection or model loading.

Run against this directory and a previous extracted release with --project.
Compare hashes as well as timings. Does not estimate GPU or full-library time.
"""
import argparse
import gc
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def synthetic_records(sources, frames, labels):
    return {source: [dict(mode='frames', meta=dict(origin='inference',
        marker_threshold=.8, frame_input_signature='benchmark',
        inference_signature=source + '-benchmark'), items=[
            dict(start=i, end=i+1, tags=[(f'tag_{t}', .9) for t in range(labels)
                if (i // (4+t % 7)+t) % 4 != 0]) for i in range(frames)])]
        for source in sources}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--frames', type=int, default=400)
    parser.add_argument('--labels', type=int, default=28)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if min(args.frames, args.labels, args.repeats) < 1:
        parser.error('frames, labels and repeats must be positive')
    sys.path.insert(0, str(args.project.resolve()))
    import numpy as np
    from stash_ai import cli as app
    from stash_ai import inference
    from stash_ai import merge
    from stash_ai.temporal import TemporalPolicy
    from stash_ai.taggers import SOURCES
    app.TEMPORAL_POLICY = TemporalPolicy()
    app.MARKER_TAG_BLOCKLIST = set()
    app.MARKER_MIN_HITS = 2
    app.MARKER_MERGE_GAP_SECONDS = 0
    app.MARKER_MAX_TAGS_PER_FRAME = app.MAX_MARKERS_PER_SCENE = 0
    optimized = 'borrow' in inspect.signature(merge.union_intervals).parameters
    records = synthetic_records(SOURCES, args.frames, args.labels)
    policies = {source: frozenset() for source in SOURCES}
    times, hashes = [], set()
    for _ in range(args.repeats):
        gc.collect()
        start = time.perf_counter()
        spans = merge.marker_contributions(app, records, policies)
        merged = merge.union_intervals(spans, **({'borrow': True} if optimized else {}))
        times.append(time.perf_counter()-start)
        hashes.add(fingerprint(merged))
        sizes = dict(contributions=len(spans), markers=len(merged))
        del spans, merged
    if len(hashes) != 1:
        raise RuntimeError('Non-deterministic merge')
    # Sparse model output: retain only a small fraction of a large vocabulary.
    values = np.random.default_rng(43).uniform(0, .3, (512, 20000)).astype(np.float32)
    values[:, ::313] = .9
    labels = [f'tag_{i}' for i in range(values.shape[1])]
    filtering = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        selected = [inference.select_scores(row if optimized else row.tolist(), labels, .65, 0) for row in values]
        filtering.append(time.perf_counter()-start)
    print(json.dumps(dict(
        frames=args.frames, labels=args.labels, sources=len(SOURCES), repeats=args.repeats,
        merge=dict(seconds=times, median_seconds=statistics.median(times), sha256=hashes.pop(), **sizes),
        score_filter=dict(seconds=filtering, median_seconds=statistics.median(filtering),
            shape=list(values.shape), sha256=fingerprint(selected))), indent=2))


if __name__ == '__main__':
    main()
