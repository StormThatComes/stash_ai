"""Offline CPU benchmark; no Stash connection or model loading.

Run against checkouts exposing the current benchmark API with --project.
Compare hashes as well as timings. Does not estimate GPU or full-library time.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def synthetic_records(sources, frames, labels):
    return {source: [dict(mode='frames', meta=dict(origin='inference',
        marker_threshold=.8, marker_score_floor=.1, marker_tag_blocklist=[],
        frame_input_signature='benchmark',
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
    from stash_ai.runtime.context import RuntimeContext
    app = RuntimeContext()
    from stash_ai.inference import prediction
    import stash_ai.evidence.intervals as merge_intervals
    import stash_ai.evidence.source_markers as merge_source_markers
    from stash_ai.evidence.temporal.policy import TemporalPolicy
    from stash_ai.models import SOURCES
    app.TEMPORAL_POLICY = TemporalPolicy()
    app.MARKER_MIN_HITS = 2
    app.MARKER_MERGE_GAP_SECONDS = 0
    app.MARKER_MAX_TAGS_PER_FRAME = 0
    records = synthetic_records(SOURCES, args.frames, args.labels)
    policies = {source: frozenset() for source in SOURCES}
    times, hashes = [], set()
    for _ in range(args.repeats):
        gc.collect()
        start = time.perf_counter()
        spans = merge_source_markers.marker_contributions(app, records, policies)
        merged = merge_intervals.union_intervals(spans, borrow=True)
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
        selected = [prediction.select_scores(row, labels, .65, 0) for row in values]
        filtering.append(time.perf_counter()-start)
    print(json.dumps(dict(
        frames=args.frames, labels=args.labels, sources=len(SOURCES), repeats=args.repeats,
        merge=dict(seconds=times, median_seconds=statistics.median(times), sha256=hashes.pop(), **sizes),
        score_filter=dict(seconds=filtering, median_seconds=statistics.median(filtering),
            shape=list(values.shape), sha256=fingerprint(selected))), indent=2))


if __name__ == '__main__':
    main()

