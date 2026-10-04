"""Offline timing of representative-point selection, without Stash or GPU."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=500)
    parser.add_argument('--labels', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if min(args.frames, args.labels, args.repeats) < 1:
        parser.error('frames, labels and repeats must be positive')
    from stash_ai import cli as app
    from stash_ai.evidence.occurrences import occurrence_contributions
    from stash_ai.storage.segments import StoredSegment, segment_digest
    from stash_ai.models import SOURCES
    from stash_ai.evidence.temporal.policy import TemporalPolicy
    app.TEMPORAL_POLICY = TemporalPolicy()
    app.MAX_MARKERS_PER_SCENE = 0
    policies = {s: frozenset() for s in SOURCES}
    labels = [f'tag_{i}' for i in range(args.labels)]
    output = dict(frames=args.frames, labels=args.labels, models=len(SOURCES), repeats=args.repeats, cases={})
    for case in ('old_sparse_frames', 'new_scene_peaks', 'already_localized'):
        records, hits = {}, []
        for source in SOURCES:
            meta = dict(origin='inference', scene_threshold=.35, marker_threshold=.65,
                        inference_signature=source+'-benchmark', frame_input_signature='same-input')
            segment = StoredSegment(mode='frames', meta=meta, items=[dict(start=i, end=i+1,
                tags=[(tag, .9) for tag in labels] if i % 3 == 0 and case != 'new_scene_peaks' else [])
                for i in range(args.frames)])
            # The merge's existing FrameIndex has already hashed each borrowed
            # record before the occurrence stage begins in the application.
            segment_digest(segment)
            records[source] = [segment]
            hits.extend(dict(tag=tag, source=source, score=.9, meta=meta,
                statistics={'max_score': .9, 'peak_time': 0} if case == 'new_scene_peaks' else {}) for tag in labels)
        spans = [dict(tag=tag) for tag in labels] if case == 'already_localized' else []
        times = []
        for _ in range(args.repeats):
            gc.collect()
            start = time.perf_counter()
            points = occurrence_contributions(app, hits, records, spans, policies, policies)
            times.append(time.perf_counter()-start)
            expected = 0 if spans else len(labels)
            assert len(points) == expected
            assert all(p['start'] == 0 and p['end'] is None and len(p['occurrence_evidence']) == len(SOURCES) for p in points)
        output['cases'][case] = dict(seconds=times, median_seconds=statistics.median(times), points=len(points))
    rendered = json.dumps(output, indent=2)
    if args.output:
        args.output.write_text(rendered+'\n', encoding='utf-8')
    print(rendered)


if __name__ == '__main__':
    main()
