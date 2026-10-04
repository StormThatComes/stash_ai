"""Marker confidence and retained frame witnesses for review signals."""
from collections import defaultdict
from bisect import bisect_left, bisect_right
from .signals import _score, _candidate, _flag


def _supporting_scores(app, contribution, tag, segments, postings):
    from stash_ai.evidence.temporal.policy import marker_floor
    from stash_ai.evidence.thresholds import marker_threshold
    from stash_ai.evidence.timeline import Series

    identity = contribution["meta"].get("frame_input_signature")
    if not identity:
        return
    for model in contribution.get("support_sources", ()):
        if model == contribution["source"]:
            continue
        for signature in contribution.get("support_context", {}).get(model, ()):
            key = model, signature
            segment = segments.get(key)
            if (segment is None or segment["mode"] != "frames"
                    or segment["meta"].get("frame_input_signature") != identity):
                continue
            threshold = marker_threshold(segment)
            if not _score(threshold):
                continue
            if key not in postings:
                series = {}
                floor = marker_floor(app.TEMPORAL_POLICY, threshold)
                for position, frame in enumerate(sorted(segment["items"], key=lambda row: row["start"])):
                    for label, score in frame["tags"]:
                        if _score(score) and score >= floor:
                            if label not in series:
                                series[label] = Series()
                            series[label].append(position, frame["start"], frame["end"], score)
                postings[key] = series
            series = postings[key].get(tag)
            if series is None:
                continue
            lo = bisect_left(series.starts, contribution["start"])
            end = contribution["end"]
            hi = (bisect_right(series.starts, contribution["start"]) if end is None
                  else bisect_left(series.starts, end))
            if lo < hi:
                yield model, max(series.scores[lo:hi]), threshold


def _marker_signals(app, store, scene, spans, rows, options):
    from stash_ai.evidence.identity import marker_segment_signature
    from stash_ai.evidence.intervals import union_intervals
    from stash_ai.evidence.marker_identity import is_occurrence
    from stash_ai.signatures import digest
    from stash_ai.evidence.thresholds import marker_threshold

    effective, by_meta, segments_by_signature, postings = {}, defaultdict(set), {}, {}
    for model, segments in store.records(scene, "markers", borrow=True).items():
        for segment in segments:
            threshold = marker_threshold(segment)
            key = model, marker_segment_signature(segment)
            effective[key] = threshold
            segments_by_signature[key] = segment
            by_meta[model, digest(segment["meta"])].add(threshold)

    for marker in union_intervals(spans, borrow=True):
        if is_occurrence(marker):
            continue  # Its scene qualification is reviewed above.
        duration = (marker["end"] - marker["start"]) if marker["end"] is not None else None
        for contribution in marker["contributors"]:
            model, tag = contribution["source"], marker["tag"]
            threshold = effective.get((model, contribution.get("evidence_signature")))
            if threshold is None:
                values = by_meta.get((model, digest(contribution["meta"])), set())
                threshold = next(iter(values)) if len(values) == 1 else None
            score = contribution.get("score")
            witnesses = [(model, score, threshold, False)]
            witnesses.extend((peer, score, threshold, True) for peer, score, threshold in
                             _supporting_scores(app, contribution, tag, segments_by_signature, postings))
            for source, score, threshold, support_only in witnesses:
                row = _candidate(rows, source, tag)
                detail = dict(kind="marker", start=marker["start"], end=marker["end"],
                              score=score, threshold=threshold, support_only=support_only)
                if _score(score) and _score(threshold) and score <= threshold + options.margin + 1e-12:
                    _flag(row, "weak_or_borderline", detail)
                if duration is not None and duration >= options.long_marker_seconds:
                    _flag(row, "long_marker", {**detail, "duration_seconds": duration})

