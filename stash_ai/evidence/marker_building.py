"""Build temporal or classic marker specs from explicit runtime settings."""
from stash_ai.domain import FramePrediction, MarkerSpec


def build_markers(app, frames: list[FramePrediction], core_exclusions: frozenset[str], *,
                  threshold=None, seeds=None, identity=None, cache=None) -> tuple[list[MarkerSpec], int]:
    """Group eligible cores independently; exclusions may still be extras."""
    if app.TEMPORAL_POLICY.enabled:
        from stash_ai.evidence.temporal.markers import build_markers as temporal_markers
        return temporal_markers(app, frames, core_exclusions,
                                threshold=threshold, seeds=seeds, identity=identity, extra_cache=cache)
    if threshold is not None:
        frames = [FramePrediction(frame.cue, tuple((tag, score) for tag, score in frame.tags if score >= threshold)) for frame in frames]
    active: dict[str, dict] = {}
    finished: list[MarkerSpec] = []

    def finish(tag: str) -> None:
        run = active.pop(tag)
        if run["hits"] >= app.MARKER_MIN_HITS:
            finished.append(MarkerSpec(tag=tag, **run))

    for frame in frames:
        core_tags = [(tag, score) for tag, score in frame.tags if tag not in core_exclusions]
        additional_tags = frame.tags
        if app.MARKER_MAX_TAGS_PER_FRAME:
            core_tags = core_tags[:app.MARKER_MAX_TAGS_PER_FRAME]
            additional_tags = additional_tags[:app.MARKER_MAX_TAGS_PER_FRAME]
        for tag, score in core_tags:
            if tag in active and frame.cue.start - active[tag]["end"] > app.MARKER_MERGE_GAP_SECONDS + 0.001:
                finish(tag)
            if tag not in active:
                active[tag] = {"start": frame.cue.start, "end": frame.cue.end,
                               "score": score, "hits": 1,
                               "extra_tags": tuple(name for name, _ in additional_tags if name != tag)}
            else:
                run = active[tag]
                run["end"] = max(run["end"], frame.cue.end)
                run["score"] = max(run["score"], score)
                run["hits"] += 1
    for tag in list(active):
        finish(tag)
    total = len(finished)
    finished.sort(key=lambda marker: (-marker.score, marker.start, marker.tag))
    if app.MAX_MARKERS_PER_SCENE:
        finished = finished[:app.MAX_MARKERS_PER_SCENE]
    return sorted(finished, key=lambda marker: (marker.start, marker.tag)), total
