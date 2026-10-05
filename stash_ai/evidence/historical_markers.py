"""Reconstruct pre-temporal markers for frozen publication/cleanup snapshots.

Fresh inference always uses the temporal engine. This adapter preserves the
first-frame extras and interval identities recorded by historical snapshots.
"""
from dataclasses import asdict, dataclass
from stash_ai.domain import FramePrediction, MarkerSpec
from stash_ai.evidence.temporal.policy import TemporalPolicy


def replay_markers(app, frames, core_exclusions, *, threshold):
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



@dataclass(frozen=True)
class HistoricalPolicy(TemporalPolicy):
    """Type identifying a frozen pre-temporal publication, only for replay."""


def frozen_policy(options):
    """Decode the historical discriminator at the snapshot boundary."""
    algorithm = options['algorithm']
    if algorithm not in ('temporal', 'historical'):
        raise ValueError('Invalid frozen publication algorithm.')
    policy = TemporalPolicy.from_snapshot(options['policy'])
    if algorithm == 'temporal':
        return policy
    return HistoricalPolicy(**{**asdict(policy),
        'marker_continue_ratio': 1.0, 'scene_support_ratio': 1.0,
        'shared_marker_frames': False, 'shared_marker_extras': False,
        'shared_scene_tags': False, 'occurrence_markers': False})


def marker_contributions(app, records, core_policies, *, tag_policies=None, **validation):
    """Replay pre-temporal spans without entering the current temporal builder."""
    from stash_ai.domain import Cue, FramePrediction
    from stash_ai.evidence.labels import raw_label
    from stash_ai.evidence.thresholds import marker_threshold, validate_marker_cache
    from stash_ai.evidence.shared_markers import prune_blocked_extras
    from stash_ai.exclusions import policies_or_empty, core_policies_with_global
    tags = policies_or_empty(tag_policies)
    cores = core_policies_with_global(core_policies, tags)
    validate_marker_cache(app, records, **validation)
    spans = []
    for model, segments in records.items():
        for segment in segments:
            if segment['mode'] == 'frames':
                frames = [FramePrediction(Cue(row['start'], row['end'], '', (0, 0, 1, 1)),
                    tuple((raw_label(tag), score) for tag, score in row['tags'] if tag not in tags[model]))
                    for row in segment['items']]
                markers, _ = replay_markers(app, frames, cores[model],
                                           threshold=marker_threshold(segment, app.MARKER_THRESHOLDS))
                values = [dict(tag=m.tag, start=m.start, end=m.end, score=m.score, hits=m.hits,
                               extra_tags=list(m.extra_tags)) for m in markers]
            elif segment['mode'] == 'spans':
                values = segment['items']
            else:
                raise ValueError('Unsupported stored marker evidence.')
            spans.extend(prune_blocked_extras({**item, 'source': model, 'meta': segment['meta']}, tags[model])
                         for item in values if item['tag'] not in cores[model])
    return spans
