"""Consensus seeds, bounded weak support and per-source marker runs."""
from collections import defaultdict
from stash_ai.domain import MarkerSpec
from .policy import marker_floor
from stash_ai.evidence.observations import qualify_marker_extras


def consensus_seeds(records, policies, policy, max_tags=0):
    """Only identical sampled inputs/times can vote, at most once per model."""
    votes = defaultdict(set)
    for model, segment in records.items():
        if segment["mode"] != "frames":
            continue
        meta = segment["meta"]
        identity = meta["frame_input_signature"]
        from stash_ai.evidence.thresholds import marker_threshold
        high = marker_threshold(segment)
        floor = marker_floor(policy, high)
        for frame in segment["items"]:
            eligible = sorted(((tag, score) for tag, score in frame['tags']
                               if score >= floor and tag not in policies[model]),
                              key=lambda item: (-item[1], item[0]))
            for tag, score in eligible[:max_tags] if max_tags else eligible:
                votes[(identity, frame["start"], tag)].add(model)
    return {key: tuple(sorted(models)) for key, models in votes.items() if len(models) >= policy.consensus_min_models}


def build_markers(app, frames, core_exclusions, *, threshold, seeds=None, identity=None,
                  collect_extras=True, extra_group=None, extra_cache=None):
    """High/consensus seeds start runs; weaker observations have a time limit."""
    policy = app.TEMPORAL_POLICY
    high = threshold
    floor = marker_floor(policy, high)
    sort_floor = min(floor, high)
    seeds = seeds or {}
    active, finished = {}, []
    frames = sorted(frames, key=lambda frame: frame.cue.start)
    extra_frames = []

    def finish(tag):
        run = active.pop(tag)
        if run["hits"] < app.MARKER_MIN_HITS:
            return
        finished.append(MarkerSpec(tag, run["start"], run["end"], run["score"], run["hits"],
            (), (), tuple(sorted(run["sources"])), run["strong_hits"]))

    for frame in frames:
        cue = frame.cue
        ordered = sorted(((tag, score) for tag, score in frame.tags
                          if score >= sort_floor),
                         key=lambda item: (-item[1], item[0]))
        extras = [(tag, score) for tag, score in ordered if score >= high]
        cores = [(tag, score) for tag, score in ordered if tag not in core_exclusions and score >= floor]
        if app.MARKER_MAX_TAGS_PER_FRAME:
            extras, cores = extras[:app.MARKER_MAX_TAGS_PER_FRAME], cores[:app.MARKER_MAX_TAGS_PER_FRAME]
        if collect_extras:
            extra_frames.append((cue.start, cue.end, extras))
        for tag, score in cores:
            if tag in active and cue.start - active[tag]["end"] > app.MARKER_MERGE_GAP_SECONDS + 0.001:
                finish(tag)
            support = seeds.get((identity, cue.start, tag), ())
            strong = score >= high
            seed = strong or bool(support)
            if tag not in active:
                if not seed:
                    continue
                active[tag] = dict(start=cue.start, end=cue.start, score=score, hits=0,
                                   last_seed_end=cue.end, strong_hits=0, sources=set())
            run = active[tag]
            if seed:
                run["last_seed_end"] = cue.end
            end = min(cue.end, run["last_seed_end"] + policy.marker_max_weak_seconds)
            if end <= cue.start:
                finish(tag)
                continue
            run["end"] = max(run["end"], end)
            run["score"] = max(run["score"], score)
            run["hits"] += 1
            run["strong_hits"] += int(strong)
            if not strong:
                run["sources"].update(support)
    for tag in list(active):
        finish(tag)
    total = len(finished)
    finished.sort(key=lambda marker: (-marker.score, marker.start, marker.tag))
    if app.MAX_MARKERS_PER_SCENE:
        finished = finished[:app.MAX_MARKERS_PER_SCENE]
    if not collect_extras:
        return sorted(finished, key=lambda marker: (marker.start, marker.tag)), total
    qualified = qualify_marker_extras(finished, extra_frames, policy, identity, extra_group, extra_cache)
    return sorted(qualified, key=lambda marker: (marker.start, marker.tag)), total
