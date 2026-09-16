"""Automatic qualification from sparse detections; no label allowlist or images."""
from __future__ import annotations

from collections import defaultdict
from bisect import bisect_left
from dataclasses import asdict, dataclass, fields, replace
import math


@dataclass(frozen=True)
class TemporalPolicy:
    enabled: bool = True
    shared_marker_frames: bool = True
    shared_marker_extras: bool = True
    short_marker_extras: bool = True
    extra_short_min_hits: int = 3
    shared_scene_tags: bool = True
    occurrence_markers: bool = True
    marker_continue_ratio: float = 0.85
    marker_max_weak_seconds: float = 15.0
    extra_min_hits: int = 2
    extra_min_fraction: float = 0.20
    consensus_min_models: int = 2
    scene_support_ratio: float = 0.85
    scene_min_hits: int = 2
    scene_repeat_hits: int = 4
    scene_repeat_run: int = 3
    scene_repeat_fraction: float = 0.10
    scene_strong_margin: float = 0.50

    @classmethod
    def from_snapshot(cls, values):
        # Old published markers may retain retired video options. Never use them.
        saved = {key: value for key, value in values.items() if not key.startswith("refine_")}
        # Cleanup must retain the algorithm used by older published snapshots.
        saved.setdefault("shared_marker_frames", False)
        saved.setdefault("shared_marker_extras", False)
        saved.setdefault("short_marker_extras", False)
        saved.setdefault("shared_scene_tags", False)
        saved.setdefault("occurrence_markers", False)
        policy = cls(**saved)
        policy.validate()
        return policy

    def validate(self):
        for name in ("enabled", "shared_marker_frames", "shared_marker_extras", "short_marker_extras", "shared_scene_tags", "occurrence_markers"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be true or false.")
        for name in ("marker_continue_ratio", "scene_support_ratio"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1].")
        for name in ("extra_min_fraction", "scene_repeat_fraction", "scene_strong_margin"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1].")
        for name in ("extra_min_hits", "extra_short_min_hits", "scene_min_hits", "scene_repeat_hits", "scene_repeat_run"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be an integer >= 1.")
        if type(self.consensus_min_models) is not int or not 2 <= self.consensus_min_models <= 4:
            raise ValueError("consensus_min_models must be in 2..4.")
        value = self.marker_max_weak_seconds
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid marker_max_weak_seconds.")


def load_policy(app, path, enabled=True):
    if not enabled:
        return TemporalPolicy(enabled=False)
    data = app.load_unique_yaml(path, "Temporal policy")
    if not isinstance(data, dict) or type(data.get("version")) is not int or data["version"] != 1:
        raise ValueError("Temporal policy requires version: 1.")
    allowed = {f.name for f in fields(TemporalPolicy)}
    if any(isinstance(key, str) and key.startswith("refine_") for key in data):
        raise ValueError("Video-file analysis has been removed. Remove refine_* settings from the temporal policy; only Stash VTT/sprites are supported.")
    if set(data) - allowed - {"version"}:
        raise ValueError("Unknown temporal policy keys: " + ", ".join(sorted(set(data) - allowed - {"version"})))
    policy = TemporalPolicy(**{k: v for k, v in data.items() if k != "version"})
    policy.validate()
    return policy


def marker_floor(policy, threshold):
    return threshold * policy.marker_continue_ratio if policy.enabled else threshold


class SceneAccumulator:
    """One small record per observed label, never a frame-by-vocabulary matrix."""
    def __init__(self, threshold, policy, state=None):
        self.threshold, self.policy = threshold, policy
        state = state or {}
        self.items = {int(k): dict(v) for k, v in state.get("items", {}).items()}
        self.frames = state.get("frames", 0)
        self.covered = state.get("covered", 0.0)
        self.last_end = state.get("last_end", 0.0)

    def add(self, scores, cues):
        import numpy as np
        floor = self.threshold * self.policy.scene_support_ratio
        for row, cue in zip(scores, cues):
            self.frames += 1
            self.covered += max(0.0, cue.end - max(cue.start, self.last_end))
            self.last_end = max(self.last_end, cue.end)
            for index in np.flatnonzero(row >= floor):
                index, score = int(index), float(row[index])
                if score < floor:
                    continue
                item = self.items.setdefault(index, dict(hits=0, high_hits=0, max_score=0.0,
                    seconds=0.0, run_hits=0, run_start=cue.start, last_end=cue.start,
                    longest_hits=0, longest_seconds=0.0))
                if (not item["hits"] or item.get("last_frame") != self.frames - 1
                        or cue.start > item["last_end"] + 0.001):
                    item["run_hits"], item["run_start"] = 0, cue.start
                item["hits"] += 1
                item["high_hits"] += int(score >= self.threshold)
                # Keep an actual sampled time, including detections below the
                # marker retention floor. Old checkpoints may lack this field;
                # only an observed maximum can supply its missing timestamp.
                if score > item["max_score"] or (score == item["max_score"] and "peak_time" not in item):
                    item["peak_time"] = cue.start
                item["max_score"] = max(item["max_score"], score)
                item["seconds"] += max(0.0, cue.end - max(cue.start, item["last_end"]))
                item["last_end"] = max(item["last_end"], cue.end)
                item["run_hits"] += 1
                item["last_frame"] = self.frames
                item["longest_hits"] = max(item["longest_hits"], item["run_hits"])
                item["longest_seconds"] = max(item["longest_seconds"], item["last_end"] - item["run_start"])

    def state(self):
        # A checkpoint must not share mutable counters with an unfinished batch.
        return dict(items={k: dict(v) for k, v in self.items.items()}, frames=self.frames,
                    covered=self.covered, last_end=self.last_end)

    def statistics(self, labels):
        return {labels[index]: dict(hits=item["hits"], high_hits=item["high_hits"], max_score=item["max_score"],
            sampled_frames=self.frames, sampled_seconds=self.covered, support_seconds=item["seconds"],
            sample_fraction=item["hits"] / self.frames if self.frames else 0.0,
            coverage_fraction=min(1.0, item["seconds"] / self.covered) if self.covered else 0.0,
            longest_hits=item["longest_hits"], longest_seconds=item["longest_seconds"],
            **({"peak_time": item["peak_time"]} if "peak_time" in item else {}))
            for index, item in self.items.items()}


def scene_reason(score, stats, threshold, policy):
    # Legacy records do not contain missing per-frame evidence: keep their rule.
    if not policy.enabled or stats is None:
        return "aggregate" if score is None or score >= threshold else None
    if stats["max_score"] >= threshold + (1.0 - threshold) * policy.scene_strong_margin:
        return "strong_peak"
    if score >= threshold and stats["high_hits"] >= policy.scene_min_hits:
        return "repeated_high"
    if (stats["hits"] >= policy.scene_repeat_hits and stats["longest_hits"] >= policy.scene_repeat_run
            and stats["sample_fraction"] >= policy.scene_repeat_fraction):
        return "persistent_support"
    return None


def consensus_seeds(records, policies, blocklist, policy):
    """Only identical sampled inputs/times can vote, at most once per model."""
    if not policy.enabled:
        return {}
    votes = defaultdict(set)
    for model, segments in records.items():
        for segment in segments:
            meta = segment["meta"]
            identity = meta.get("frame_input_signature")
            high = meta.get("marker_threshold")
            if segment["mode"] != "frames" or not identity or high is None:
                continue
            floor = marker_floor(policy, high)
            for frame in segment["items"]:
                for tag, score in frame["tags"]:
                    if score >= floor and tag not in policies[model] and tag not in blocklist:
                        votes[(identity, frame["start"], tag)].add(model)
    return {key: tuple(sorted(models)) for key, models in votes.items() if len(models) >= policy.consensus_min_models}


def build_markers(app, frames, core_exclusions, *, threshold=None, seeds=None, identity=None,
                  collect_extras=True, extra_group=None, extra_cache=None):
    """High/consensus seeds start runs; weaker observations have a time limit."""
    policy = app.TEMPORAL_POLICY
    high = threshold if threshold is not None else 0.0
    floor = marker_floor(policy, high)
    seeds = seeds or {}
    active, finished = {}, []
    frames = sorted(frames, key=lambda frame: frame.cue.start)
    extra_frames = []

    def finish(tag):
        run = active.pop(tag)
        if run["hits"] < app.MARKER_MIN_HITS:
            return
        finished.append(app.MarkerSpec(tag, run["start"], run["end"], run["score"], run["hits"],
            (), (), tuple(sorted(run["sources"])), run["strong_hits"]))

    for frame in frames:
        cue = frame.cue
        ordered = sorted(((tag, score) for tag, score in frame.tags if tag not in app.MARKER_TAG_BLOCKLIST),
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
    # Inspect every observation inside the finished interval, including a short
    # permitted gap in primary detections. Compute this only for retained markers.
    from stash_fusion import ObservationIndex
    from stash_progress import digest
    events = {}
    for start, end, extras in extra_frames:
        event = events.setdefault(start, dict(start=start, end=end, extras={}))
        event['end'] = max(event['end'], end)
        for label, score in extras:
            event['extras'].setdefault(label, {})[None] = max(score, event['extras'].get(label, {}).get(None, score))
    index = ObservationIndex([events[t] for t in sorted(events)], {}, extra_group or digest(
        {'input': identity, 'frames': extra_frames}), extra_cache)
    qualified = []
    try:
        for marker in finished:
            evidence = {label: value for label, value in index.support(marker.start, marker.end, policy).items() if label != marker.tag}
            extras = [item for item in index.qualified(marker.start, marker.end, policy) if item[0] != marker.tag]
            qualified.append(replace(marker, extra_tags=tuple(item[0] for item in extras), extra_evidence=tuple(extras),
                                     extra_support=evidence))
    finally:
        index.close()
    return sorted(qualified, key=lambda marker: (marker.start, marker.tag)), total


def build_options(app):
    return {"policy": asdict(app.TEMPORAL_POLICY), "settings": {name: getattr(app, name) for name in (
        "MARKER_MIN_HITS", "MARKER_MERGE_GAP_SECONDS", "MARKER_MAX_TAGS_PER_FRAME", "MAX_MARKERS_PER_SCENE")}}


def append_interval(intervals, start, end):
    if intervals and start <= intervals[-1][1]:
        intervals[-1][1] = max(intervals[-1][1], end)
    else:
        intervals.append([start, end])


def longest_extra_run(evidence):
    """Union consecutive index runs only within the same verified timeline."""
    groups = defaultdict(list)
    for item in evidence:
        if item.get('run_group') is not None:
            groups[item['run_group']].extend(item.get('runs', []))
    longest = 0
    for runs in groups.values():
        current = None
        for first, last, start, end in sorted(runs):
            if current is not None and first <= current[1] and start <= current[3] + .001:
                current[1], current[3] = max(current[1], last), max(current[3], end)
            else:
                current = [first, last, start, end]
            longest = max(longest, current[1] - current[0])
    return longest


def qualified_extras(evidence, duration, policy):
    accepted = []
    for label, item in evidence.items():
        hits = item.get('hits', len(item['sample_times']))
        fraction = min(1.0, sum(b-a for a,b in item['intervals']) / duration) if duration > 0 else 0
        short = policy.short_marker_extras and longest_extra_run([item]) >= policy.extra_short_min_hits
        if hits >= policy.extra_min_hits and (fraction + 1e-12 >= policy.extra_min_fraction or short):
            accepted.append((label, hits, fraction))
    return sorted(accepted, key=lambda item: (-item[2], -item[1], item[0]))


def merged_extras(component):
    """Qualify extras over the whole union; duplicate model coverage counts once."""
    records = component["contributors"]
    settings = next((r["build_options"]["policy"] for r in records if r.get("build_options")), None)
    if settings is None or not settings["enabled"]:
        return
    policy = TemporalPolicy.from_snapshot(settings)
    known, legacy, intervals, times = set(), set(), defaultdict(list), defaultdict(set)
    source_evidence = defaultdict(lambda: defaultdict(list))
    support_records = defaultdict(list)
    seen_support = set()
    for record in records:
        if "extra_support" not in record:
            legacy.update(record.get("extra_tags", []))
            continue
        for tag, evidence in record["extra_support"].items():
            # Several source spans can share exactly the same indexed proof.
            # Its intervals, sample times and source records are set-like data.
            identity = (tag, id(evidence))
            if identity in seen_support:
                continue
            seen_support.add(identity)
            known.add(tag)
            intervals[tag].extend(evidence["intervals"])
            times[tag].update(evidence["sample_times"])
            support_records[tag].append(evidence)
            for source, values in evidence.get("sources", {}).items():
                source_evidence[tag][source].append(values)
    duration = component["end"] - component["start"] if component["end"] is not None else 0.0
    accepted, statistics = set(legacy), {}
    def bounded_ranges(intervals):
        ranges = []
        for start, end in sorted(intervals):
            start, end = max(start, component["start"]), min(end, component["end"])
            if end > start:
                append_interval(ranges, start, end)
        return ranges

    for tag in sorted(known - {component["tag"]}):
        ranges = bounded_ranges(intervals[tag])
        seconds = sum(end-start for start,end in ranges)
        fraction = min(1.0, seconds / duration) if duration > 0 else 0.0
        # Each source retains only enough distinct sample times to prove min_hits.
        # This prevents a frame-by-primary-by-extra tensor in the result cache.
        hits_at_least = sum(component["start"] <= time < component["end"] for time in times[tag])
        run_hits = longest_extra_run(support_records[tag]) if policy.short_marker_extras else 0
        coverage = fraction + 1e-12 >= policy.extra_min_fraction
        short = policy.short_marker_extras and run_hits >= policy.extra_short_min_hits
        qualifies = hits_at_least >= policy.extra_min_hits and (coverage or short)
        if qualifies:
            accepted.add(tag)
        statistics[tag] = dict(support_seconds=seconds, coverage_fraction=fraction,
                               hits_at_least=hits_at_least, qualified=qualifies)
        if policy.short_marker_extras:
            statistics[tag].update(longest_run_hits=run_hits,
                                  qualification=('coverage' if coverage else 'short_episode') if qualifies else None)
        if source_evidence[tag]:
            statistics[tag]["source_evidence"] = {}
            for source, records in sorted(source_evidence[tag].items()):
                own_ranges = bounded_ranges([interval for record in records for interval in record["intervals"]])
                own_times = {time for record in records for time in record["sample_times"]
                             if component["start"] <= time < component["end"]}
                if own_ranges:
                    statistics[tag]["source_evidence"][source] = dict(
                        score=max(record["score"] for record in records),
                        support_seconds=sum(b-a for a,b in own_ranges), hits_at_least=len(own_times),
                        signatures=sorted({signature for record in records for signature in record["signatures"]}))
    component["extra_tags"] = sorted(accepted - {component["tag"]})
    component["extra_evidence"] = statistics
