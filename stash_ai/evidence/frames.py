"""Verified frame groups and lazily prepared source-aware observations."""
from collections import defaultdict
from stash_ai.models import SOURCES
from stash_ai.evidence.temporal.policy import marker_floor
from .observations import ObservationIndex


class FrameIndex:
    """Verify groups once; observations are prepared lazily and reused."""
    def __init__(self, app, records, policies, signatures=None, cache=None, tag_policies=None):
        from stash_ai.evidence.identity import marker_segment_signature
        from stash_ai.exclusions import policies_or_empty
        self.app, self.policies, self.cache = app, policies, cache
        self.tag_policies = policies_or_empty(tag_policies)
        self.signatures = signatures if signatures is not None else {
            id(segment): marker_segment_signature(segment) for segments in records.values() for segment in segments}
        self.groups = make_groups(records, self.signatures)
        self.by_signature = {sig: key for key, group in self.groups.items() for _, _, sig in group['members']}
        self.indexes = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for index in self.indexes.values():
            index.close()

    def get(self, key):
        if key not in self.indexes:
            group = self.groups[key]
            self.indexes[key] = ObservationIndex(observations(self.app, group, self.policies, self.tag_policies), group['context'], key, self.cache)
        return self.indexes[key]


def make_groups(records, signatures=None):
    from stash_ai.evidence.identity import marker_segment_signature
    from stash_ai.signatures import digest
    groups = {}
    for source in SOURCES:
        for segment in records.get(source, []):
            meta = segment["meta"]
            identity = meta.get("frame_input_signature")
            if segment["mode"] != "frames" or not identity or meta.get("marker_threshold") is None:
                continue
            # A signature plus the actual timeline: shifted/missing cues never
            # become matching observations merely because metadata was copied.
            from stash_ai.performance import frame_group
            key = frame_group(segment)
            signature = signatures[id(segment)] if signatures is not None else marker_segment_signature(segment)
            group = groups.setdefault(key, {"members": [], "context": defaultdict(set)})
            if signature not in group["context"][source]:
                group["members"].append((source, segment, signature))
                group["context"][source].add(signature)
    for group in groups.values():
        group["members"].sort(key=lambda item: (list(SOURCES).index(item[0]), item[2]))
        group["context"] = {source: sorted(signatures) for source, signatures in group["context"].items()}
    return dict(sorted(groups.items()))


def observations(app, group, core_policies, tag_policies=None):
    """One event per timestamp; duplicate runs of a model count only once."""
    from stash_ai.evidence.labels import raw_label
    from stash_ai.exclusions import policies_or_empty
    tag_policies = policies_or_empty(tag_policies)
    times = {}
    for source, segment, signature in group["members"]:
        from stash_ai.evidence.thresholds import marker_threshold
        high = marker_threshold(segment, getattr(app, "MARKER_THRESHOLDS", None))
        floor = marker_floor(app.TEMPORAL_POLICY, high)
        sort_floor = min(floor, high)
        for row in segment["items"]:
            start, end = row["start"], row["end"]
            if start < 0 or end <= start:
                raise ValueError("Invalid cached VTT interval for shared observations.")
            event = times.setdefault(start, {"start": start, "end": end, "cores": {}, "extras": {}})
            event["end"] = max(event["end"], end)
            eligible = []
            for tag, score in row["tags"]:
                if tag not in tag_policies[source]:
                    # Validate even observations below the current publication floor.
                    label = raw_label(tag)
                    if score >= sort_floor:
                        eligible.append((label, score))
            ordered = sorted(eligible, key=lambda item: (-item[1], item[0]))
            cores = [(tag, score) for tag, score in ordered if score >= floor and tag not in core_policies[source]]
            extras = [(tag, score) for tag, score in ordered if score >= high]
            if app.MARKER_MAX_TAGS_PER_FRAME:
                cores = cores[:app.MARKER_MAX_TAGS_PER_FRAME]
                extras = extras[:app.MARKER_MAX_TAGS_PER_FRAME]
            for tag, score in cores:
                evidence = event["cores"].setdefault(tag, {}).setdefault(source,
                    {"score": score, "strong": False, "signatures": set()})
                evidence["score"] = max(evidence["score"], score)
                evidence["strong"] |= score >= high
                evidence["signatures"].add(signature)
            for tag, score in extras:
                sources = event["extras"].setdefault(tag, {})
                sources[source] = max(sources.get(source, score), score)
    return [times[start] for start in sorted(times)]
