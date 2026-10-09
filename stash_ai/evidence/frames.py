"""Verified frame groups and lazily prepared source-aware observations."""
from collections import defaultdict
from stash_ai.models import SOURCES
from stash_ai.evidence.temporal.policy import marker_floor
from .observations import ObservationIndex


class FrameIndex:
    """Verify groups once; observations are prepared lazily and reused."""
    def __init__(self, app, records, policies, *, tag_policies, cache=None, report=None):
        from stash_ai.evidence.identity import marker_segment_signature
        from stash_ai.exclusions import require_policies
        self.app, self.policies, self.cache = app, require_policies(policies), cache
        self.tag_policies = require_policies(tag_policies)
        self.report = report
        from .eligibility import FrameMemo
        self.memo = FrameMemo(cache)
        from stash_ai.evidence.formats import validate_current
        for record in records.values():
            validate_current(record)
        self.signatures = {
            id(segment): marker_segment_signature(segment) for segment in records.values()}
        self.groups = make_groups(records, self.signatures)
        self.by_signature = {sig: key for key, group in self.groups.items() for _, _, sig in group['members']}
        self.indexes = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        try:
            for index in self.indexes.values():
                index.close()
        finally:
            self.memo.close()

    def get(self, key):
        if key not in self.indexes:
            group = self.groups[key]
            self.indexes[key] = ObservationIndex(observations(self.app, group, self.policies, self.tag_policies, memo=self.memo), group['context'], key, self.cache)
        return self.indexes[key]


def make_groups(records, signatures=None):
    from stash_ai.evidence.identity import marker_segment_signature
    groups = {}
    for source, segment in records.items():
        if segment["mode"] != "frames":
            raise ValueError("Frame grouping requires current frame evidence.")
        # A signature plus the actual timeline: shifted/missing cues never
        # become matching observations merely because metadata was copied.
        from stash_ai.storage.segments import frame_group
        key = frame_group(segment)
        signature = signatures[id(segment)] if signatures is not None else marker_segment_signature(segment)
        group = groups.setdefault(key, {"members": [], "context": {}})
        group["members"].append((source, segment, signature))
        group["context"][source] = signature
    for group in groups.values():
        group["members"].sort(key=lambda item: list(SOURCES).index(item[0]))
    return dict(sorted(groups.items()))


def observations(app, group, core_policies, tag_policies, *, memo=None):
    """One event per timestamp; duplicate runs of a model count only once."""
    from stash_ai.exclusions import require_policies
    tag_policies = require_policies(tag_policies)
    times = {}
    for source, segment, signature in group["members"]:
        from stash_ai.evidence.thresholds import marker_threshold
        high = marker_threshold(segment)
        floor = marker_floor(app.TEMPORAL_POLICY, high)
        for row in segment["items"]:
            start, end = row["start"], row["end"]
            if start < 0 or end <= start:
                raise ValueError("Invalid cached VTT interval for shared observations.")
            event = times.setdefault(start, {"start": start, "end": end, "cores": {}, "extras": {}})
            event["end"] = max(event["end"], end)
            from .eligibility import selections
            cores, extras = selections(row['tags'], high, floor, core_policies[source],
                                       tag_policies[source], app.MARKER_MAX_TAGS_PER_FRAME,
                                       validate=True, memo=memo, source=source)
            for tag, score in cores:
                evidence = event["cores"].setdefault(tag, {}).setdefault(source,
                    {"score": score, "strong": False, "signature": signature})
                evidence["score"] = max(evidence["score"], score)
                evidence["strong"] |= score >= high
            for tag, score in extras:
                sources = event["extras"].setdefault(tag, {})
                sources[source] = max(sources.get(source, score), score)
    return [times[start] for start in sorted(times)]
