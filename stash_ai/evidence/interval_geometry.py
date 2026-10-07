"""Pure exact-label geometry and native marker overlap rules."""
from collections import defaultdict
from copy import deepcopy
import math
from stash_ai.models import SOURCES
from .labels import raw_label
from .contracts import MarkerContribution, MergedMarker


def append_interval(intervals, start, end):
    if intervals and start <= intervals[-1][1]:
        intervals[-1][1] = max(intervals[-1][1], end)
    else:
        intervals.append([start, end])


def _content_hash(value, memo):
    """Equality-compatible bucket key; collisions still use exact comparison."""
    if not isinstance(value, (dict, list, tuple, set, frozenset)):
        return hash(value)
    identity = id(value)
    if identity in memo:
        return memo[identity]
    if isinstance(value, dict):
        result = hash(frozenset((_content_hash(k, memo), _content_hash(v, memo))
                                for k, v in value.items()))
    elif isinstance(value, (set, frozenset)):
        result = hash(frozenset(_content_hash(v, memo) for v in value))
    else:
        result = hash((type(value), tuple(_content_hash(v, memo) for v in value)))
    memo[identity] = result
    return result


def union_intervals(contributions: list[MarkerContribution], *, borrow=False) -> list[MergedMarker]:
    """One component per exact label. Positive gaps are never bridged here.

    Unknown legacy end times remain points, unless covered by a known interval.
    Scores are kept on contributors; there is no artificial ensemble probability.
    """
    grouped = defaultdict(list)
    memo = {}
    for contribution in contributions:
        item = contribution if borrow else deepcopy(contribution, memo)
        tag = raw_label(item["tag"])
        if item["source"] not in SOURCES:
            raise ValueError("Unknown interval source.")
        start, end = item["start"], item["end"]
        if (isinstance(start, bool) or not isinstance(start, (int, float)) or not math.isfinite(start) or start < 0
                or (end is not None and (isinstance(end, bool) or not isinstance(end, (int, float))
                    or not math.isfinite(end) or end < start))):
            raise ValueError(f"Invalid marker interval for {tag!r}.")
        grouped[tag].append(item)
    merged = []
    hashes = {}
    for tag, items in sorted(grouped.items()):
        items.sort(key=lambda item: (item["start"], item["end"] if item["end"] is not None else item["start"], item["source"]))
        components = []
        buckets = {}
        for item in items:
            end = item["end"] if item["end"] is not None else item["start"]
            if not components or item["start"] > components[-1]["bound"]:
                components.append(dict(tag=tag, start=item["start"], bound=end, contributors=[]))
                buckets = {}
            component = components[-1]
            component["bound"] = max(component["bound"], end)
            key = tuple(item.get(field) for field in
                        ('source', 'start', 'end', 'score', 'hits', 'derived', 'evidence_signature'))
            bucket = buckets.setdefault(key, [])
            if isinstance(bucket, list) and len(bucket) >= 8:
                indexed = defaultdict(list)
                for previous in bucket:
                    indexed[_content_hash(previous, hashes)].append(previous)
                buckets[key] = bucket = indexed
            if isinstance(bucket, dict):
                bucket = bucket.setdefault(_content_hash(item, hashes), [])
            if item not in bucket:
                bucket.append(item)
                component["contributors"].append(item)
        for component in components:
            contributors = component["contributors"]
            bound = component.pop("bound")
            component["end"] = bound if any(item["end"] is not None for item in contributors) else None
            component["sources"] = [key for key in SOURCES if any(item["source"] == key or
                                    key in item.get("support_sources", []) for item in contributors)]
            component["extra_tags"] = sorted({raw_label(tag) for item in contributors for tag in item.get("extra_tags", [])}
                                             - {tag})
            merged.append(component)
    return sorted(merged, key=lambda item: (item["start"], item["tag"]))


def overlaps(marker, wanted, tolerance):
    start, end = marker["seconds"], marker.get("end_seconds")
    end = start if end is None else end
    target_end = wanted.get("end_seconds")
    target_end = wanted["seconds"] if target_end is None else target_end
    return (max(start, wanted["seconds"]) <= min(end, target_end)
            or abs(start - wanted["seconds"]) <= tolerance)
