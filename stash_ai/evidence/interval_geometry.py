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
    for tag, items in sorted(grouped.items()):
        items.sort(key=lambda item: (item["start"], item["end"] if item["end"] is not None else item["start"], item["source"]))
        components = []
        for item in items:
            end = item["end"] if item["end"] is not None else item["start"]
            if not components or item["start"] > components[-1]["bound"]:
                components.append(dict(tag=tag, start=item["start"], bound=end, contributors=[]))
            component = components[-1]
            component["bound"] = max(component["bound"], end)
            if item not in component["contributors"]:
                component["contributors"].append(item)
        for component in components:
            contributors = component["contributors"]
            bound = component.pop("bound")
            component["end"] = bound if any(item["end"] is not None for item in contributors) else None
            component["sources"] = [key for key in SOURCES if any(item["source"] == key or
                                    key in item.get("support_sources", []) for item in contributors)]
            component["extra_tags"] = sorted({raw_label(tag) for item in contributors for tag in item.get("extra_tags", [])}
                                             - {tag})
            component["extra_ids"] = sorted({str(i) for item in contributors for i in item.get("extra_ids", [])})
            merged.append(component)
    return sorted(merged, key=lambda item: (item["start"], item["tag"]))


def overlaps(marker, wanted, tolerance):
    start, end = marker["seconds"], marker.get("end_seconds")
    end = start if end is None else end
    target_end = wanted.get("end_seconds")
    target_end = wanted["seconds"] if target_end is None else target_end
    return (max(start, wanted["seconds"]) <= min(end, target_end)
            or abs(start - wanted["seconds"]) <= tolerance)

