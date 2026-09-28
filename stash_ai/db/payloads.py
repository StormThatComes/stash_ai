"""Structural checks for saved evidence, without loading models or vocabularies."""

import math


def _number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _score(value):
    return type(value) in (int, float) and 0 <= value <= 1


def _label(value):
    # Keep the same label contract as merge, including reserved namespaces.
    from stash_ai.merge import raw_label
    try:
        raw_label(value)
        return True
    except ValueError:
        return False


def _detections(values):
    return isinstance(values, list) and all(
        isinstance(pair, list) and len(pair) == 2
        and _label(pair[0]) and _score(pair[1])
        for pair in values
    )


def valid_frames(rows, *, checkpoint=False):
    """Sparse frame rows may contain no detections, but need usable timing."""
    return isinstance(rows, list) and all(
        isinstance(row, dict)
        and _number(row.get("start")) and _number(row.get("end"))
        and row["end"] > row["start"]
        and _detections(row.get("tags"))
        and (not checkpoint or "scene_tags" not in row or _detections(row["scene_tags"]))
        for row in rows
    )


def _statistics(value):
    if value is None:  # Older aggregate-only results have no temporal statistics.
        return True
    if not isinstance(value, dict):
        return False
    if not all(type(value.get(key)) is int and value[key] >= 0
               for key in ("hits", "high_hits", "longest_hits")):
        return False
    if not all(_score(value.get(key)) for key in ("max_score", "sample_fraction")):
        return False
    if "sampled_frames" in value and not (
            type(value["sampled_frames"]) is int and value["sampled_frames"] >= value["hits"]):
        return False
    return (
        value["high_hits"] <= value["hits"] and value["longest_hits"] <= value["hits"]
        and all(_number(value[key]) for key in (
            "sampled_seconds", "support_seconds", "longest_seconds", "peak_time") if key in value)
        and ("coverage_fraction" not in value or _score(value["coverage_fraction"]))
    )


def _item(item, *, legacy=False):
    return (
        isinstance(item, dict) and _label(item.get("tag")) and "score" in item
        and (_score(item["score"]) or (legacy and item["score"] is None))
    )


def _span(item, *, legacy=False):
    return (
        _item(item, legacy=legacy) and _number(item.get("start")) and "end" in item
        # Imported markers can be points, including an unknown end time.
        and (item["end"] is None or (_number(item["end"]) and item["end"] >= item["start"]))
        and (item.get("hits") is None or (type(item["hits"]) is int and item["hits"] >= 0))
        and ("extra_tags" not in item or (isinstance(item["extra_tags"], list)
                                         and all(_label(tag) for tag in item["extra_tags"])))
        and ("extra_ids" not in item or (isinstance(item["extra_ids"], list)
                                        and all(isinstance(i, str) for i in item["extra_ids"])))
    )


def valid_result_segments(segments, kind):
    if not isinstance(segments, list):
        return False
    allowed_modes = {"scene"} if kind == "scene" else {"frames", "spans"}
    for segment in segments:
        if (not isinstance(segment, dict) or segment.get("mode") not in allowed_modes
                or not isinstance(segment.get("meta"), dict)
                or not isinstance(segment.get("items"), list)):
            return False
        mode, items = segment["mode"], segment["items"]
        legacy = segment["meta"].get("origin") == "legacy"
        if mode == "frames":
            if not valid_frames(items):
                return False
        elif mode == "spans":
            if not all(_span(item, legacy=legacy) for item in items):
                return False
        else:
            if not all(_item(item, legacy=legacy) and _statistics(item.get("statistics"))
                       for item in items):
                return False
            if "frames" in segment:
                if not valid_frames(segment["frames"]):
                    return False
                # Scene replay indexes every retained detection by its aggregate label.
                labels = {item["tag"] for item in items}
                if any(tag not in labels for row in segment["frames"] for tag, _ in row["tags"]):
                    return False
    return True
