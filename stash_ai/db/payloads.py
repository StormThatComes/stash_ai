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
    from stash_ai.evidence.labels import raw_label
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


def _archived_statistics(value):
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


def _item(item):
    return (
        isinstance(item, dict) and _label(item.get("tag")) and "score" in item
        and _score(item["score"])
    )


def _archived_span(item):
    return (
        _item(item) and _number(item.get("start")) and "end" in item
        # Archived inference can retain old point geometry.
        and (item["end"] is None or (_number(item["end"]) and item["end"] >= item["start"]))
        and (item.get("hits") is None or (type(item["hits"]) is int and item["hits"] >= 0))
        and ("extra_tags" not in item or (isinstance(item["extra_tags"], list)
                                         and all(_label(tag) for tag in item["extra_tags"])))
        and ("extra_ids" not in item or (isinstance(item["extra_ids"], list)
                                        and all(isinstance(i, str) for i in item["extra_ids"])))
    )


def valid_result(record, kind, *, archived=False):
    """One inference record; historical span/statistic shapes belong to archives."""
    if kind not in ('scene', 'markers'):
        return False
    allowed_modes = ('scene',) if kind == 'scene' else (('frames', 'spans') if archived else ('frames',))
    if (not isinstance(record, dict) or record.get('mode') not in allowed_modes
            or not isinstance(record.get('meta'), dict)
            or record['meta'].get('origin') != 'inference'
            or not isinstance(record.get('items'), list)):
        return False
    mode, items = record['mode'], record['items']
    if mode == 'frames':
        return valid_frames(items)
    if mode == 'spans':
        return all(_archived_span(item) for item in items)
    if not all(_item(item) for item in items):
        return False
    if archived and not all(_archived_statistics(item.get('statistics')) for item in items):
        return False
    if 'frames' in record:
        if not valid_frames(record['frames']):
            return False
        # Scene replay indexes every retained detection by its aggregate label.
        labels = {item['tag'] for item in items}
        if any(tag not in labels for row in record['frames'] for tag, _ in row['tags']):
            return False
    return True
