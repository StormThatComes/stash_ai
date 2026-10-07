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
        isinstance(pair, (list, tuple)) and len(pair) == 2
        and _label(pair[0]) and _score(pair[1])
        for pair in values
    ) and len({pair[0] for pair in values}) == len(values)


def valid_frames(rows, *, checkpoint=False):
    """Sparse frame rows may contain no detections, but need usable timing."""
    return isinstance(rows, list) and all(
        isinstance(row, dict)
        and _number(row.get("start")) and _number(row.get("end"))
        and row["end"] > row["start"]
        and _detections(row.get("tags"))
        and (not checkpoint or "scene_tags" not in row or _detections(row["scene_tags"]))
        for row in rows
    ) and all(b['start'] >= a['end'] - .001 and b['start'] > a['start']
              for a, b in zip(rows, rows[1:]))


def _item(item):
    return (
        isinstance(item, dict) and _label(item.get("tag")) and "score" in item
        and _score(item["score"])
    )


def valid_result(record, kind):
    """One current inference record, active or archived."""
    if kind not in ('scene', 'markers'):
        return False
    allowed_modes = ('scene',) if kind == 'scene' else ('frames',)
    if (not isinstance(record, dict) or record.get('mode') not in allowed_modes
            or not isinstance(record.get('meta'), dict)
            or record['meta'].get('origin') != 'inference'
            or not isinstance(record.get('items'), list)):
        return False
    mode, items = record['mode'], record['items']
    if mode == 'frames':
        return valid_frames(items)
    if not all(_item(item) for item in items):
        return False
    if len({item['tag'] for item in items}) != len(items):
        return False
    if 'frames' in record:
        if not valid_frames(record['frames']):
            return False
        # Scene replay indexes every retained detection by its aggregate label.
        labels = {item['tag'] for item in items}
        if any(tag not in labels for row in record['frames'] for tag, _ in row['tags']):
            return False
    return True
