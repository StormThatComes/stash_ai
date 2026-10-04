"""Per-segment publication policy, stored separately from inference evidence."""
import json

THRESHOLD_KEYS = {
    'scene': ('scene_threshold_override', 'scene_threshold'),
    'markers': ('marker_threshold_override', 'marker_threshold'),
}


def decode(data):
    from stash_ai.evidence.thresholds import _score
    values = json.loads(data)
    if not isinstance(values, list) or any(value is not None and not _score(value) for value in values):
        raise ValueError('Invalid result thresholds; run --db-check.')
    return values


def encode(values):
    data = json.dumps(values, allow_nan=False, separators=(',', ':'))
    decode(data)
    return data


def overlay(segments, kind, thresholds):
    """Expose transient overrides to domain code, never persist them in evidence."""
    if thresholds is None:
        return segments
    values = decode(thresholds)
    if len(values) != len(segments):
        raise ValueError('Result thresholds do not match segment history; run --db-check.')
    key, meta_key = THRESHOLD_KEYS[kind]
    for segment, threshold in zip(segments, values):
        segment.pop(key, None)
        if threshold is not None and threshold != segment['meta'].get(meta_key):
            segment[key] = threshold
    return segments


def separate(segments, kind):
    """Return immutable evidence and its per-segment threshold policy."""
    key, _ = THRESHOLD_KEYS[kind]
    other = THRESHOLD_KEYS['markers' if kind == 'scene' else 'scene'][0]
    if any(other in segment for segment in segments):
        raise ValueError('Threshold override does not match result kind.')
    values = [segment.get(key) for segment in segments]
    raw = [{field: value for field, value in segment.items() if field != key} for segment in segments]
    return raw, encode(values) if any(value is not None for value in values) else None
