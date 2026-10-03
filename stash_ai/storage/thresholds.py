"""Effective publication thresholds over unchanged inference payloads."""

THRESHOLD_KEYS = {
    'scene': ('scene_threshold_override', 'scene_threshold'),
    'markers': ('marker_threshold_override', 'marker_threshold'),
}


def overlay(segments, kind, threshold):
    """A table value supersedes legacy inline overrides, including a restoration.

    Mutate only freshly decoded lists. Keep the original metadata and evidence,
    and expose the same inline view used by merge, cleanup and signature checks.
    """
    if threshold is None:
        return segments
    key, meta_key = THRESHOLD_KEYS[kind]
    for segment in segments:
        segment.pop(key, None)
        if threshold != segment['meta'].get(meta_key):
            segment[key] = threshold
    return segments
