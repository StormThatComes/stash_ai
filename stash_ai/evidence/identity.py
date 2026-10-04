"""Stable evidence identity, excluding transient publication threshold views."""
OVERRIDE = "marker_threshold_override"
SCENE_OVERRIDE = "scene_threshold_override"


def marker_segment_signature(segment):
    # Cleanup must use exactly the evidence that supported published markers.
    from stash_ai.performance import StoredSegment, segment_digest
    if isinstance(segment, StoredSegment) and hasattr(segment, "_evidence_digest"):
        return segment._evidence_digest
    if OVERRIDE not in segment and SCENE_OVERRIDE not in segment:
        return segment_digest(segment)
    signature = segment_digest({k: v for k, v in segment.items() if k not in (OVERRIDE, SCENE_OVERRIDE)})
    if isinstance(segment, StoredSegment):
        segment._evidence_digest = signature
    return signature
