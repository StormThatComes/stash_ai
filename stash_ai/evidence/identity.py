"""Stable evidence identity, excluding the transient publication threshold."""
from stash_ai.storage.thresholds import PUBLICATION_THRESHOLD


def marker_segment_signature(segment):
    from stash_ai.storage.segments import StoredSegment, segment_digest
    if isinstance(segment, StoredSegment) and hasattr(segment, '_evidence_digest'):
        return segment._evidence_digest
    signature = segment_digest(segment if PUBLICATION_THRESHOLD not in segment else
                               {k: v for k, v in segment.items() if k != PUBLICATION_THRESHOLD})
    if isinstance(segment, StoredSegment):
        segment._evidence_digest = signature
    return signature
