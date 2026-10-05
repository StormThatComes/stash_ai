"""Marker publication thresholds over immutable, bounded inference evidence."""

import math
from stash_ai.storage.thresholds import PUBLICATION_THRESHOLD, effective


def indexed_signature(segment, inference=None):
    from stash_ai.signatures import digest
    signature = segment['meta'].get('inference_signature') if inference is None else inference
    if PUBLICATION_THRESHOLD not in segment:
        return signature
    value = segment[PUBLICATION_THRESHOLD]
    return digest([signature, {'scene_threshold': value} if segment['mode'] == 'scene' else value])


def marker_threshold(segment):
    return effective(segment)


def _score(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def validate_marker_cache(app, records):
    """Current inference must retain all observations needed by the frozen policy."""
    from stash_ai.evidence.temporal.policy import marker_floor
    from stash_ai.evidence.formats import validate_current
    for model, segment in records.items():
        validate_current(segment)
        if segment['mode'] != 'frames':
            continue
        meta = segment['meta']
        threshold = marker_threshold(segment)
        if not _score(threshold):
            raise ValueError(f'{model}: invalid cached marker threshold; run inference.')
        needed = marker_floor(app.TEMPORAL_POLICY, threshold)
        floor = meta['marker_score_floor']
        if not _score(floor) or floor > needed:
            raise ValueError(f'{model}: marker cache floor {floor!r} exceeds required {needed:g}; '
                             'run inference with --restart for this scene/model.')
