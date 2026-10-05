"""Marker publication thresholds over immutable, bounded inference evidence."""

import math
from dataclasses import replace

from stash_ai.evidence.identity import OVERRIDE


def indexed_signature(segment, inference=None):
    from stash_ai.signatures import digest
    signature = segment['meta'].get('inference_signature') if inference is None else inference
    from stash_ai.evidence.identity import SCENE_OVERRIDE
    if SCENE_OVERRIDE in segment:
        if segment['mode'] != 'scene' or OVERRIDE in segment:
            raise ValueError('Threshold override does not match result kind.')
        return digest([signature, {'scene_threshold': segment[SCENE_OVERRIDE]}])
    if OVERRIDE in segment and segment['mode'] != 'frames':
        raise ValueError('Threshold override does not match result kind.')
    return digest([signature, segment[OVERRIDE]]) if OVERRIDE in segment else signature


def marker_context(app, records):
    """Freeze effective thresholds with each publication, including cleanup replay."""
    if app.MARKER_THRESHOLDS is not None:
        return app
    from stash_ai.evidence.identity import marker_segment_signature
    context = replace(app)
    context.MARKER_THRESHOLDS = {
        marker_segment_signature(segment): segment[OVERRIDE]
        for segment in records.values() if OVERRIDE in segment}
    return context


def marker_threshold(segment, thresholds=None):
    from stash_ai.evidence.identity import marker_segment_signature
    if thresholds is not None:
        return thresholds.get(marker_segment_signature(segment), segment['meta'].get('marker_threshold'))
    return segment.get(OVERRIDE, segment['meta'].get('marker_threshold'))


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
        threshold = marker_threshold(segment, app.MARKER_THRESHOLDS)
        if not _score(threshold):
            raise ValueError(f'{model}: invalid cached marker threshold; run inference.')
        needed = marker_floor(app.TEMPORAL_POLICY, threshold)
        floor = meta['marker_score_floor']
        if not _score(floor) or floor > needed:
            raise ValueError(f'{model}: marker cache floor {floor!r} exceeds required {needed:g}; '
                             'run inference with --restart for this scene/model.')
