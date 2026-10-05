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
    if OVERRIDE in segment and segment['mode'] not in ('frames', 'spans'):
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
        for segments in records.values() for segment in segments if OVERRIDE in segment}
    return context


def marker_threshold(segment, thresholds=None):
    from stash_ai.evidence.identity import marker_segment_signature
    if thresholds is not None:
        return thresholds.get(marker_segment_signature(segment), segment['meta'].get('marker_threshold'))
    return segment.get(OVERRIDE, segment['meta'].get('marker_threshold'))


def _score(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def validate_marker_blocklist(segment, identity, *, allow_unknown=False,
                              frozen_exclusions=frozenset()):
    """Guard coverage lost to the retired option; cleanup can replay frozen bans."""
    if segment['mode'] != 'frames' or segment['meta'].get('origin') != 'inference':
        return
    from stash_ai.evidence.formats import current, validate_current
    if current(segment['meta']):
        validate_current(segment)
        return
    if allow_unknown and 'marker_tag_blocklist' not in segment['meta']:
        return
    retained = segment['meta'].get('marker_tag_blocklist')
    if not isinstance(retained, list) or any(not isinstance(tag, str) for tag in retained):
        raise ValueError(f'{identity}: marker blocklist metadata is missing or invalid; '
                         'run inference with --restart for this scene/model.')
    unblocked = sorted(set(retained) - frozen_exclusions)
    if unblocked:
        raise ValueError(f'{identity}: unblocked marker tags {unblocked!r} are absent from cached frames; '
                         'run inference with --restart for this scene/model.')


def validate_marker_cache(app, records, *, allow_unknown_blocklist=False,
                          allow_unknown_retention=False, frozen_exclusions=None):
    """Check retained coverage before cached frames can build markers or votes."""
    from stash_ai.evidence.temporal.policy import marker_floor
    for model, segments in records.items():
        for segment in segments:
            meta = segment['meta']
            validate_marker_blocklist(segment, model, allow_unknown=allow_unknown_blocklist,
                                      frozen_exclusions=frozen_exclusions[model] if frozen_exclusions is not None else frozenset())
            # Imported frames have no inference retention contract. Cleanup may
            # also revoke verified published proofs with unknown old retention.
            if segment['mode'] != 'frames':
                continue
            if 'marker_score_floor' not in meta and (
                    meta.get('origin') != 'inference' or allow_unknown_retention):
                continue
            threshold = marker_threshold(segment, app.MARKER_THRESHOLDS)
            if not _score(threshold):
                raise ValueError(f'{model}: invalid cached marker threshold; run inference.')
            needed = marker_floor(app.TEMPORAL_POLICY, threshold)
            floor = meta.get('marker_score_floor')
            if not _score(floor) or floor > needed:
                raise ValueError(f'{model}: marker cache floor {floor!r} exceeds required {needed:g}; '
                                 'run inference with --restart for this scene/model.')
