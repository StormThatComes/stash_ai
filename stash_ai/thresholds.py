"""Marker publication thresholds over immutable, bounded inference evidence."""

import math
from types import SimpleNamespace

OVERRIDE = 'marker_threshold_override'


def indexed_signature(segment, inference=None):
    from stash_ai.signatures import digest
    signature = segment['meta'].get('inference_signature') if inference is None else inference
    from stash_ai.scene_cache import SCENE_OVERRIDE
    if SCENE_OVERRIDE in segment and OVERRIDE in segment:
        return digest([signature, {
            'scene_threshold': segment[SCENE_OVERRIDE],
            'marker_threshold': segment[OVERRIDE],
        }])
    if SCENE_OVERRIDE in segment:
        return digest([signature, {'scene_threshold': segment[SCENE_OVERRIDE]}])
    return digest([signature, segment[OVERRIDE]]) if OVERRIDE in segment else signature


def marker_context(app, records):
    """Freeze effective thresholds with each publication, including cleanup replay."""
    if hasattr(app, 'MARKER_THRESHOLDS'):
        return app
    from stash_ai.merge import marker_segment_signature
    context = SimpleNamespace(**vars(app))
    context.MARKER_THRESHOLDS = {
        marker_segment_signature(segment): segment[OVERRIDE]
        for segments in records.values() for segment in segments if OVERRIDE in segment}
    return context


def marker_threshold(segment, thresholds=None):
    from stash_ai.merge import marker_segment_signature
    if thresholds is not None:
        return thresholds.get(marker_segment_signature(segment), segment['meta'].get('marker_threshold'))
    return segment.get(OVERRIDE, segment['meta'].get('marker_threshold'))


def _score(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def validate_marker_blocklist(segment, identity, *, allow_unknown=False,
                              legacy_blocklist=frozenset()):
    """Guard coverage lost to the retired option; cleanup can replay frozen bans."""
    if segment['mode'] != 'frames' or segment['meta'].get('origin') != 'inference':
        return
    if allow_unknown and 'marker_tag_blocklist' not in segment['meta']:
        return
    retained = segment['meta'].get('marker_tag_blocklist')
    if not isinstance(retained, list) or any(not isinstance(tag, str) for tag in retained):
        raise ValueError(f'{identity}: marker blocklist metadata is missing or invalid; '
                         'run inference with --restart for this scene/model.')
    unblocked = sorted(set(retained) - legacy_blocklist)
    if unblocked:
        raise ValueError(f'{identity}: unblocked marker tags {unblocked!r} are absent from cached frames; '
                         'run inference with --restart for this scene/model.')


def validate_marker_cache(app, records, *, allow_unknown_blocklist=False,
                          allow_unknown_retention=False, legacy_blocklist=frozenset()):
    """Check retained coverage before cached frames can build markers or votes."""
    from stash_ai.evidence.temporal.policy import marker_floor
    for model, segments in records.items():
        for segment in segments:
            meta = segment['meta']
            validate_marker_blocklist(segment, model, allow_unknown=allow_unknown_blocklist,
                                      legacy_blocklist=legacy_blocklist)
            # Imported frames have no inference retention contract. Cleanup may
            # also revoke verified published proofs with unknown old retention.
            if segment['mode'] != 'frames':
                continue
            if 'marker_score_floor' not in meta and (
                    meta.get('origin') != 'inference' or allow_unknown_retention):
                continue
            threshold = marker_threshold(segment, getattr(app, 'MARKER_THRESHOLDS', None))
            if not _score(threshold):
                raise ValueError(f'{model}: invalid cached marker threshold; run inference.')
            needed = marker_floor(app.TEMPORAL_POLICY, threshold)
            # Before retention metadata existed, only high detections may have
            # survived. Overrides change publication, not the capture floor.
            floor = meta.get('marker_score_floor', meta.get('marker_threshold'))
            if not _score(floor) or floor > needed:
                raise ValueError(f'{model}: marker cache floor {floor!r} exceeds required {needed:g}; '
                                 'run inference with --restart for this scene/model.')


def reuse(app, store, scene, config, identity, cues, hashes, kinds):
    """Prove the original inference matches, allowing only a marker threshold edit."""
    from stash_ai.storage.progress import completed
    from stash_ai.rethreshold.markers import validate, apply
    from stash_ai.signatures import digest
    from stash_ai.scene_cache import reuse as reuse_scenes
    if reuse_scenes(app, store, scene, config, identity, cues, hashes, kinds):
        return True
    if not app.CREATE_MARKERS:
        return False
    try:
        segments = validate(app, store, scene, config)
    except ValueError:
        return False
    original = segments[-1]['meta'].get('marker_threshold')
    signature = digest({'model': {**identity, 'marker_threshold': original},
        'cues': [[c.start, c.end, c.image, list(c.rect)] for c in cues], 'media': hashes})
    for kind in kinds:
        expected = indexed_signature(segments[-1], signature) if kind == 'markers' else signature
        if not completed(store, scene, config.source.key, [kind], expected):
            return False
    apply(app, store, scene, config, segments)
    return True
