"""Marker publication thresholds over immutable, bounded inference evidence."""

import math
from types import SimpleNamespace

OVERRIDE = 'marker_threshold_override'


def indexed_signature(segment, inference=None):
    from stash_progress import digest
    signature = segment['meta'].get('inference_signature') if inference is None else inference
    from stash_scene_cache import SCENE_OVERRIDE
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
    from stash_merge import marker_segment_signature
    context = SimpleNamespace(**vars(app))
    context.MARKER_THRESHOLDS = {
        marker_segment_signature(segment): segment[OVERRIDE]
        for segments in records.values() for segment in segments if OVERRIDE in segment}
    return context


def marker_threshold(segment, thresholds=None):
    from stash_merge import marker_segment_signature
    if thresholds is not None:
        return thresholds.get(marker_segment_signature(segment), segment['meta'].get('marker_threshold'))
    return segment.get(OVERRIDE, segment['meta'].get('marker_threshold'))


def _score(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def validate(app, store, scene, config, *, check_scene=True):
    """Reject incomplete evidence; a missing detection is never treated as zero."""
    from stash_temporal import marker_floor
    model = config.source.key
    segments = store.records(scene, 'markers', model, borrow=True).get(model, [])
    if not segments:
        raise ValueError(f'{scene}/{model}: no cached marker frames; run inference for this scene/model.')
    from stash_scene_cache import scene_threshold
    scene_segments = store.records(scene, 'scene', model, borrow=True).get(model, [])
    for segment in scene_segments:
        if check_scene and segment['meta'].get('origin') == 'inference' and scene_threshold(segment) != config.scene_threshold:
            raise ValueError(f'{scene}/{model}: scene_threshold changed; scene statistics require inference.')
    needed = marker_floor(app.TEMPORAL_POLICY, config.marker_threshold)
    for segment in segments:
        meta = segment['meta']
        if segment['mode'] != 'frames' or meta.get('origin') != 'inference':
            raise ValueError(f'{scene}/{model}: imported ranges cannot be rethresholded; run inference.')
        # Before temporal retention was introduced, only high detections survived.
        floor = meta.get('marker_score_floor', meta.get('marker_threshold'))
        if not _score(floor) or floor > needed:
            raise ValueError(f'{scene}/{model}: cache floor {floor!r} exceeds required {needed:g}; run inference.')
        if check_scene and not scene_segments and meta.get('scene_threshold') != config.scene_threshold:
            raise ValueError(f'{scene}/{model}: scene_threshold changed; scene statistics require inference.')
    return segments


def apply(app, store, scene, config, segments=None):
    """Store policy separately; retain every original frame, score and signature."""
    segments = validate(app, store, scene, config) if segments is None else segments
    if all(marker_threshold(s) == config.marker_threshold for s in segments):
        return False
    if any(store.snapshot(scene, kind, borrow=True)[1] is not None for kind in ('scene', 'markers')):
        raise RuntimeError(f'{scene}: finish the pending merge or use --abandon-pending before changing thresholds.')
    updated = []
    for segment in segments:
        value = dict(segment)
        value.pop(OVERRIDE, None)
        if config.marker_threshold != segment['meta']['marker_threshold']:
            value[OVERRIDE] = config.marker_threshold
        updated.append(value)
    with store.connection:
        store.replace_records(scene, config.source.key, 'markers', updated)
    return True


def reuse(app, store, scene, config, identity, cues, hashes, kinds):
    """Prove the original inference matches, allowing only a marker threshold edit."""
    from stash_progress import completed, digest
    from stash_scene_cache import reuse as reuse_scenes
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


def rethreshold(app, store, scenes, settings, report):
    """Preflight the whole selection before changing policy or publishing to Stash."""
    errors = 0
    for scene in scenes:
        for config in settings:
            try:
                validate(app, store, scene, config)
                if any(store.snapshot(scene, kind, borrow=True)[1] is not None for kind in ('scene', 'markers')):
                    raise RuntimeError('Finish the pending merge or use --abandon-pending first.')
            except (ValueError, RuntimeError) as exc:
                report.issue('marker thresholds', config.source.key, scene, exc, str(exc))
                errors += 1
    if errors:
        return dict(errors=errors)
    changed = 0
    for scene in scenes:
        for config in settings:
            changed += apply(app, store, scene, config)
    return dict(marker_thresholds_updated=changed)
