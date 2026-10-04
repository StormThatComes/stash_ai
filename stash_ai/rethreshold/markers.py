"""Validate marker coverage and update its effective publication threshold."""
from stash_ai.evidence.identity import OVERRIDE
from stash_ai.evidence.thresholds import _score, marker_threshold, validate_marker_blocklist


def _model_records(store, scene, kind, model, *, direct=False, metadata_only=False):
    if metadata_only:
        return store.result_metadata(scene, kind, model)
    if direct:
        return store.model_records(scene, kind, model, borrow=True)
    return store.records(scene, kind, model, borrow=True).get(model, [])


def validate(app, store, scene, config, *, check_scene=True, direct=False, metadata_only=False):
    """Reject incomplete evidence; a missing detection is never treated as zero."""
    from stash_ai.evidence.temporal.policy import marker_floor
    model = config.source.key
    segments = _model_records(store, scene, 'markers', model, direct=direct, metadata_only=metadata_only)
    if not segments:
        raise ValueError(f'{scene}/{model}: no cached marker frames; run inference for this scene/model.')
    from stash_ai.evidence.cache_views import scene_threshold
    scene_segments = (_model_records(store, scene, 'scene', model, direct=direct, metadata_only=metadata_only)
                      if check_scene else [])
    for segment in scene_segments:
        if segment['meta'].get('origin') == 'inference' and scene_threshold(segment) != config.scene_threshold:
            raise ValueError(f'{scene}/{model}: scene_threshold changed; scene statistics require inference.')
    needed = marker_floor(app.TEMPORAL_POLICY, config.marker_threshold)
    for segment in segments:
        meta = segment['meta']
        if segment['mode'] != 'frames' or meta.get('origin') != 'inference':
            raise ValueError(f'{scene}/{model}: imported ranges cannot be rethresholded; run inference.')
        validate_marker_blocklist(segment, f'{scene}/{model}')
        # Before temporal retention was introduced, only high detections survived.
        floor = meta.get('marker_score_floor', meta.get('marker_threshold'))
        if not _score(floor) or floor > needed:
            raise ValueError(f'{scene}/{model}: cache floor {floor!r} exceeds required {needed:g}; run inference.')
        if check_scene and not scene_segments and meta.get('scene_threshold') != config.scene_threshold:
            raise ValueError(f'{scene}/{model}: scene_threshold changed; scene statistics require inference.')
    return segments


def apply(app, store, scene, config, segments=None, *, direct=False, pending_checked=False,
          metadata_only=False):
    """Store policy separately; retain every original frame, score and signature."""
    segments = validate(app, store, scene, config, direct=direct,
                        metadata_only=metadata_only) if segments is None else segments
    if all(marker_threshold(s) == config.marker_threshold for s in segments):
        return False
    if not pending_checked and store.has_pending_publication(scene):
        raise RuntimeError(f'{scene}: finish the pending merge or use --abandon-pending before changing thresholds.')
    updated = []
    for segment in segments:
        value = dict(segment)
        value.pop(OVERRIDE, None)
        if config.marker_threshold != segment['meta']['marker_threshold']:
            value[OVERRIDE] = config.marker_threshold
        updated.append(value)
    with store.connection:
        store.set_result_threshold(scene, config.source.key, 'markers', config.marker_threshold, updated,
                                   metadata_only=metadata_only)
    return True
