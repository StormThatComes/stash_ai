"""Validate marker coverage and update its effective publication threshold."""
from stash_ai.evidence.identity import OVERRIDE
from stash_ai.evidence.thresholds import _score, marker_threshold, validate_marker_blocklist


from .plans import RecordAccess, PendingPolicy, ThresholdPlan, apply_plan


def validate(app, store, scene, config, *, check_scene=True, access=RecordAccess.BATCH):
    """Reject incomplete evidence; a missing detection is never treated as zero."""
    from stash_ai.evidence.temporal.policy import marker_floor
    model = config.source.key
    segments = access.read(store, scene, 'markers', model)
    if not segments:
        raise ValueError(f'{scene}/{model}: no cached marker frames; run inference for this scene/model.')
    from stash_ai.evidence.cache_views import scene_threshold
    scene_segments = (access.read(store, scene, 'scene', model)
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



def plan(app, store, scene, config, segments=None, *, access=RecordAccess.BATCH,
         pending=PendingPolicy.VERIFY):
    segments = validate(app, store, scene, config, access=access) if segments is None else segments
    if all(marker_threshold(s) == config.marker_threshold for s in segments):
        return ThresholdPlan({}, access)
    if pending is PendingPolicy.VERIFY and store.has_pending_publication(scene):
        raise RuntimeError(f'{scene}: finish the pending merge or use --abandon-pending before changing thresholds.')
    updated = []
    for segment in segments:
        value = dict(segment)
        value.pop(OVERRIDE, None)
        if config.marker_threshold != segment['meta']['marker_threshold']:
            value[OVERRIDE] = config.marker_threshold
        updated.append(value)
    return ThresholdPlan({'markers': updated}, access)


def apply(app, store, scene, config, segments=None):
    """Validate and apply a marker threshold during inference evidence reuse."""
    prepared = plan(app, store, scene, config, segments)
    apply_plan(store, scene, config.source.key, prepared)
    return bool(prepared)
