"""Validate marker coverage and update its effective publication threshold."""
from stash_ai.evidence.formats import validate_current
from stash_ai.storage.thresholds import PUBLICATION_THRESHOLD
from stash_ai.evidence.thresholds import _score, marker_threshold


from .plans import RecordAccess, PendingPolicy, ThresholdPlan, apply_plan


def validate(app, store, scene, config, *, check_scene=True, access=RecordAccess.BATCH):
    from stash_ai.evidence.temporal.policy import marker_floor
    from stash_ai.evidence.cache_views import scene_threshold
    model = config.source.key
    record = access.read(store, scene, 'markers', model)
    if record is None:
        raise ValueError(f'{scene}/{model}: no cached marker frames; run inference for this scene/model.')
    scene_record = access.read(store, scene, 'scene', model) if check_scene else None
    if scene_record is not None and scene_threshold(scene_record) != config.scene_threshold:
        raise ValueError(f'{scene}/{model}: scene_threshold changed; scene statistics require inference.')
    validate_current(record)
    needed = marker_floor(app.TEMPORAL_POLICY, config.marker_threshold)
    floor = record['meta']['marker_score_floor']
    if not _score(floor) or floor > needed:
        raise ValueError(f'{scene}/{model}: cache floor {floor!r} exceeds required {needed:g}; run inference.')
    if check_scene and scene_record is None and record['meta']['scene_threshold'] != config.scene_threshold:
        raise ValueError(f'{scene}/{model}: scene_threshold changed; scene statistics require inference.')
    return record


def plan(app, store, scene, config, record=None, *, access=RecordAccess.BATCH,
         pending=PendingPolicy.VERIFY):
    record = validate(app, store, scene, config, access=access) if record is None else record
    if marker_threshold(record) == config.marker_threshold:
        return ThresholdPlan({}, access)
    if pending is PendingPolicy.VERIFY and store.has_pending_publication(scene):
        raise RuntimeError(f'{scene}: finish the pending merge or use --abandon-pending before changing thresholds.')
    updated = dict(record)
    updated.pop(PUBLICATION_THRESHOLD, None)
    if config.marker_threshold != record['meta']['marker_threshold']:
        updated[PUBLICATION_THRESHOLD] = config.marker_threshold
    return ThresholdPlan({'markers': updated}, access)


def apply(app, store, scene, config, record=None):
    """Validate and apply a marker threshold during inference evidence reuse."""
    prepared = plan(app, store, scene, config, record)
    apply_plan(store, scene, config.source.key, prepared)
    return bool(prepared)
