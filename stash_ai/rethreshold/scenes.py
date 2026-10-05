"""Plan scene and optional marker threshold changes from retained evidence."""
from stash_ai.storage.thresholds import PUBLICATION_THRESHOLD, effective
from stash_ai.evidence.cache_views import validate_scene
from .plans import RecordAccess, PendingPolicy, ThresholdRequest, ThresholdPlan, apply_plan


def plan(app, store, scene, config, *, request=ThresholdRequest()) -> ThresholdPlan:
    from .markers import validate
    result = {}
    scenes = 'scene' in request.kinds
    if scenes:
        record = request.access.read(store, scene, 'scene', config.source.key)
        if not record:
            raise ValueError('no scene results; run inference for this scene/model.')
        validate_scene(app, record, config.scene_threshold,
                       has_frames=record['has_frames'] if request.access is RecordAccess.METADATA else None)
        result['scene'] = record
    if 'markers' in request.kinds:
        check_scene = not scenes if request.marker_check_scene is None else request.marker_check_scene
        result['markers'] = validate(app, store, scene, config, check_scene=check_scene, access=request.access)
    updated = {}
    for kind, record in result.items():
        threshold = config.scene_threshold if kind == 'scene' else config.marker_threshold
        if threshold != effective(record):
            value = dict(record)
            value.pop(PUBLICATION_THRESHOLD, None)
            meta_key = 'scene_threshold' if kind == 'scene' else 'marker_threshold'
            if threshold != record['meta'][meta_key]:
                value[PUBLICATION_THRESHOLD] = threshold
            updated[kind] = value
    if updated and request.pending is PendingPolicy.VERIFY and store.has_pending_publication(scene):
        raise RuntimeError('Finish the pending merge or use --abandon-pending before changing thresholds.')
    return ThresholdPlan(updated, request.access)



def apply(store, scene, model, prepared: ThresholdPlan):
    apply_plan(store, scene, model, prepared)
