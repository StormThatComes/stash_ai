"""Plan scene and optional marker threshold changes from retained evidence."""
from stash_ai.evidence.identity import SCENE_OVERRIDE
from stash_ai.evidence.cache_views import validate_scene
from .plans import RecordAccess, PendingPolicy, ThresholdRequest, ThresholdPlan, apply_plan


def plan(app, store, scene, config, *, request=ThresholdRequest()) -> ThresholdPlan:
    from stash_ai.evidence.identity import OVERRIDE
    from .markers import validate
    result = {}
    scenes = 'scene' in request.kinds
    if scenes:
        segments = request.access.read(store, scene, 'scene', config.source.key)
        if not segments:
            raise ValueError('no scene results; run inference for this scene/model.')
        for segment in segments:
            validate_scene(app, segment, config.scene_threshold,
                           has_frames=segment['has_frames'] if request.access is RecordAccess.METADATA else None)
        result['scene'] = segments
    if 'markers' in request.kinds:
        check_scene = not scenes if request.marker_check_scene is None else request.marker_check_scene
        result['markers'] = validate(app, store, scene, config, check_scene=check_scene, access=request.access)
    updated = {}
    for kind, segments in result.items():
        key, meta_key, threshold = ((SCENE_OVERRIDE, 'scene_threshold', config.scene_threshold) if kind == 'scene'
                                   else (OVERRIDE, 'marker_threshold', config.marker_threshold))
        values = []
        for segment in segments:
            value = dict(segment)
            value.pop(key, None)
            if threshold != segment['meta'][meta_key]:
                value[key] = threshold
            values.append(value)
        if values != segments:
            updated[kind] = values
    if updated and request.pending is PendingPolicy.VERIFY and store.has_pending_publication(scene):
        raise RuntimeError('Finish the pending merge or use --abandon-pending before changing thresholds.')
    return ThresholdPlan(updated, request.access)



def apply(store, scene, model, prepared: ThresholdPlan):
    apply_plan(store, scene, model, prepared)
