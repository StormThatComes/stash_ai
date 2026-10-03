"""Plan scene and optional marker threshold changes from retained evidence."""
from stash_ai.scene_cache import SCENE_OVERRIDE, scene_threshold, validate_scene


def plan(app, store, scene, config, *, scenes=True, markers=False, direct=False,
         marker_check_scene=None, pending_checked=False, metadata_only=False):
    from stash_ai.thresholds import OVERRIDE
    from .markers import validate
    result = {}
    if scenes:
        if metadata_only:
            segments = store.result_metadata(scene, 'scene', config.source.key)
        elif direct:
            segments = store.model_records(scene, 'scene', config.source.key, borrow=True)
        else:
            segments = store.records(scene, 'scene', config.source.key, borrow=True).get(config.source.key, [])
        if not segments:
            raise ValueError('no scene results; run inference for this scene/model.')
        for segment in segments:
            validate_scene(app, segment, config.scene_threshold,
                           has_frames=segment['has_frames'] if metadata_only else None)
        result['scene'] = segments
    if markers:
        check_scene = not scenes if marker_check_scene is None else marker_check_scene
        result['markers'] = validate(app, store, scene, config, check_scene=check_scene, direct=direct,
                                     metadata_only=metadata_only)
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
    if updated and not pending_checked and store.has_pending_publication(scene):
        raise RuntimeError('Finish the pending merge or use --abandon-pending before changing thresholds.')
    return updated


def apply(store, scene, model, outputs, *, metadata_only=False):
    from stash_ai.thresholds import marker_threshold
    with store.connection:
        for kind, segments in outputs.items():
            threshold = scene_threshold(segments[0]) if kind == 'scene' else marker_threshold(segments[0])
            store.set_result_threshold(scene, model, kind, threshold, segments,
                                       metadata_only=metadata_only)
