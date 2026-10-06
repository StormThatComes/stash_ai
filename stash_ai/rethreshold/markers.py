"""Validate marker coverage and update its effective publication threshold."""
from stash_ai.evidence.formats import validate_current
from stash_ai.evidence.thresholds import _score


from .plans import RecordAccess


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
