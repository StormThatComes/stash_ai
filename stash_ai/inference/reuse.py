"""Prove saved inference identity before applying threshold-only edits."""
from .input_identity import cue_manifest
from stash_ai.rethreshold.plans import ThresholdRequest


def reuse(app, store, scene, config, identity, cues, hashes, kinds, *, diagnostics=None):
    """Prove and rethreshold the selected current outputs through one planner."""
    from stash_ai.storage.progress import completed
    from stash_ai.rethreshold.scenes import plan, apply
    from stash_ai.evidence.thresholds import indexed_signature
    records = {kind: store.model_record(scene, kind, config.source.key, borrow=True) for kind in kinds}
    if diagnostics is not None:
        diagnostics['metas'] = {kind: record['meta'] for kind, record in records.items() if record}
    if any(not record for record in records.values()):
        return False
    from .provenance import saved_identity, predictive
    manifest = cue_manifest(cues)
    from stash_ai.evidence.retention import scene_cache_floor, marker_cache_floor
    for kind, record in records.items():
        meta = record['meta']
        signature = meta['inference_signature']
        if not completed(store, scene, config.source.key, [kind], indexed_signature(record, signature)):
            return False
        candidate = saved_identity(store, scene, config.source.key, signature, identity, manifest, hashes, meta)
        if candidate is None or predictive(candidate) != predictive(identity):
            return False
        required = (scene_cache_floor(config, app.TEMPORAL_POLICY) if kind == 'scene'
                    else marker_cache_floor(config, app.TEMPORAL_POLICY))
        if meta['scene_score_floor' if kind == 'scene' else 'marker_score_floor'] > required:
            if diagnostics is not None:
                diagnostics['rethreshold_rejected'] = True
            return False
    try:
        outputs = plan(app, store, scene, config, request=ThresholdRequest(kinds=tuple(records)))
    except ValueError:
        if diagnostics is not None:
            diagnostics['rethreshold_rejected'] = True
        return False
    apply(store, scene, config.source.key, outputs)
    return True
