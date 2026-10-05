"""Prove saved inference identity before applying threshold-only edits."""
from .input_identity import cue_manifest, inference_signature
from stash_ai.rethreshold.plans import ThresholdRequest


def reuse(app, store, scene, config, identity, cues, hashes, kinds):
    """Prove the original inference matches, allowing only a marker threshold edit."""
    from stash_ai.storage.progress import completed
    from stash_ai.evidence.thresholds import indexed_signature
    from stash_ai.rethreshold.markers import validate, apply
    if reuse_scenes(app, store, scene, config, identity, cues, hashes, kinds):
        return True
    if not app.CREATE_MARKERS:
        return False
    try:
        record = validate(app, store, scene, config)
    except ValueError:
        return False
    original = record['meta'].get('marker_threshold')
    signature = inference_signature({**identity, 'marker_threshold': original}, cue_manifest(cues), hashes)
    for kind in kinds:
        expected = indexed_signature(record, signature) if kind == 'markers' else signature
        if not completed(store, scene, config.source.key, [kind], expected):
            return False
    apply(app, store, scene, config, record)
    return True


def reuse_scenes(app, store, scene, config, identity, cues, hashes, kinds):
    from stash_ai.storage.progress import completed
    from stash_ai.rethreshold.scenes import plan, apply
    from stash_ai.evidence.thresholds import indexed_signature
    records = {kind: store.model_record(scene, kind, config.source.key, borrow=True) for kind in kinds}
    if any(not record for record in records.values()):
        return False
    for kind, record in records.items():
        meta = record['meta']
        signature = inference_signature({**identity, 'scene_threshold': meta.get('scene_threshold'),
                                         'marker_threshold': meta.get('marker_threshold')},
                                        cue_manifest(cues), hashes)
        if not completed(store, scene, config.source.key, [kind], indexed_signature(record, signature)):
            return False
    if 'scene' not in records:
        return False
    try:
        outputs = plan(app, store, scene, config, request=ThresholdRequest(kinds=tuple(records)))
    except ValueError:
        return False
    apply(store, scene, config.source.key, outputs)
    return True

