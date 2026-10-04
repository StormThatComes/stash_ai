"""Prove saved inference identity before applying threshold-only edits."""
from .input_identity import cue_manifest, inference_signature


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
        segments = validate(app, store, scene, config)
    except ValueError:
        return False
    original = segments[-1]['meta'].get('marker_threshold')
    signature = inference_signature({**identity, 'marker_threshold': original}, cue_manifest(cues), hashes)
    for kind in kinds:
        expected = indexed_signature(segments[-1], signature) if kind == 'markers' else signature
        if not completed(store, scene, config.source.key, [kind], expected):
            return False
    apply(app, store, scene, config, segments)
    return True


def reuse_scenes(app, store, scene, config, identity, cues, hashes, kinds):
    from stash_ai.storage.progress import completed
    from stash_ai.rethreshold.scenes import plan, apply
    from stash_ai.evidence.thresholds import indexed_signature
    records = {kind: store.records(scene, kind, config.source.key, borrow=True).get(config.source.key, []) for kind in kinds}
    if any(not segments for segments in records.values()):
        return False
    for kind, segments in records.items():
        segment = segments[-1]
        meta = segment['meta']
        signature = inference_signature({**identity, 'scene_threshold': meta.get('scene_threshold'),
                                         'marker_threshold': meta.get('marker_threshold')},
                                        cue_manifest(cues), hashes)
        if not completed(store, scene, config.source.key, [kind], indexed_signature(segment, signature)):
            return False
    if 'scene' not in records:
        return False
    try:
        outputs = plan(app, store, scene, config, markers='markers' in records)
    except ValueError:
        return False
    apply(store, scene, config.source.key, outputs)
    return True

