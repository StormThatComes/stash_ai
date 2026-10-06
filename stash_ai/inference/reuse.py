"""Prove saved inference identity before applying threshold-only edits."""
from .input_identity import cue_manifest, inference_signature
from stash_ai.rethreshold.plans import ThresholdRequest


def reuse(app, store, scene, config, identity, cues, hashes, kinds):
    """Prove and rethreshold the selected current outputs through one planner."""
    from stash_ai.storage.progress import completed
    from stash_ai.rethreshold.scenes import plan, apply
    from stash_ai.evidence.thresholds import indexed_signature
    records = {kind: store.model_record(scene, kind, config.source.key, borrow=True) for kind in kinds}
    if any(not record for record in records.values()):
        return False
    for kind, record in records.items():
        meta = record['meta']
        signature = inference_signature({**identity, 'scene_threshold': meta['scene_threshold'],
                                         'marker_threshold': meta['marker_threshold']},
                                        cue_manifest(cues), hashes)
        if not completed(store, scene, config.source.key, [kind], indexed_signature(record, signature)):
            return False
    try:
        outputs = plan(app, store, scene, config, request=ThresholdRequest(kinds=tuple(records)))
    except ValueError:
        return False
    apply(store, scene, config.source.key, outputs)
    return True

