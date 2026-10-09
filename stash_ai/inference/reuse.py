"""Prove saved inference identity before applying threshold-only edits."""
from dataclasses import dataclass
import zlib
from .input_identity import cue_manifest
from stash_ai.rethreshold.plans import RecordAccess, ThresholdRequest, ThresholdPlan


@dataclass(frozen=True)
class CachedReuse:
    thresholds: ThresholdPlan
    frame_input: str
    rethreshold: bool


def input_options(app, paths):
    from stash_ai.signatures import digest
    return dict(paths=digest(paths), sample_limit=app.MAX_FRAMES_PER_SCENE)


def cached_reuse(app, store, scene, config, identity, paths, kinds):
    """Trust previously verified inputs, checking model and durable completion.

    The coordinator enables this per completed model with consistent saved inputs.
    Unfinished passes verify inputs and invalidate cached peers if inputs changed.
    No current media-content claim is made by this non-fetching path.
    """
    from stash_ai.storage.progress import completed
    from stash_ai.evidence.thresholds import indexed_signature
    from stash_ai.evidence.retention import scene_cache_floor, marker_cache_floor
    from stash_ai.rethreshold.scenes import plan
    from .provenance import registered_identity, predictive
    if not kinds or store.connection.execute('SELECT 1 FROM ' + store.table('checkpoints') +
            ' WHERE scope=? AND scene=? AND model=?',
            (store.scope, str(scene), config.source.key)).fetchone():
        return None
    try:
        inputs, identities = set(), {}
        for kind in kinds:
            record = store.result_metadata(scene, kind, config.source.key)
            if record is None:
                return None
            meta = record['meta']
            signature = meta['inference_signature']
            if not completed(store, scene, config.source.key, [kind], indexed_signature(record)):
                return None
            if signature not in identities:
                identities[signature] = registered_identity(store, scene, config.source.key, signature)
            candidate, options = identities[signature]
            if candidate is None or predictive(candidate) != predictive(identity):
                return None
            if options is not None and options != input_options(app, paths):
                return None
            # Old registry rows have no sampling/path context. The normal full
            # sampling default can trust their saved input; changed limits verify.
            if options is None and app.MAX_FRAMES_PER_SCENE:
                return None
            required = (scene_cache_floor(config, app.TEMPORAL_POLICY) if kind == 'scene'
                        else marker_cache_floor(config, app.TEMPORAL_POLICY))
            if meta['scene_score_floor' if kind == 'scene' else 'marker_score_floor'] > required:
                return None
            inputs.add(meta['frame_input_signature'])
        if len(inputs) != 1:
            return None
        try:
            thresholds = plan(app, store, scene, config,
                              request=ThresholdRequest(kinds=tuple(kinds), access=RecordAccess.METADATA))
        except RuntimeError:
            # Pending publication is reported by the ordinary per-scene path.
            return None
        return CachedReuse(thresholds, inputs.pop(), bool(thresholds))
    except (ValueError, KeyError, TypeError, zlib.error):
        return None  # Verify current inputs through the established normal path.


def reuse(app, store, scene, config, identity, cues, hashes, kinds, *, diagnostics=None, input_options=None):
    """Prove and rethreshold the selected current outputs through one planner."""
    from stash_ai.storage.progress import completed
    from stash_ai.rethreshold.scenes import plan, apply
    from stash_ai.evidence.thresholds import indexed_signature
    records = {kind: store.model_record(scene, kind, config.source.key, borrow=True) for kind in kinds}
    if diagnostics is not None:
        diagnostics['metas'] = {kind: record['meta'] for kind, record in records.items() if record}
    if any(not record for record in records.values()):
        return False
    from .provenance import saved_identity, predictive, register
    manifest = cue_manifest(cues)
    from stash_ai.evidence.retention import scene_cache_floor, marker_cache_floor
    verified = {}
    for kind, record in records.items():
        meta = record['meta']
        signature = meta['inference_signature']
        if not completed(store, scene, config.source.key, [kind], indexed_signature(record, signature)):
            return False
        candidate = saved_identity(store, scene, config.source.key, signature, identity, manifest, hashes, meta)
        if candidate is None or predictive(candidate) != predictive(identity):
            return False
        verified[signature] = candidate
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
    if diagnostics is not None:
        diagnostics['rethreshold'] = bool(outputs)
    if input_options is not None:
        for signature, candidate in verified.items():
            register(store, scene, config.source.key, signature, candidate, input_options=input_options)
    return True
