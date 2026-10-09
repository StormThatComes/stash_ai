"""Prove saved inference identity before applying threshold-only edits."""
import zlib
from .input_identity import cue_manifest
from stash_ai.rethreshold.plans import RecordAccess, ThresholdRequest, ThresholdPlan


from .decisions import ReuseDecision


def CachedReuse(thresholds, frame_input, rethreshold):
    """Adapter for the historical three-field constructor."""
    from .decisions import ReuseAction
    return ReuseDecision(ReuseAction.RETHRESHOLD if rethreshold else ReuseAction.REUSE,
                         thresholds, frame_input)


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
        return ReuseDecision.authorized(thresholds, inputs.pop(), input_verified=False)
    except (ValueError, KeyError, TypeError, zlib.error):
        return None  # Verify current inputs through the established normal path.


def verified_reuse_decision(app, store, scene, config, identity, cues, hashes, kinds, *, diagnostics=None, input_options=None, snapshot=None):
    """Prove and rethreshold the selected current outputs through one planner."""
    from stash_ai.storage.progress import completed
    from stash_ai.rethreshold.scenes import plan
    from stash_ai.evidence.thresholds import indexed_signature
    records = {kind: store.model_record(scene, kind, config.source.key, borrow=True) for kind in kinds}
    if diagnostics is not None:
        diagnostics['metas'] = {kind: record['meta'] for kind, record in records.items() if record}
    if any(not record for record in records.values()):
        return ReuseDecision.infer('missing_or_incompatible_evidence')
    from .provenance import saved_identity, predictive
    manifest = snapshot.manifest if snapshot is not None else cue_manifest(cues)
    from stash_ai.evidence.retention import scene_cache_floor, marker_cache_floor
    verified = {}
    for kind, record in records.items():
        meta = record['meta']
        signature = meta['inference_signature']
        if not completed(store, scene, config.source.key, [kind], indexed_signature(record, signature)):
            return ReuseDecision.infer('saved_identity_mismatch')
        candidate = saved_identity(store, scene, config.source.key, signature, identity, manifest, hashes, meta)
        if candidate is None or predictive(candidate) != predictive(identity):
            return ReuseDecision.infer('saved_identity_mismatch')
        verified[signature] = candidate
        required = (scene_cache_floor(config, app.TEMPORAL_POLICY) if kind == 'scene'
                    else marker_cache_floor(config, app.TEMPORAL_POLICY))
        if meta['scene_score_floor' if kind == 'scene' else 'marker_score_floor'] > required:
            if diagnostics is not None:
                diagnostics['rethreshold_rejected'] = True
            return ReuseDecision.infer('saved_identity_mismatch')
    try:
        outputs = plan(app, store, scene, config, request=ThresholdRequest(kinds=tuple(records)))
    except ValueError:
        if diagnostics is not None:
            diagnostics['rethreshold_rejected'] = True
        return ReuseDecision.infer('missing_or_incompatible_evidence')
    from .identity_types import ProvenanceIdentity
    return ReuseDecision.authorized(outputs, records[next(iter(records))]['meta']['frame_input_signature'],
        input_verified=True, identities=tuple((signature, ProvenanceIdentity.from_legacy(candidate))
                                             for signature, candidate in verified.items()))


def reuse(app, store, scene, config, identity, cues, hashes, kinds, *, diagnostics=None, input_options=None, snapshot=None):
    """Legacy bool adapter; only the consumer applies the authorized plan."""
    from stash_ai.rethreshold.scenes import apply
    from .provenance import register
    decision = verified_reuse_decision(app, store, scene, config, identity, cues, hashes, kinds,
                                       diagnostics=diagnostics, snapshot=snapshot)
    if diagnostics is not None:
        diagnostics['decision'] = decision
        diagnostics['rethreshold'] = decision.rethreshold
    if not decision.reusable:
        return False
    apply(store, scene, config.source.key, decision.thresholds)
    if input_options is not None:
        # Preserve the proven identity associated with each historical signature.
        for signature, candidate in decision.identities:
            register(store, scene, config.source.key, signature, candidate.legacy(), input_options=input_options)
    return True
