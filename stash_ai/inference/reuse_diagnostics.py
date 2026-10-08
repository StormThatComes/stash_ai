"""Observe existing reuse decisions; proofs are optional and never authorize reuse."""
from copy import deepcopy
from functools import wraps
from pathlib import Path
from statistics import median

from stash_ai.signatures import digest
from stash_ai.stash_api.transport import absolute_stash_url
from .input_identity import cue_manifest, frame_signature, inference_signature


def observational(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        if not self.enabled:
            return
        try:
            return method(self, *args, **kwargs)
        except Exception as exc:
            self.report.detail('Reuse diagnostics unavailable: ' + type(exc).__name__)
    return wrapped


def proof(identity, cues, hashes, paths, app):
    model = {name: value for name, value in identity.items() if name != 'files'}
    assets = [dict(name=Path(path).name, location=digest(path), sha256=checksum) for path, checksum in identity['files']]
    manifest = cue_manifest(cues)
    vtt = absolute_stash_url(paths['vtt'], app.STASH_URL)
    spacing = [b.start - a.start for a, b in zip(cues, cues[1:])]
    return dict(model=model, assets=assets, cues=digest(manifest), cue_count=len(cues),
                median_cue_spacing_seconds=median(spacing) if spacing else None,
                frame_input=frame_signature(manifest, hashes),
                vtt=dict(location=digest(vtt), sha256=hashes.get(vtt)),
                sprites=[dict(location=digest(url), sha256=hashes.get(url)) for url in sorted({cue.image for cue in cues})])


MODEL_REASONS = {
    'format': 'prediction_format_changed', 'source': 'model_source_changed', 'repo': 'model_repository_changed',
    'revision': 'model_revision_changed', 'vocabulary': 'vocabulary_changed', 'batch_size': 'batch_size_changed',
    'scene_threshold': 'scene_threshold_changed', 'marker_threshold': 'marker_threshold_changed',
    'aggregation': 'aggregation_changed', 'top_k': 'aggregation_changed', 'retention': 'retention_changed',
    'loader': 'loader_changed', 'preprocessing': 'preprocessing_changed',
}


def differences(before, after):
    reasons = [reason for name, reason in MODEL_REASONS.items()
               if before['model'].get(name) != after['model'].get(name)]
    if [row['sha256'] for row in before['assets']] != [row['sha256'] for row in after['assets']]:
        reasons.append('model_assets_changed')
    if [row['location'] for row in before['assets']] != [row['location'] for row in after['assets']]:
        reasons.append('model_asset_location_changed')
    for name, reason in (('cues', 'vtt_cues_changed'), ('vtt', 'vtt_media_changed'), ('sprites', 'sprite_media_changed')):
        if before[name] != after[name]:
            reasons.append(reason)
    return reasons


class ReuseObserver:
    """One active scene per model pass; all historical detail streams to SQLite."""
    def __init__(self, app, args, store, config, report):
        self.app, self.args, self.store, self.config, self.report = app, args, store, config, report
        self.model = config.source.key
        self.scene, self.current, self.cues, self.hashes = None, None, None, None
        self.action, self.reasons, self.details = 'infer', [], {}

    @property
    def enabled(self):
        history = self.report.history
        return self.report.verbose or (history is not None and history.store is not None)

    def begin(self, scene):
        self.scene, self.current = str(scene), None
        self.checkpoint_signature = None
        self.action, self.reasons, self.details = 'infer', [], {}

    def journal(self, operation, *args):
        if self.report.history is not None:
            return self.report.history.call(operation, *args)

    def emit(self, status):
        self.report.reuse_decision(self.store.scope, self.scene, self.model, self.action, status,
                                  sorted(set(self.reasons)), self.details)

    @observational
    def inputs(self, identity, cues, hashes, paths):
        self.identity = identity
        self.cues, self.hashes = cue_manifest(cues), dict(hashes)
        self.current = proof(identity, cues, hashes, paths, self.app)

    @observational
    def blocked(self, reason):
        self.action, self.reasons = 'blocked', [reason]
        self.emit('blocked')

    def metas(self, kinds):
        return {kind: row['meta'] for kind in kinds
                if (row := self.store.result_metadata(self.scene, kind, self.model)) is not None}

    def compare(self, kind, meta):
        saved = self.journal('get_proof', self.store.scope, self.scene, self.model, kind, meta['inference_signature'])
        if saved is not None and self.current is not None:
            changes = {name: dict(before=saved['model'].get(name), after=self.current['model'].get(name))
                       for name in MODEL_REASONS if saved['model'].get(name) != self.current['model'].get(name)}
            for name in ('assets', 'cue_count', 'median_cue_spacing_seconds', 'cues', 'vtt', 'sprites'):
                if saved.get(name) != self.current.get(name):
                    changes[name] = dict(before=saved.get(name), after=self.current.get(name))
            self.details.setdefault('changes', {})[kind] = changes
            return differences(saved, self.current)
        result = ['historical_identity_unavailable']
        if self.current is not None:
            if meta.get('frame_input_signature') != self.current['frame_input']:
                result.append('input_changed')
            elif meta['inference_signature'] != inference_signature(self.identity, self.cues, self.hashes):
                result.append('model_or_configuration_changed')
        return result

    @observational
    def required(self, kinds, diagnostics):
        metas = diagnostics.get('metas') or self.metas(kinds)
        self.reasons = []
        if self.args.restart:
            self.reasons.append('explicit_restart')
        elif not self.args.resume:
            self.reasons.append('resume_disabled')
        self.details['outputs'] = {}
        from stash_ai.evidence.temporal.policy import marker_floor
        for kind in kinds:
            meta = metas.get(kind)
            if meta is None:
                self.reasons.append('missing_' + kind + '_evidence')
                continue
            self.reasons.extend(self.compare(kind, meta))
            refresh = self.store.connection.execute('SELECT 1 FROM ' + self.store.table('refresh_requests') +
                ' WHERE scope=? AND scene=? AND model=? AND kind=?', (self.store.scope, self.scene, self.model, kind)).fetchone()
            if refresh:
                self.reasons.append('refresh_pending')
            index = self.store.connection.execute('SELECT signature FROM ' + self.store.table('result_index') +
                ' WHERE scope=? AND scene=? AND model=? AND kind=?', (self.store.scope, self.scene, self.model, kind)).fetchone()
            if index is None or index[0] is None:
                self.reasons.append('completion_index_missing')
            threshold = self.config.scene_threshold if kind == 'scene' else self.config.marker_threshold
            needed = (threshold * self.app.TEMPORAL_POLICY.scene_support_ratio if kind == 'scene'
                      else marker_floor(self.app.TEMPORAL_POLICY, threshold))
            floor = meta.get('scene_score_floor' if kind == 'scene' else 'marker_score_floor')
            requested = getattr(self.config, 'scene_cache_floor' if kind == 'scene' else 'marker_cache_floor')
            required = min(needed, requested) if requested is not None else needed
            self.details['outputs'][kind] = dict(retained_floor=floor, required_support_floor=needed,
                requested_cache_floor=requested, required_retention_floor=required,
                stored_threshold=meta['scene_threshold' if kind == 'scene' else 'marker_threshold'], target_threshold=threshold)
            if floor is not None and floor > required:
                self.reasons.append('insufficient_retention')
        if diagnostics.get('rethreshold_rejected'):
            self.reasons.append('rethreshold_policy_incompatible')
        if not self.reasons:
            self.reasons = ['saved_identity_mismatch']
        self.action = 'infer'
        self.emit('started')

    @observational
    def reused(self, kinds, *, rethreshold=False, input_verified=True):
        self.action = 'rethreshold' if rethreshold else 'cache_reused'
        self.reasons = ['retained_evidence_covers_thresholds' if rethreshold else 'matching_completed_identity']
        self.details['input_verification'] = 'fetched' if input_verified else 'skipped'
        if not input_verified:
            self.reasons.append('input_fetch_skipped')
        if rethreshold:
            self.details['outputs'] = {kind: dict(stored_scene_threshold=meta['scene_threshold'],
                stored_marker_threshold=meta['marker_threshold'], target_scene_threshold=self.config.scene_threshold,
                target_marker_threshold=self.config.marker_threshold) for kind, meta in self.metas(kinds).items()}
        self.emit('finished')
        self.remember(kinds)

    def remember(self, kinds):
        if self.current is None:
            return
        for kind, meta in self.metas(kinds).items():
            saved = deepcopy(self.current)
            saved['model'].update(scene_threshold=meta['scene_threshold'], marker_threshold=meta['marker_threshold'])
            signature = inference_signature({**self.identity, 'scene_threshold': meta['scene_threshold'],
                'marker_threshold': meta['marker_threshold']}, self.cues, self.hashes)
            if signature == meta['inference_signature']:
                self.journal('put_proof', self.store.scope, self.scene, self.model, kind, signature, saved)

    @observational
    def checkpoint(self, checkpoint):
        self.checkpoint_signature = checkpoint.signature
        self.details['checkpoint'] = dict(reason=checkpoint.restore_reason,
            resumed_frames=checkpoint.resumed_frames, total_frames=checkpoint.total)
        if checkpoint.restore_reason != 'missing':
            self.reasons.append('checkpoint_' + checkpoint.restore_reason)
        if checkpoint.previous_signature:
            saved = self.journal('get_proof', self.store.scope, self.scene, self.model, 'checkpoint', checkpoint.previous_signature)
            if saved is not None and self.current is not None:
                self.reasons.extend(differences(saved, self.current))
        if checkpoint.resumed_frames:
            self.action = 'checkpoint_resumed'
        else:
            self.action = 'infer'
            self.reasons = [reason for reason in self.reasons if reason != 'checkpoint_compatible']
        self.emit('started')

    @observational
    def finished(self, kinds):
        self.emit('finished')
        self.remember(kinds)

    @observational
    def failed(self, stage, exc):
        self.details['failure'] = dict(stage=stage, type=type(exc).__name__)
        if self.current is None:
            self.action, self.reasons = 'blocked', ['input_verification_failed' if stage == 'inference' else 'model_load_failed']
        if stage == 'load model':
            self.action = 'blocked'
            self.reasons.append('model_load_failed')
        checkpoint = self.store.connection.execute('SELECT signature FROM ' + self.store.table('checkpoints') +
            ' WHERE scope=? AND scene=? AND model=?', (self.store.scope, self.scene, self.model)).fetchone()
        if checkpoint and self.current is not None and checkpoint[0] == self.checkpoint_signature:
            self.journal('put_proof', self.store.scope, self.scene, self.model, 'checkpoint', checkpoint[0], self.current)
        self.emit('error' if isinstance(exc, Exception) else 'interrupted')
