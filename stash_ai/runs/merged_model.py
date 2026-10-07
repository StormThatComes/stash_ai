"""Lazy model lifecycle and per-scene inference for merged output."""
from __future__ import annotations

from functools import partial

from tqdm import tqdm

import stash_ai.inference.results as merge_results
from stash_ai.storage import progress
from stash_ai.inference import checkpoints, prediction, preparation
from stash_ai.inference.identity import model_identity
from stash_ai.inference.input_identity import cue_manifest, frame_signature, inference_signature


class MergedModelRun:
    """Own one tagger; borrow the parent run's store, counters and report."""

    def __init__(self, app, args, store, config, vocabulary, *, stats, report):
        self.app, self.args, self.store = app, args, store
        self.config, self.vocabulary = config, vocabulary
        self.stats, self.report = stats, report
        self.key = config.source.key
        self.tagger = None
        with report.measure('model_identity', model=self.key):
            self.identity = model_identity(app, config, vocabulary)
        report.record_model(config, identity=self.identity)
        from stash_ai.inference.reuse_diagnostics import ReuseObserver
        self.observer = ReuseObserver(app, args, store, config, report)

    def run(self, scenes, kinds, inputs):
        remaining = set(scenes)
        try:
            bar = tqdm(scenes.items(), desc=self.key + " inference", unit="scene")
            for scene_id, current in bar:
                self.observer.begin(scene_id)
                self.report.work('')
                self.report.progress(bar, self.stats)
                paths = current["paths"]
                if not self._eligible(scene_id, paths):
                    remaining.discard(scene_id)
                    continue
                try:
                    with self.report.measure('model_scene', model=self.key, units=1, unit='scenes'):
                        if not self._process_scene(scene_id, current, paths, kinds, inputs):
                            progress.block_publication(self.store, remaining, self.key, kinds)
                            break
                except Exception as exc:
                    from stash_ai.storage.input_checks import pending
                    if self.key not in pending(self.store, scene_id):
                        progress.block_publication(self.store, [scene_id], self.key, kinds)
                    self._record_error('inference', scene_id, exc)
                except BaseException as exc:
                    self.observer.failed('inference', exc)
                    from stash_ai.storage.input_checks import pending
                    if self.key not in pending(self.store, scene_id):
                        progress.block_publication(self.store, [scene_id], self.key, kinds)
                    raise
                remaining.discard(scene_id)
        finally:
            if self.tagger is not None:
                self.tagger.close()

    def _eligible(self, scene_id, paths):
        if not paths.get("sprite") or not paths.get("vtt"):
            from stash_ai.storage.input_checks import defer
            defer(self.store, scene_id, self.key)
            self.stats["skipped"] += 1
            self.report.model_outcome(self.key, 'missing_media')
            self.observer.blocked('missing_media')
            return False
        return True

    def _reuse(self, scene_id, signature, cues, media, kinds):
        args, store, key = self.args, self.store, self.key
        if args.resume and progress.completed(store, scene_id, key, kinds, signature):
            from stash_ai.inference.provenance import register
            register(store, scene_id, key, signature, self.identity)
            self.stats["reused"] += 1
            self.report.model_outcome(key, 'reused')
            self.observer.reused(kinds)
            return True
        diagnostics = {}
        if args.resume and not args.restart:
            from stash_ai.inference.reuse import reuse
            if reuse(self.app, store, scene_id, self.config, self.identity, cues, media.hashes, kinds, diagnostics=diagnostics):
                self.stats['threshold_cache_reused'] += 1
                self.report.model_outcome(key, 'rethreshold')
                self.observer.reused(kinds, rethreshold=True)
                return True
        self.observer.required(kinds, diagnostics)
        return False

    def _load_tagger(self):
        try:
            with self.report.measure('model_load', model=self.key):
                self.tagger = self.app.services.load_tagger(
                    self.config, self.vocabulary, partial(preparation.prepare_image, self.app))
            self.report.record_model(self.config, tagger=self.tagger)
        except Exception as exc:
            self._record_error('load model', None, exc)
            return False
        return True

    def _process_scene(self, scene_id, current, paths, kinds, inputs):
        """Return False only when model loading fails and its remaining scenes must stop."""
        app, args, store = self.app, self.args, self.store
        store.cache.pressure(force=True)
        from stash_ai.storage.input_checks import defer, verified
        try:
            with self.report.measure('input_verify', model=self.key):
                cues, media, signature = inputs.inspect(app, paths, self.identity)
        except BaseException:
            defer(store, scene_id, self.key, 'read_error')
            raise
        verified(store, scene_id, self.key)
        self.observer.inputs(self.identity, cues, media.hashes, paths)
        with self.report.measure('inference_reuse_check', model=self.key):
            if self._reuse(scene_id, signature, cues, media, kinds):
                return True
        if not args.restart:
            stale = progress.request_stale_refresh(store, scene_id, self.key, kinds, signature)
            self.stats['auto_refresh_requested'] += len(stale)
        if self.tagger is None:
            if not self._load_tagger():
                return False
            # Auto-download may just have supplied missing weight files.
            self.identity = model_identity(app, self.config, self.vocabulary,
                                           known_files=self.identity['files'])
            self.report.record_model(self.config, identity=self.identity)
            signature = inference_signature(self.identity, cue_manifest(cues), media.hashes)
            self.observer.inputs(self.identity, cues, media.hashes, paths)
        self._predict_scene(scene_id, paths, cues, media, signature, kinds)
        return True

    def _predict_scene(self, scene_id, paths, cues, media, signature, kinds):
        from stash_ai.evidence.temporal.statistics import CHECKPOINT_FORMAT
        app, args, store = self.app, self.args, self.store
        if not args.resume:
            with store.connection:
                checkpoints.clear_checkpoint(store, scene_id, self.key)
        checkpoint = checkpoints.Checkpoint(
            store, scene_id, self.key,
            self._checkpoint_signature(scene_id, signature, cues, media, kinds, CHECKPOINT_FORMAT),
            len(cues), len(self.vocabulary.labels), args.checkpoint_batches)
        checkpoint.on_restore = self.observer.checkpoint
        analysis = {}
        with self.report.measure('inference_scene', model=self.key, units=1, unit='scenes'):
            predictions, frames = prediction.predict_scene(
                app, self.tagger, media, paths, cues, checkpoint=checkpoint,
                prefetch=args.prefetch_batches, analysis=analysis, report=self.report,
                prepare_workers=args.prepare_workers)
        outputs = merge_results.capture_outputs(
            app, self.config, self.vocabulary, cues, predictions, frames, analysis,
            inference_signature=signature,
            frame_input_signature=frame_signature(cue_manifest(cues), media.hashes))
        with self.report.measure('inference_commit', model=self.key):
            checkpoints.finish(store, scene_id, self.key, outputs)
            from stash_ai.inference.provenance import prune
            prune(store, scene_id, self.key)
        self.report.model_outcome(self.key, 'checkpoint_resumed' if getattr(checkpoint, 'resumed_frames', 0) else 'fresh')
        self.stats["processed"] += 1
        self.observer.finished(kinds)

    def _checkpoint_signature(self, scene_id, signature, cues, media, kinds, analysis_format):
        from stash_ai.inference.provenance import register, resume_signature, prune
        result = resume_signature(self.store, scene_id, self.key, self.identity,
                                  cue_manifest(cues), media.hashes, kinds, analysis_format)
        register(self.store, scene_id, self.key, signature, self.identity)
        prune(self.store, scene_id, self.key, keep=(signature,), checkpoint=result,
              kinds=kinds, analysis_format=analysis_format)
        return result

    def _record_error(self, stage, scene_id, exc):
        self.stats["errors"] += 1
        self.report.model_outcome(self.key, 'error')
        self.observer.failed(stage, exc)
        self.report.issue(stage, self.key, scene_id, exc,
                          self.app.services.safe_error(exc, api_key=self.app.STASH_API_KEY))
