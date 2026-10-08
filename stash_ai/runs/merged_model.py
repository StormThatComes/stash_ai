"""Lazy model lifecycle and per-scene inference for merged output."""
from __future__ import annotations

from functools import partial
from concurrent.futures import ThreadPoolExecutor

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
        self.inference_executor = None
        self.input_prefetch = None
        self.prepare_executor = None
        self._staged = None
        with report.measure('model_identity', model=self.key):
            self.identity = model_identity(app, config, vocabulary)
        report.record_model(config, identity=self.identity)
        from stash_ai.inference.reuse_diagnostics import ReuseObserver
        self.observer = ReuseObserver(app, args, store, config, report)

    def run(self, scenes, kinds, inputs, *, input_plan=None):
        remaining = set(scenes)
        self._scene_sequence = tuple(scenes.items())
        if self.args.prefetch_batches:
            from stash_ai.inference.input_prefetch import SceneInputPrefetch
            self.input_prefetch = SceneInputPrefetch(self.app, inputs, self.report, self.key)
        try:
            bar = tqdm(self._scene_sequence, desc=self.key + " inference", unit="scene")
            for self._scene_position, (scene_id, current) in enumerate(bar):
                self.observer.begin(scene_id)
                self.report.work('')
                self.report.progress(bar, self.stats)
                paths = current["paths"]
                if not self._eligible(scene_id, paths):
                    remaining.discard(scene_id)
                    continue
                try:
                    with self.report.measure('model_scene', model=self.key, units=1, unit='scenes'):
                        if not self._process_scene(scene_id, current, paths, kinds, inputs,
                                                   trust_cached=(input_plan is not None and
                                                       scene_id in input_plan.cached[self.key]),
                                                   input_plan=input_plan):
                            progress.block_publication(self.store, remaining, self.key, kinds)
                            for blocked_scene in scenes:
                                if blocked_scene in remaining and blocked_scene != scene_id:
                                    self.observer.begin(blocked_scene)
                                    self.observer.blocked('model_load_failed')
                                    self.report.model_outcome(self.key, 'blocked')
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
            try:
                try:
                    try:
                        self._close_staged(kinds)
                    finally:
                        if self.input_prefetch is not None:
                            self.input_prefetch.close()
                finally:
                    if self.prepare_executor is not None:
                        self.prepare_executor.shutdown(wait=True)
                        self.prepare_executor = None
                    if self.inference_executor is None:
                        self._close_tagger()
                    else:
                        # Queue cleanup after any interrupted model initialization.
                        # Load, inference and close all use the same backend thread.
                        self.inference_executor.submit(self._close_tagger).result()
            finally:
                if self.inference_executor is not None:
                    self.inference_executor.shutdown(wait=True)
                    self.inference_executor = None
                self.input_prefetch = None
                self._staged = None
                self._scene_sequence = ()

    def _prefetch_next_input(self, input_plan):
        if self.input_prefetch is None or self.input_prefetch.pending is not None:
            return
        for index in range(self._scene_position + 1, len(self._scene_sequence)):
            scene, current = self._scene_sequence[index]
            paths = current['paths']
            if not paths.get('vtt') or not paths.get('sprite'):
                continue
            if input_plan is not None and scene in input_plan.cached[self.key]:
                continue
            self._prefetch_position = index
            self.input_prefetch.submit(scene, paths, self.identity, tagger=self.tagger,
                                       prefetch=self.args.prefetch_batches,
                                       workers=self.args.prepare_workers,
                                       transforms=self.prepare_executor)
            break

    def _close_tagger(self):
        if self.tagger is not None:
            try:
                self.tagger.close()
            finally:
                self.tagger = None

    def _close_staged(self, kinds):
        if self._staged is None:
            return
        scene, job, observer, _ = self._staged
        if job is not None and job is not False:
            try:
                job.close()
            finally:
                from stash_ai.runtime.cancellation import Cancelled
                progress.block_publication(self.store, [scene], self.key, kinds)
                observer.failed('inference', Cancelled('Run stopped before processing this scene.'))

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
            register(store, scene_id, key, signature, self.identity, input_options=self.input_options)
            self.stats["reused"] += 1
            self.report.model_outcome(key, 'reused')
            self.observer.reused(kinds)
            return True
        diagnostics = {}
        if args.resume and not args.restart:
            from stash_ai.inference.reuse import reuse
            if reuse(self.app, store, scene_id, self.config, self.identity, cues, media.hashes, kinds,
                     diagnostics=diagnostics, input_options=self.input_options):
                self.stats['threshold_cache_reused'] += 1
                self.report.model_outcome(key, 'rethreshold')
                self.observer.reused(kinds, rethreshold=True)
                return True
        self.observer.required(kinds, diagnostics)
        return False

    def _load_tagger(self):
        def load():
            # Assign on the owner thread so finally can close a model even if
            # the caller is interrupted while waiting for initialization.
            self.tagger = self.app.services.load_tagger(
                self.config, self.vocabulary, partial(preparation.prepare_image, self.app))

        try:
            with self.report.measure('model_load', model=self.key):
                if self.args.prefetch_batches:
                    self.prepare_executor = ThreadPoolExecutor(
                        max_workers=self.args.prepare_workers, thread_name_prefix='transform-frames')
                    self.inference_executor = ThreadPoolExecutor(
                        max_workers=1, thread_name_prefix='inference-model')
                    self.inference_executor.submit(load).result()
                else:
                    load()
            self.report.record_model(self.config, tagger=self.tagger)
        except Exception as exc:
            self._record_error('load model', None, exc)
            return False
        return True

    def cached_plan(self, scene_id, paths, kinds):
        from stash_ai.inference.reuse import cached_reuse
        return cached_reuse(self.app, self.store, scene_id, self.config, self.identity, paths, kinds)

    def _reuse_cached(self, scene_id, paths, kinds):
        saved = self.cached_plan(scene_id, paths, kinds)
        if saved is None:
            return False
        from stash_ai.rethreshold.scenes import apply
        apply(self.store, scene_id, self.key, saved.thresholds)
        self.stats['threshold_cache_reused' if saved.rethreshold else 'reused'] += 1
        self.stats['input_fetch_skipped'] += 1
        self.report.model_outcome(self.key, 'rethreshold' if saved.rethreshold else 'reused')
        self.observer.reused(kinds, rethreshold=saved.rethreshold, input_verified=False)
        return True

    def _process_scene(self, scene_id, current, paths, kinds, inputs, *, trust_cached=False,
                       input_plan=None):
        """Return False only when model loading fails and its remaining scenes must stop."""
        if self._staged is not None and self._staged[0] == scene_id:
            _, job, self.observer, error = self._staged
            self._staged = None
            if error is not None:
                raise error
        else:
            job = self._prepare_scene(scene_id, current, paths, kinds, inputs,
                                      trust_cached=trust_cached, input_plan=input_plan)
        if job is False:
            return False
        if job is None:
            if self.input_prefetch is not None:
                self.input_prefetch.take_batches(scene_id, 1)
            return True
        self.input_options = job.input_options
        try:
            self._prefetch_next_input(input_plan)
            self._predict_scene(job, kinds, inputs, input_plan)
        finally:
            job.close()
        return True

    def _prepare_scene(self, scene_id, current, paths, kinds, inputs, *, trust_cached=False,
                       input_plan=None):
        app, args, store = self.app, self.args, self.store
        store.cache.pressure(force=True)
        if trust_cached:
            with self.report.measure('inference_reuse_check', model=self.key):
                if self._reuse_cached(scene_id, paths, kinds):
                    return None
        from stash_ai.inference.reuse import input_options
        self.input_options = input_options(app, paths)
        from stash_ai.storage.input_checks import defer, verified
        try:
            with self.report.measure('input_verify', model=self.key):
                prepared = self.input_prefetch.take(scene_id) if self.input_prefetch is not None else None
                cues, media, signature = (prepared if prepared is not None
                                          else inputs.inspect(app, paths, self.identity))
                if prepared is not None:
                    # Model auto-download can update identity after lookahead
                    # was submitted; signatures always use the current assets.
                    signature = inference_signature(self.identity, cue_manifest(cues), media.hashes)
        except BaseException:
            defer(store, scene_id, self.key, 'read_error')
            raise
        if input_plan is not None:
            input_plan.verified(scene_id, cues, media.hashes)
        self.report.record_input_workload(scene_id, self.key, cues, media.hashes)
        verified(store, scene_id, self.key)
        self.observer.inputs(self.identity, cues, media.hashes, paths)
        with self.report.measure('inference_reuse_check', model=self.key):
            if self._reuse(scene_id, signature, cues, media, kinds):
                return None
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
        from .model_scene import ModelScene
        return ModelScene(scene_id, paths, cues, media, signature, input_options=self.input_options)

    def _initialize_scene(self, job, kinds):
        from stash_ai.evidence.temporal.statistics import CHECKPOINT_FORMAT
        app, args, store = self.app, self.args, self.store
        if not args.resume:
            with store.connection:
                checkpoints.clear_checkpoint(store, job.scene, self.key)
        job.checkpoint = checkpoints.Checkpoint(
            store, job.scene, self.key,
            self._checkpoint_signature(job.scene, job.signature, job.cues, job.media, kinds, CHECKPOINT_FORMAT),
            len(job.cues), len(self.vocabulary.labels), args.checkpoint_batches)
        job.checkpoint.on_restore = self.observer.checkpoint
        job.state = prediction._restore_prediction_state(app, self.tagger, job.cues,
                                                         checkpoint=job.checkpoint, report=self.report)
        if self.input_prefetch is not None:
            job.prepared = self.input_prefetch.take_batches(job.scene, job.state.processed)

    def _stage_next_scene(self, kinds, inputs, input_plan):
        lookahead = self.input_prefetch
        if self._staged is not None or lookahead is None or lookahead.pending is None:
            return
        future = lookahead.pending
        # Slow/failed input must never hold up finalization of the previous
        # scene. Its read error is still observed only when that scene is visited.
        if not future.done() or future.cancelled() or future.exception() is not None:
            return
        scene_id, current = self._scene_sequence[self._prefetch_position]
        from stash_ai.inference.reuse_diagnostics import ReuseObserver
        previous = self.observer, self.input_options, self.report.work()
        self.observer = ReuseObserver(self.app, self.args, self.store, self.config, self.report)
        self.observer.begin(scene_id)
        job, error = None, None
        try:
            with self.report.measure('scene_handoff', model=self.key, units=1, unit='scenes'):
                job = self._prepare_scene(scene_id, current, current['paths'], kinds, inputs,
                                          trust_cached=(input_plan is not None and
                                                        scene_id in input_plan.cached[self.key]),
                                          input_plan=input_plan)
                if job is not None and job is not False:
                    self._initialize_scene(job, kinds)
                    from stash_ai.inference.batches import BatchPredictions
                    job.batches = BatchPredictions(
                        self.app, self.tagger, job.media, job.paths, job.cues[job.state.processed:],
                        prefetch=self.args.prefetch_batches, prepare_workers=self.args.prepare_workers,
                        report=self.report, inference_executor=self.inference_executor,
                        prepared_inputs=job.prepared, transform_executor=self.prepare_executor)
                else:
                    lookahead.take_batches(scene_id, 1)
        except Exception as exc:
            error = exc
            if job is not None and job is not False:
                job.close()
            lookahead.take_batches(scene_id, 1)
            job = None
        finally:
            self._staged = (scene_id, job, self.observer, error)
            self.observer, self.input_options, work = previous
            self.report.work(work)

    def _predict_scene(self, job, kinds, inputs, input_plan):
        app, args, store = self.app, self.args, self.store
        if job.state is None:
            self._initialize_scene(job, kinds)
        self.report.work('checkpoint_resumed' if job.checkpoint.resumed_frames else 'fresh')
        analysis = {}
        with self.report.measure('inference_scene', model=self.key, units=1, unit='scenes'):
            predictions, frames = prediction.predict_scene(
                app, self.tagger, job.media, job.paths, job.cues, checkpoint=job.checkpoint,
                prefetch=args.prefetch_batches, analysis=analysis, report=self.report,
                prepare_workers=args.prepare_workers,
                inference_executor=self.inference_executor, state=job.state,
                batch_results=job.batches, prepared_inputs=job.prepared,
                transform_executor=self.prepare_executor,
                before_finalize=lambda: self._stage_next_scene(kinds, inputs, input_plan))
        # Input may finish while finalization/capture runs. Retry handoff at
        # these boundaries without ever waiting for an unfinished input.
        self._stage_next_scene(kinds, inputs, input_plan)
        with self.report.measure('inference_output_capture', model=self.key, units=1, unit='scenes'):
            outputs = merge_results.capture_outputs(
                app, self.config, self.vocabulary, job.cues, predictions, frames, analysis,
                inference_signature=job.signature,
                frame_input_signature=frame_signature(cue_manifest(job.cues), job.media.hashes))
        self._stage_next_scene(kinds, inputs, input_plan)
        with self.report.measure('inference_commit', model=self.key):
            checkpoints.finish(store, job.scene, self.key, outputs)
            from stash_ai.inference.provenance import prune
            prune(store, job.scene, self.key)
        self.report.model_outcome(self.key, 'checkpoint_resumed' if job.checkpoint.resumed_frames else 'fresh')
        self.stats["processed"] += 1
        self.observer.finished(kinds)

    def _checkpoint_signature(self, scene_id, signature, cues, media, kinds, analysis_format):
        from stash_ai.inference.provenance import register, resume_signature, prune
        result = resume_signature(self.store, scene_id, self.key, self.identity,
                                  cue_manifest(cues), media.hashes, kinds, analysis_format)
        register(self.store, scene_id, self.key, signature, self.identity, input_options=self.input_options)
        prune(self.store, scene_id, self.key, keep=(signature,), checkpoint=result,
              kinds=kinds, analysis_format=analysis_format)
        return result

    def _record_error(self, stage, scene_id, exc):
        self.stats["errors"] += 1
        self.report.model_outcome(self.key, 'error')
        self.observer.failed(stage, exc)
        self.report.issue(stage, self.key, scene_id, exc,
                          self.app.services.safe_error(exc, api_key=self.app.STASH_API_KEY))
