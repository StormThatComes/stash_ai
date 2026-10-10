"""Lazy model lifecycle and per-scene inference for merged output."""
from __future__ import annotations

from functools import partial
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

import stash_ai.inference.results as merge_results
from stash_ai.storage import progress
from stash_ai.inference import checkpoints, prediction, preparation
from stash_ai.inference.identity import model_identity, asset_stamps, require_assets_unchanged
from stash_ai.inference.identity_types import ProvenanceIdentity, ExecutionIdentity
from .model_scene import ModelScene, ScenePreparation, PreparationStatus, StagedScene, SceneContext
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
        self._load_failure = None
        self._cleanup_primary = None
        with report.measure('model_identity', model=self.key):
            self.asset_stamps = asset_stamps(config)
            self.identity = model_identity(app, config, vocabulary,
                known_stamps=self.asset_stamps, require_unchanged=True)
        self.provenance = ProvenanceIdentity.from_legacy(self.identity)
        self.execution = ExecutionIdentity.from_legacy(self.identity, args)
        report.record_model(config, identity=self.identity)
        from stash_ai.inference.reuse_diagnostics import ReuseObserver
        self.observer = ReuseObserver(app, args, store, config, report)

    def run(self, scenes, kinds, inputs, *, input_plan=None):
        self._scene_sequence = tuple(scenes.items())
        if self.args.prefetch_batches:
            from stash_ai.inference.input_prefetch import SceneInputPrefetch
            self.input_prefetch = SceneInputPrefetch(self.app, inputs, self.report, self.key)
        bar = None
        try:
            bar = tqdm(self._scene_sequence, desc=self.key + " inference", unit="scene",
                       position=0, dynamic_ncols=True)
            for self._scene_position, (scene_id, current) in enumerate(bar):
                self.observer.begin(scene_id)
                self.report.work('')
                paths = current["paths"]
                if not self._eligible(scene_id, paths):
                    self.report.progress(bar, self.stats)
                    continue
                try:
                    if self._load_failure is not None:
                        # A failed backend does not invalidate completed evidence.
                        # Continue cheap authoritative reuse without another load/GET.
                        if (self.args.resume and not self.args.restart and
                                not self.args.force_fetch_vtt and
                                self._reuse_cached(scene_id, paths, kinds, input_plan=input_plan)):
                            continue
                        progress.block_publication(self.store, [scene_id], self.key, kinds)
                        self.observer.blocked('model_load_failed')
                        self.report.model_outcome(self.key, 'blocked')
                        continue
                    with self.report.measure('model_scene', model=self.key, units=1, unit='scenes'):
                        if not self._process_scene(scene_id, current, paths, kinds, inputs,
                                                   trust_cached=(input_plan is not None and
                                                       scene_id in input_plan.cached[self.key]),
                                                   input_plan=input_plan):
                            progress.block_publication(self.store, [scene_id], self.key, kinds)
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
                finally:
                    self.report.progress(bar, self.stats)
        finally:
            from stash_ai.runtime.cleanup import cleanup
            def close_backend():
                if self.inference_executor is None:
                    self._close_tagger()
                else:
                    self.inference_executor.submit(self._close_tagger).result()
            def reset():
                self.prepare_executor = self.inference_executor = self.input_prefetch = None
                self._staged, self._scene_sequence = None, ()
            cleanup([
                lambda: bar.close() if bar is not None else None,
                lambda: self._close_staged(kinds),
                lambda: self.input_prefetch.close() if self.input_prefetch is not None else None,
                lambda: self.prepare_executor.shutdown(wait=True) if self.prepare_executor is not None else None,
                close_backend,
                lambda: self.inference_executor.shutdown(wait=True) if self.inference_executor is not None else None,
                reset,
            ], report=self.report, primary=self._cleanup_primary)

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
        staged, self._staged = self._staged, None
        preparation = staged.preparation
        if preparation.status is PreparationStatus.READY:
            try:
                preparation.job.close()
            finally:
                from stash_ai.runtime.cancellation import Cancelled
                progress.block_publication(self.store, [staged.scene], self.key, kinds)
                staged.observer.failed('inference', Cancelled('Run stopped before processing this scene.'))

        elif preparation.status is PreparationStatus.FAILED:
            progress.block_publication(self.store, [staged.scene], self.key, kinds)
            staged.observer.failed('inference', preparation.error)

    def _eligible(self, scene_id, paths):
        if not paths.get("sprite") or not paths.get("vtt"):
            from stash_ai.storage.input_checks import defer
            defer(self.store, scene_id, self.key)
            self.stats["skipped"] += 1
            self.report.model_outcome(self.key, 'missing_media')
            self.observer.blocked('missing_media')
            return False
        return True

    def _reuse(self, scene_id, signature, cues, media, kinds, *, context, snapshot=None):
        args, store, key = self.args, self.store, self.key
        if args.resume and progress.completed(store, scene_id, key, kinds, signature):
            from stash_ai.inference.provenance import register
            register(store, scene_id, key, signature, self.identity, input_options=context.input_options)
            self.stats["reused"] += 1
            self.report.model_outcome(key, 'reused')
            context.observer.reused(kinds)
            return True
        diagnostics = {}
        if args.resume and not args.restart:
            context.observer.before_reuse(kinds)
            from stash_ai.inference.reuse import reuse
            if reuse(self.app, store, scene_id, self.config, self.identity, cues, media.hashes, kinds,
                     diagnostics=diagnostics, input_options=context.input_options, snapshot=snapshot):
                changed = diagnostics.get('rethreshold', False)
                self.stats['threshold_cache_reused' if changed else 'reused'] += 1
                self.report.model_outcome(key, 'rethreshold' if changed else 'reused')
                context.observer.reused(kinds, rethreshold=changed)
                return True
        context.observer.required(kinds, diagnostics)
        return False

    def _load_tagger(self):
        def load():
            # Assign on the owner thread so finally can close a model even if
            # the caller is interrupted while waiting for initialization.
            before = asset_stamps(self.config)
            require_assets_unchanged(self.asset_stamps, before, allow_download=True)
            from stash_ai.backends.profiling import use
            with use(self.report, enabled=bool(self.args.profile_merge)):
                self.tagger = self.app.services.load_tagger(
                    self.config, self.vocabulary, partial(preparation.prepare_image, self.app))
            after = asset_stamps(self.config)
            require_assets_unchanged(before, after, allow_download=True)
            self.asset_stamps = after

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
            self._load_failure = exc
            self._record_error('load model', None, exc)
            return False
        return True

    def cached_plan(self, scene_id, paths, kinds):
        from stash_ai.inference.reuse import cached_reuse
        return cached_reuse(self.app, self.store, scene_id, self.config, self.identity, paths, kinds)

    def _reuse_cached(self, scene_id, paths, kinds, *, input_plan=None, observer=None):
        saved = input_plan.plan(scene_id, self.key) if input_plan is not None else self.cached_plan(scene_id, paths, kinds)
        if saved is None:
            return False
        observer = observer or self.observer
        observer.before_reuse(kinds)
        from stash_ai.rethreshold.scenes import apply
        apply(self.store, scene_id, self.key, saved.thresholds)
        self.stats['threshold_cache_reused' if saved.rethreshold else 'reused'] += 1
        self.stats['input_fetch_skipped'] += 1
        self.report.model_outcome(self.key, 'rethreshold' if saved.rethreshold else 'reused')
        observer.reused(kinds, rethreshold=saved.rethreshold, input_verified=False)
        return True

    def _process_scene(self, scene_id, current, paths, kinds, inputs, *, trust_cached=False,
                       input_plan=None):
        """Transfer a prepared job to the consumer, which alone processes/commits."""
        if self._staged is not None and self._staged.scene == scene_id:
            staged, self._staged = self._staged, None
            self.observer = staged.observer
            prepared = staged.preparation
        else:
            prepared = self._prepare_scene(scene_id, current, paths, kinds, inputs,
                                           trust_cached=trust_cached, input_plan=input_plan)
        if prepared.status is PreparationStatus.FAILED:
            raise prepared.error
        if prepared.status is PreparationStatus.BLOCKED:
            return False
        if prepared.status is PreparationStatus.REUSED:
            if self.input_prefetch is not None:
                self.input_prefetch.take_batches(scene_id, 1)
            return True
        job = prepared.job
        try:
            self._prefetch_next_input(input_plan)
            self._predict_scene(job, kinds, inputs, input_plan)
        finally:
            from stash_ai.runtime.cleanup import cleanup
            cleanup([job.close], report=self.report)
        return True

    def _prepare_scene(self, scene_id, current, paths, kinds, inputs, *, trust_cached=False,
                       input_plan=None, context=None):
        context = context or SceneContext(scene_id, self.observer)
        app, args, store = self.app, self.args, self.store
        store.cache.pressure(force=True)
        if trust_cached:
            with self.report.measure('inference_reuse_check', model=self.key):
                if self._reuse_cached(scene_id, paths, kinds, input_plan=input_plan, observer=context.observer):
                    return ScenePreparation(PreparationStatus.REUSED)
        from stash_ai.inference.reuse import input_options
        context.input_options = input_options(app, paths)
        from stash_ai.storage.input_checks import defer, verified
        try:
            with self.report.measure('input_verify', model=self.key):
                prepared = self.input_prefetch.take(scene_id) if self.input_prefetch is not None else None
                inspected = (prepared if prepared is not None else inputs.inspect(app, paths, self.identity))
                snapshot, media, signature = inspected.snapshot, inspected.media, inspected.signature
                cues = snapshot.cues
                if prepared is not None:
                    # Model auto-download can update identity after lookahead
                    # was submitted; signatures always use the current assets.
                    signature = snapshot.inference_signature(self.identity)
        except BaseException:
            defer(store, scene_id, self.key, 'read_error')
            raise
        if input_plan is not None:
            input_plan.verified(scene_id, cues, media.hashes, snapshot=snapshot)
        self.report.record_input_workload(scene_id, self.key, cues, media.hashes, snapshot=snapshot)
        verified(store, scene_id, self.key)
        context.observer.inputs(self.identity, cues, media.hashes, paths, snapshot=snapshot)
        with self.report.measure('inference_reuse_check', model=self.key):
            if self._reuse(scene_id, signature, cues, media, kinds, context=context, snapshot=snapshot):
                return ScenePreparation(PreparationStatus.REUSED)
        if not args.restart:
            stale = progress.request_stale_refresh(store, scene_id, self.key, kinds, signature)
            self.stats['auto_refresh_requested'] += len(stale)
        if self.tagger is None:
            if not self._load_tagger():
                return ScenePreparation(PreparationStatus.BLOCKED)
            # Auto-download may just have supplied missing weight files.
            self.identity = model_identity(app, self.config, self.vocabulary,
                                           known_files=self.identity['files'],
                                           known_stamps=self.asset_stamps, require_unchanged=True)
            self.provenance = ProvenanceIdentity.from_legacy(self.identity)
            self.execution = ExecutionIdentity.from_legacy(self.identity, args)
            self.report.record_model(self.config, identity=self.identity)
            signature = snapshot.inference_signature(self.identity)
            context.observer.inputs(self.identity, cues, media.hashes, paths, snapshot=snapshot)
        return ScenePreparation(PreparationStatus.READY,
            ModelScene(scene_id, paths, cues, media, signature, input_options=context.input_options, verified_input=snapshot, context=context))

    def _initialize_scene(self, job, kinds):
        from stash_ai.evidence.temporal.statistics import CHECKPOINT_FORMAT
        app, args, store = self.app, self.args, self.store
        if not args.resume:
            with store.connection:
                checkpoints.clear_checkpoint(store, job.scene, self.key)
        job.checkpoint = checkpoints.Checkpoint(
            store, job.scene, self.key,
            self._checkpoint_signature(job.scene, job.signature, job.cues, job.media, kinds, CHECKPOINT_FORMAT, snapshot=job.verified_input, input_options=job.input_options),
            len(job.cues), len(self.vocabulary.labels), args.checkpoint_batches)
        job.checkpoint.on_restore = job.context.observer.checkpoint
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
        observer = ReuseObserver(self.app, self.args, self.store, self.config, self.report)
        observer.begin(scene_id)
        context = SceneContext(scene_id, observer)
        prepared = ScenePreparation(PreparationStatus.REUSED)
        try:
            with self.report.model(self.key), self.report.measure('scene_handoff', model=self.key, units=1, unit='scenes'):
                prepared = self._prepare_scene(scene_id, current, current['paths'], kinds, inputs,
                    trust_cached=(input_plan is not None and scene_id in input_plan.cached[self.key]),
                    input_plan=input_plan, context=context)
                if prepared.status is PreparationStatus.READY:
                    job = prepared.job
                    self._initialize_scene(job, kinds)
                    from stash_ai.inference.batches import BatchPredictions
                    job.batches = BatchPredictions(
                        self.app, self.tagger, job.media, job.paths, job.cues[job.state.processed:],
                        prefetch=self.args.prefetch_batches, prepare_workers=self.args.prepare_workers,
                        result_prefetch=self.args.result_prefetch_batches,
                        report=self.report, inference_executor=self.inference_executor,
                        prepared_inputs=job.prepared, transform_executor=self.prepare_executor)
                else:
                    lookahead.take_batches(scene_id, 1)
        except BaseException as exc:
            from stash_ai.runtime.cleanup import cleanup
            cleanup(([prepared.job.close] if prepared.job is not None else []) +
                    [lambda: lookahead.take_batches(scene_id, 1)], report=self.report)
            prepared = ScenePreparation(PreparationStatus.FAILED, error=exc)
            if not isinstance(exc, Exception):
                raise
        finally:
            self._staged = StagedScene(scene_id, prepared, context.observer)

    def _predict_scene(self, job, kinds, inputs, input_plan):
        app, args, store = self.app, self.args, self.store
        if job.state is None:
            self._initialize_scene(job, kinds)
        self.report.counts['checkpoint_restored_frames'] += job.state.processed
        self.report.work('checkpoint_resumed' if job.checkpoint.resumed_frames else 'fresh')
        from .inference_progress import InferenceProgress
        with InferenceProgress(tqdm, total=len(job.cues), initial=job.state.processed,
                               desc=f'{self.key} scene {job.scene}',
                               sample=lambda: self._buffer_snapshot(job)) as display:
            analysis = {}
            with self.report.measure('inference_scene', model=self.key, units=1, unit='scenes'):
                predictions, frames = prediction.predict_scene(
                    app, self.tagger, job.media, job.paths, job.cues, checkpoint=job.checkpoint,
                    prefetch=args.prefetch_batches, analysis=analysis, report=self.report,
                    prepare_workers=args.prepare_workers,
                    result_prefetch=args.result_prefetch_batches,
                    inference_executor=self.inference_executor, state=job.state,
                    batch_results=job.batches, prepared_inputs=job.prepared,
                    transform_executor=self.prepare_executor,
                    on_progress=display.update,
                    on_batches=lambda batches: setattr(job, 'batch_usage', batches.buffer_usage),
                    before_finalize=lambda: self._stage_next_scene(kinds, inputs, input_plan))
            # Input may finish while finalization/capture runs. Retry handoff at
            # these boundaries without ever waiting for an unfinished input.
            self._stage_next_scene(kinds, inputs, input_plan)
            display.refresh()
            with self.report.measure('inference_output_capture', model=self.key, units=1, unit='scenes'):
                outputs = merge_results.capture_outputs(
                    app, self.config, self.vocabulary, job.cues, predictions, frames, analysis,
                    inference_signature=job.signature, verified_input=job.verified_input,
                    frame_input_signature=(job.verified_input.frame_signature if job.verified_input is not None else
                                           frame_signature(cue_manifest(job.cues), job.media.hashes)))
            self._stage_next_scene(kinds, inputs, input_plan)
            display.refresh()
            with self.report.measure('inference_commit', model=self.key):
                checkpoints.finish(store, job.scene, self.key, outputs)
                from stash_ai.inference.provenance import prune
                prune(store, job.scene, self.key)
            self.report.model_outcome(self.key, 'checkpoint_resumed' if job.checkpoint.resumed_frames else 'fresh')
            self.stats["processed"] += 1
            job.context.observer.finished(kinds)
            display.refresh()

    def _buffer_snapshot(self, job):
        args = self.args
        read = job.batches.buffer_usage if job.batches is not None else job.batch_usage
        current = (read() if read is not None else
                   dict(input=(0, args.prefetch_batches) if args.prefetch_batches else None,
                        result=(0, args.result_prefetch_batches) if args.prefetch_batches else None))
        following = None
        if self._staged is not None and self._staged.preparation.job is not None:
            staged = self._staged.preparation.job
            following = dict(scene=staged.scene, **staged.batches.buffer_usage())
        elif self.input_prefetch is not None and self.input_prefetch.scene is not None:
            scene = self.input_prefetch.scene
            following = dict(scene=scene,
                input=self.input_prefetch.buffer_usage(scene) or (0, args.prefetch_batches),
                result=(0, args.result_prefetch_batches))
        return dict(current=current, next=following, cache=self.store.cache.usage(),
                    checkpoint=(job.checkpoint.batches, job.checkpoint.every))

    def _checkpoint_signature(self, scene_id, signature, cues, media, kinds, analysis_format, *, snapshot=None, input_options=None):
        from stash_ai.inference.provenance import register, resume_signature, prune
        result = resume_signature(self.store, scene_id, self.key, self.identity,
                                  snapshot.manifest if snapshot is not None else cue_manifest(cues),
                                  snapshot.hashes if snapshot is not None else media.hashes, kinds, analysis_format)
        register(self.store, scene_id, self.key, signature, self.identity, input_options=input_options)
        prune(self.store, scene_id, self.key, keep=(signature,), checkpoint=result,
              kinds=kinds, analysis_format=analysis_format)
        return result

    def _record_error(self, stage, scene_id, exc):
        if self._cleanup_primary is None:
            self._cleanup_primary = exc
        self.stats["errors"] += 1
        self.report.model_outcome(self.key, 'error')
        self.observer.failed(stage, exc)
        self.report.issue(stage, self.key, scene_id, exc,
                          self.app.services.safe_error(exc, api_key=self.app.STASH_API_KEY))
