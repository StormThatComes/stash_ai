"""Lazy model lifecycle and per-scene inference for merged output."""
from __future__ import annotations

from functools import partial

from tqdm import tqdm

from stash_ai import merge, progress
from stash_ai.inference import checkpoints, prediction, preparation
from stash_ai.inference.identity import model_identity
from stash_ai.signatures import digest


class MergedModelRun:
    """Own one tagger; borrow the parent run's store, counters and report."""

    def __init__(self, app, args, store, config, vocabulary, *, stats, report):
        self.app, self.args, self.store = app, args, store
        self.config, self.vocabulary = config, vocabulary
        self.stats, self.report = stats, report
        self.key = config.source.key
        self.tagger = None
        self.identity = model_identity(app, config, vocabulary)

    def run(self, scenes, kinds, replacement, inputs):
        try:
            bar = tqdm(scenes.items(), desc=self.key + " inference", unit="scene")
            for scene_id, current in bar:
                self.report.progress(bar, self.stats)
                paths = current["paths"]
                if not self._eligible(scene_id, paths):
                    continue
                try:
                    if not self._process_scene(scene_id, current, paths, kinds, replacement, inputs):
                        break
                except Exception as exc:
                    self._record_error('inference', scene_id, exc)
        finally:
            if self.tagger is not None:
                self.tagger.close()

    def _eligible(self, scene_id, paths):
        if not paths.get("sprite") or not paths.get("vtt"):
            self.stats["skipped"] += 1
            return False
        if (self.app.SKIP_EXISTING_OUTPUTS and not self.args.restart
                and merge.has_outputs(self.app, self.store, scene_id, self.key)):
            self.stats["skipped"] += 1
            return False
        return True

    def _reuse(self, scene_id, current, signature, cues, media, kinds, replacement):
        args, store, key = self.args, self.store, self.key
        if args.resume and not args.restart and current['publication_reason'] != 'reused':
            collapsed = progress.reuse_replacement_policy(
                store, scene_id, key, kinds, signature, replacement)
            if collapsed is not None:
                self.stats["replacement_history_collapsed"] += collapsed
                self.stats["reused"] += 1
                return True
        if args.resume and progress.completed(store, scene_id, key, kinds, signature):
            self.stats["reused"] += 1
            return True
        if args.resume and not args.restart:
            from stash_ai.thresholds import reuse
            if reuse(self.app, store, scene_id, self.config, self.identity, cues, media.hashes, kinds):
                if current['publication_reason'] != 'reused':
                    collapsed = progress.collapse_replacement_history(
                        store, scene_id, key, kinds, replacement)
                    if collapsed is None:
                        raise RuntimeError(
                            f'{scene_id}/{key}: threshold cache did not leave a current inference segment.')
                    self.stats["replacement_history_collapsed"] += collapsed
                self.stats['threshold_cache_reused'] += 1
                return True
        return False

    def _load_tagger(self):
        try:
            self.tagger = self.app.load_tagger(
                self.config, self.vocabulary, partial(preparation.prepare_image, self.app))
        except Exception as exc:
            self._record_error('load model', None, exc)
            return False
        return True

    def _process_scene(self, scene_id, current, paths, kinds, replacement, inputs):
        """Return False only when model loading fails and its remaining scenes must stop."""
        app, args, store = self.app, self.args, self.store
        store.cache.pressure(force=True)
        cues, media, signature = inputs.inspect(app, paths, self.identity)
        if self._reuse(scene_id, current, signature, cues, media, kinds, replacement):
            return True
        if not args.restart:
            stale = progress.request_stale_refresh(store, scene_id, self.key, kinds, signature)
            self.stats['auto_refresh_requested'] += len(stale)
        if self.tagger is None:
            if not self._load_tagger():
                return False
            # Auto-download may just have supplied missing weight files.
            self.identity = model_identity(app, self.config, self.vocabulary)
            signature = digest({
                "model": self.identity,
                "cues": [[c.start, c.end, c.image, list(c.rect)] for c in cues],
                "media": media.hashes,
            })
        self._predict_scene(scene_id, paths, cues, media, signature, kinds, replacement)
        return True

    def _predict_scene(self, scene_id, paths, cues, media, signature, kinds, replacement):
        app, args, store = self.app, self.args, self.store
        if not args.resume:
            with store.connection:
                checkpoints.clear_checkpoint(store, scene_id, self.key)
        checkpoint = checkpoints.Checkpoint(
            store, scene_id, self.key,
            digest({"input": signature, "kinds": kinds, "scene_frames": 1}),
            len(cues), len(self.vocabulary.labels), args.checkpoint_batches)
        analysis = {}
        predictions, frames = prediction.predict_scene(
            app, self.tagger, media, paths, cues, raw_scene=True, checkpoint=checkpoint,
            prefetch=args.prefetch_batches, analysis=analysis, report=self.report,
            prepare_workers=args.prepare_workers)
        outputs = merge.capture_outputs(
            app, self.config, self.vocabulary, cues, predictions, frames, analysis)
        for segment in outputs.values():
            segment["meta"].update(
                inference_signature=signature,
                frame_input_signature=digest({
                    "cues": [[c.start, c.end, c.image, list(c.rect)] for c in cues],
                    "media": media.hashes,
                }))
        checkpoints.finish(store, scene_id, self.key, outputs, replacement)
        self.stats["processed"] += 1

    def _record_error(self, stage, scene_id, exc):
        self.stats["errors"] += 1
        self.report.issue(stage, self.key, scene_id, exc,
                          self.app.safe_error(exc, api_key=self.app.STASH_API_KEY))
