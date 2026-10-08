"""Reuse completed model inputs while other models resume the same scene."""
from collections import defaultdict

from stash_ai.inference.input_identity import cue_manifest, frame_signature
from stash_ai.storage import progress
from stash_ai.storage.input_checks import selected


class CachedInputPlan:
    """Defer media reads to unfinished passes; invalidate changed cached peers."""

    def __init__(self, store, scenes, kinds, runs, stats):
        self.store, self.kinds, self.stats = store, kinds, stats
        self.cached = defaultdict(set)
        self.frames = {}
        self.invalidated = defaultdict(set)
        deferred = selected(store, scenes)
        for scene, current in scenes.items():
            paths = current['paths']
            if scene in deferred or not paths.get('vtt') or not paths.get('sprite'):
                continue
            plans = {run.key: plan for run in runs
                     if (plan := run.cached_plan(scene, paths, kinds)) is not None}
            # Already conflicting completed results must all verify, just as
            # before. Missing results/checkpoints only affect their own model.
            if not plans or len({plan.frame_input for plan in plans.values()}) != 1:
                continue
            self.frames[scene] = {key: plan.frame_input for key, plan in plans.items()}
            for key in plans:
                self.cached[key].add(scene)

    def verified(self, scene, cues, hashes):
        """Block stale peers before new inference/checkpoints can be committed.

        Earlier cached passes are revisited after the ordinary model passes.
        Later passes simply lose their cached-input permission. Refresh requests
        survive interruption or failure, preventing publication of mixed inputs.
        """
        saved = self.frames.get(scene, {})
        if not saved:
            return
        current = frame_signature(cue_manifest(cues), hashes)
        for model, signature in list(saved.items()):
            if signature != current:
                progress.block_publication(self.store, [scene], model, self.kinds)
                self.stats['auto_refresh_requested'] += len(self.kinds)
                self.cached[model].discard(scene)
                self.invalidated[model].add(scene)
                del saved[model]

    def finished(self, model):
        # A pass already attempted these scenes; failures retain their durable
        # blocker and are retried by the next run, never repeatedly in this run.
        self.invalidated.pop(model, None)
