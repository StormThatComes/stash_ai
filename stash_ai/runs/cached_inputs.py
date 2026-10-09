"""Reuse completed model inputs while other models resume the same scene."""
from collections import defaultdict
from itertools import batched

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
        self.plans = {}
        from stash_ai.storage.metadata import backfill_batch
        from stash_ai.inference.identity_batch import identity_batch
        pairs = ((scene, current, run) for scene, current in scenes.items()
                 if scene not in deferred and current['paths'].get('vtt') and current['paths'].get('sprite')
                 for run in runs)
        candidates, current_scene = {}, None
        for group in batched(pairs, 128):
            keys = [(scene, run.key) for scene, _, run in group]
            with backfill_batch(store), store.metadata.batch(keys), identity_batch(store, keys):
                for scene, current, run in group:
                    if current_scene is not None and scene != current_scene:
                        self._accept(current_scene, candidates)
                        candidates = {}
                    current_scene = scene
                    plan = run.cached_plan(scene, current['paths'], kinds)
                    if plan is not None:
                        candidates[run.key] = plan
        if current_scene is not None:
            self._accept(current_scene, candidates)

    def _accept(self, scene, plans):
        # Conflicting completed inputs must verify; unfinished peers do not
        # remove ready models' non-fetching authorization.
        if not plans or len({plan.frame_input for plan in plans.values()}) != 1:
            return
        self.frames[scene] = {key: plan.frame_input for key, plan in plans.items()}
        for key, plan in plans.items():
            self.cached[key].add(scene)
            self.plans[scene, key] = plan

    def plan(self, scene, model):
        return self.plans.get((scene, model)) if scene in self.cached[model] else None

    def verified(self, scene, cues, hashes, *, snapshot=None):
        """Block stale peers before new inference/checkpoints can be committed.

        Earlier cached passes are revisited after the ordinary model passes.
        Later passes simply lose their cached-input permission. Refresh requests
        survive interruption or failure, preventing publication of mixed inputs.
        """
        saved = self.frames.get(scene, {})
        if not saved:
            return
        current = snapshot.frame_signature if snapshot is not None else frame_signature(cue_manifest(cues), hashes)
        for model, signature in list(saved.items()):
            if signature != current:
                progress.block_publication(self.store, [scene], model, self.kinds)
                self.stats['auto_refresh_requested'] += len(self.kinds)
                self.cached[model].discard(scene)
                self.plans.pop((scene, model), None)
                self.invalidated[model].add(scene)
                del saved[model]

    def finished(self, model):
        # A pass already attempted these scenes; failures retain their durable
        # blocker and are retried by the next run, never repeatedly in this run.
        self.invalidated.pop(model, None)
        for scene in self.cached[model]:
            self.plans.pop((scene, model), None)
        self.cached[model].clear()
