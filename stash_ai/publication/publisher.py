"""Schedule merged evidence and publish it through the parent process's writer."""
from __future__ import annotations

from stash_ai.runtime.console import progress as console_bar
from collections import Counter
from itertools import batched

from tqdm import tqdm

import stash_ai.storage.merge_checks as merge_checks
from stash_ai.evidence import pipeline as merge
from stash_ai.storage import progress
from stash_ai.publication import merged as merge_sync
from stash_ai.runtime.timing import stage


class MergePublisher:
    """Borrow one run's resources for its publication phase."""

    def __init__(self, app, args, store, stash, resolver, core_policies, scene_policies,
                 *, publication_key, stats, report):
        self.app, self.args = app, args
        self.store, self.stash, self.resolver = store, stash, resolver
        self.core_policies, self.scene_policies = core_policies, scene_policies
        self.publication_key, self.stats, self.report = publication_key, stats, report
        self.pending_refreshes = {}

    def run(self, scenes):
        refresh_kinds = progress.output_kinds(self.app)
        if (self.app.CREATE_MARKERS
                and self.app.TEMPORAL_POLICY.occurrence_markers):
            refresh_kinds = [*refresh_kinds, "scene"]
        self.pending_refreshes = progress.pending_refreshes(self.store, refresh_kinds, scenes)
        workers = self.args.merge_workers
        if self.app.DRY_RUN or self.args.profile_merge is not None:
            # Workers cannot see the temporary dry-run overlay. Profiling needs
            # per-stage timings from the parent, so it also remains serial.
            workers = 1
        if workers <= 1 or len(scenes) < 2:
            return self._serial(scenes)
        return self._parallel(scenes, workers)

    def _observed_scenes(self, scenes):
        """Refresh reuse candidates in bounded batches immediately before checks."""
        from stash_ai.publication.fingerprints import scene_fingerprint
        size = min(self.app.SCENE_PAGE_SIZE, 1000)
        for group in batched(scenes.items(), size):
            candidates = [scene_id for scene_id, current in group
                          if not self.args.force_merge and not self.pending_refreshes.get(str(scene_id))
                          and merge_checks.check_reason(self.store, scene_id, self.publication_key,
                                                        current['fingerprint']) == 'reused']
            refreshed, errors = {}, {}
            if candidates:
                with stage(self.report, 'reuse_refresh'):
                    try:
                        snapshots = self.stash.merge_snapshots(candidates)
                    except Exception as exc:
                        errors.update((scene_id, exc) for scene_id in candidates)
                    else:
                        for scene_id in candidates:
                            try:
                                if snapshots is None:
                                    observed = self.stash.find_scene_for_merge(scene_id)
                                else:
                                    observed = snapshots.get(str(scene_id))
                                if observed is None or str(observed['id']) != str(scene_id):
                                    raise RuntimeError(f'Scene {scene_id} no longer exists or returned an inconsistent ID.')
                                refreshed[scene_id] = scene_fingerprint(observed)
                            except Exception as exc:
                                errors[scene_id] = exc
            for scene_id, current in group:
                yield scene_id, ({**current, 'fingerprint': refreshed[scene_id]}
                                 if scene_id in refreshed else current), errors.get(scene_id)

    def _reason(self, scene_id, current, refresh_error=None):
        self.store.initialization.require_ready(scene_id)
        from stash_ai.storage.input_checks import pending as unverified_inputs
        deferred = unverified_inputs(self.store, scene_id)
        if deferred:
            self.stats['merge_input_unverified'] += 1
            raise RuntimeError('Input verification pending for ' +
                               ', '.join(f'{model}/{reason}' for model, reason in sorted(deferred.items())) +
                               '; existing results retained. Restore media and resume to verify reuse.')
        pending = self.pending_refreshes.get(str(scene_id), ())
        if pending:
            self.stats["merge_refresh_blocked"] += 1
            details = ", ".join(f"{model}/{kind}" for model, kind in pending)
            raise RuntimeError(
                "Fresh inference is still pending for " + details +
                "; merge skipped to avoid reusing stale evidence. "
                "Rerun without --merge-only to resume inference.")
        if refresh_error is not None:
            raise refresh_error
        with stage(self.report, 'publication_check'):
            reason = (
                'forced' if self.args.force_merge
                else merge_checks.check_reason(
                    self.store, scene_id, self.publication_key, current['fingerprint']))
        self.stats['merge_' + reason] += 1
        return reason

    def _issue(self, scene_id, exc):
        self.stats["errors"] += 1
        if self.report.dry_run_plan is not None:
            self.report.dry_run_plan.write(dict(type='status', scene_id=str(scene_id), status='error',
                reason=self.app.services.safe_error(exc, api_key=self.app.STASH_API_KEY)))
        self.report.issue(
            'merge', '', scene_id, exc,
            self.app.services.safe_error(exc, api_key=self.app.STASH_API_KEY))

    def _merge_one(self, scene_id, current, prepared=None):
        """Publish local or worker evidence with the same accounting and errors."""
        try:
            if prepared is not None:
                if prepared.scene_id != str(scene_id):
                    raise RuntimeError(
                        f"Merge worker returned scene {prepared.scene_id!r} while {scene_id!r} was expected.")
                if prepared.error is not None:
                    raise RuntimeError(prepared.error)
                if prepared.payload is None:
                    raise RuntimeError('Merge worker returned no evidence.')
                scene_hits, spans = prepared.payload.scene_hits, prepared.payload.spans
                markers, diagnostics = prepared.payload.markers, prepared.payload.diagnostics
            cache_before = self.store.cache.stats.copy()
            if prepared is None:
                diagnostics = Counter()
                with self.report.measure('merge_compute'):
                    scene_hits, spans = merge.evidence(
                        self.app, self.store, scene_id, self.core_policies,
                        self.scene_policies, diagnostics)
            with self.report.measure('merge_plan'):
                result = self._sync(current, scene_hits, spans,
                                    prepared_markers=markers if prepared is not None else None)
            self.stats.update(result)
            self.report.record_counts(**{
                'cache_' + key: value - cache_before[key]
                for key, value in self.store.cache.stats.items()
            })
            self.stats.update(diagnostics)
            self.stats["scenes_synced"] += 1
            if prepared is not None:
                self.stats["merge_parallel_scenes"] += 1
            return result + diagnostics
        except Exception as exc:
            self._issue(scene_id, exc)
            return None

    def _sync(self, current, scene_hits, spans, *, prepared_markers=None):
        return merge_sync.sync_merged(
            self.app, self.stash, self.resolver, self.store, current, scene_hits, spans,
            report=self.report, tag_policies=self.scene_policies,
            publication_key=self.publication_key, prepared_markers=prepared_markers)

    def _serial(self, scenes):
        bar = console_bar(tqdm, self._observed_scenes(scenes), total=len(scenes), desc="Merge to ai:", unit="scene")
        for scene_id, current, refresh_error in bar:
            with self.report.diagnostics.scene(scene_id) as sample:
                try:
                    reason = self._reason(scene_id, current, refresh_error)
                    sample['reason'] = reason
                    if reason == 'reused':
                        sample['status'] = 'reused'
                        if self.report.dry_run_plan is not None:
                            self.report.dry_run_plan.write(dict(type='status', scene_id=str(scene_id),
                                                               status='reused', reason='current_publication'))
                    else:
                        statistics = self._merge_one(scene_id, current)
                        if statistics is None:
                            sample['status'] = 'error'
                        else:
                            sample['statistics'] = dict(statistics)
                except Exception as exc:
                    sample['status'] = 'error'
                    self._issue(scene_id, exc)
                self.report.progress(bar, self.stats)

    def _parallel(self, scenes, workers):
        """Prepare evidence in read-only processes; publish serially here."""
        from stash_ai.publication.parallel import iter_parallel_evidence
        from itertools import zip_longest

        pending = []
        bar = console_bar(tqdm, total=len(scenes), desc="Merge to ai:", unit="scene")
        try:
            # Check in the writer process before scheduling expensive replay.
            for scene_id, current, refresh_error in self._observed_scenes(scenes):
                try:
                    if self._reason(scene_id, current, refresh_error) != 'reused':
                        pending.append((scene_id, current))
                        continue
                except Exception as exc:
                    self._issue(scene_id, exc)
                bar.update(1)
                self.report.progress(bar, self.stats)

            if not pending:
                return
            workers = min(workers, len(pending))
            if workers <= 1:
                # Avoid process startup when only one scene needs publication.
                for scene_id, current in pending:
                    self._merge_one(scene_id, current)
                    bar.update(1)
                    self.report.progress(bar, self.stats)
                return

            # Inference/scan caches are not shared with spawned processes.
            self.store.cache.clear()
            cache_share = int(self.args.ram_cache_gb * 2**30) // (workers + 1)
            budget_share = max(1, int(self.args.ram_budget_gb * 2**30) // (workers + 1))
            with self.store.cache.temporary_budget(
                    limit=min(self.store.cache.limit, cache_share), rss_budget=budget_share):
                evidence = iter_parallel_evidence(
                    self.app, self.args.state_db, self.store.scope,
                    [scene_id for scene_id, _ in pending],
                    self.core_policies, self.scene_policies,
                    workers=workers,
                    cache_bytes=int(self.args.ram_cache_gb * 2**30),
                    memory_budget_bytes=int(self.args.ram_budget_gb * 2**30),
                    **({'profile': True} if self.report.profile is not None else {}),
                )
                completed = 0
                try:
                    missing = object()
                    for item, prepared in zip_longest(pending, evidence, fillvalue=missing):
                        if item is missing or prepared is missing:
                            raise RuntimeError('Merge worker returned incomplete or extra scene results.')
                        scene_id, current = item
                        if prepared.scene_id != str(scene_id):
                            raise RuntimeError('Merge worker returned an inconsistent scene ID.')
                        if self.report.profile is not None and prepared.metrics is not None:
                            self.report.profile.ingest(prepared.metrics)
                        self._merge_one(scene_id, current, prepared)
                        completed += 1
                        bar.update(1)
                        self.report.progress(bar, self.stats)
                except Exception as exc:
                    # Never retry a second merge pass after a pool failure: some
                    # scenes may already have been published by the parent.
                    remaining = len(pending) - completed
                    self.stats["errors"] += max(1, remaining)
                    self.report.issue(
                        'merge workers', '', None, exc,
                        self.app.services.safe_error(exc, api_key=self.app.STASH_API_KEY))
                    bar.update(remaining)
                    self.report.progress(bar, self.stats)
                finally:
                    evidence.close()
        finally:
            bar.close()
