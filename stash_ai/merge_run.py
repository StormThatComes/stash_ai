"""Merged-output run orchestration.

The compatibility entry points remain in :mod:`stash_ai.merge`; this module owns
the stateful lifecycle of one merged run so the merge algorithms stay function-oriented.
"""
from __future__ import annotations

from collections import Counter

from tqdm import tqdm

from stash_ai import merge
from stash_ai import performance
from stash_ai import progress
from stash_ai.runtime import RunReport, stage


class MergedRun:
    """Own the resources and phases of one merged-output execution."""

    def __init__(self, app, settings, args, catalogs, hierarchy, core_policies, scene_policies):
        self.app = app
        self.settings = settings
        self.args = args
        self.catalogs = catalogs
        self.hierarchy = hierarchy
        self.core_policies = core_policies
        self.scene_policies = scene_policies
        self.stats = Counter()
        self.report = getattr(args, 'report', None) or RunReport(getattr(args, 'verbose', False), tqdm.write)
        self.store = None
        self.stash = None
        self.resolver = None
        self.scenes = {}
        self.use_catalog = False
        self.base_key = None
        self.merge_key = None

    def run(self):
        app, args = self.app, self.args
        with merge.StateStore(
                args.state_db, merge.scope_for(app.STASH_URL), app.DRY_RUN,
                cache_bytes=int(getattr(args, 'ram_cache_gb', 0) * 2**30),
                allow_cleanup=getattr(args, 'abandon_pending', False),
                memory_budget_bytes=int(getattr(args, 'ram_budget_gb', 32) * 2**30)) as store:
            self.store = store
            store.report = self.report
            if getattr(args, 'abandon_pending', False):
                from stash_ai.recovery import abandon
                self.stats.update(abandon(store, self.report.write))
            self.stash = app.create_stash_client(app.DRY_RUN)
            self.stash.report = self.report
            try:
                self._configure_stash()
                self._prepare_resolver()
                self._scan_scenes()
                threshold_result = self._apply_thresholds()
                if threshold_result is not None:
                    return threshold_result
                self._run_inference()
                self._merge_scenes()
                catalog_pruned = self._prune_outputs()
                self._finalize_catalog_key(refresh=catalog_pruned)
            finally:
                self.stash.close()
            self.report.finish('Merged' + (' (planned)' if app.DRY_RUN else ''), self.stats, store.cache)
        return int(bool(self.stats["errors"]))

    def _configure_stash(self):
        app, stash, args = self.app, self.stash, self.args
        if self.hierarchy or app.CREATE_MARKERS:
            stash.detect_tag_hierarchy()
        if app.CREATE_MARKERS:
            stash.detect_marker_capabilities()
            if not stash.has_end_time:
                print("Stash supports point markers only; full merged ranges remain in the source state.")
        if hasattr(stash, 'configure_merge_reads'):
            stash.configure_merge_reads(self.report, merge_only=args.merge_only)

    def _prepare_resolver(self):
        app, args, stash = self.app, self.args, self.stash
        self.resolver = app.TagResolver(
            stash, self.hierarchy, source=merge.AI, report=self.report.detail,
            create_missing=app.CREATE_MISSING)
        self.use_catalog = bool(self.hierarchy or app.CREATE_MARKERS)
        if self.use_catalog or not args.scene_id or len(args.scene_id) > 1:
            self.resolver.preload()
        self.base_key = performance.publication_key(
            app, args, self.hierarchy, self.core_policies, self.scene_policies, stash)
        self.merge_key = (
            performance.digest([self.base_key, performance.catalog_fingerprint(self.resolver)])
            if self.use_catalog else self.base_key)
        self.store.preload_sources()

    def _scan_scenes(self):
        app, args = self.app, self.args
        with stage(self.report, 'initial_scan'):
            for current in tqdm(self.stash.iter_scenes(args.scene_id), desc="Read sources", unit="scene"):
                scene_id = str(current["id"])
                try:
                    with stage(self.report, 'scan_fingerprint'):
                        fingerprint = performance.scene_fingerprint(current)
                    publication_reason = performance.check_reason(
                        self.store, scene_id, self.merge_key, fingerprint)
                    if publication_reason == 'reused':
                        self.stats['bootstrap_reused'] += 1
                    else:
                        with self.report.measure('bootstrap'):
                            merge.bootstrap_legacy(app, self.store, current)
                    # Keep only input paths/ID between passes; marker bodies need
                    # no collection-sized in-memory cache and are refreshed at sync.
                    self.scenes[scene_id] = {
                        "id": scene_id,
                        "paths": current.get("paths") or {},
                        'fingerprint': fingerprint,
                        'publication_reason': publication_reason,
                    }
                except Exception as exc:
                    self.stats["errors"] += 1
                    self.report.issue(
                        'read sources', '', scene_id, exc,
                        app.safe_error(exc, api_key=app.STASH_API_KEY))

    def _apply_thresholds(self):
        args, app = self.args, self.app
        if not (getattr(args, 'rethreshold_markers', False) or getattr(args, 'rethreshold_scenes', False)):
            return None
        from stash_ai.thresholds import rethreshold
        if self.stats['errors']:
            self.report.finish('Threshold update aborted: 0 changes applied', self.stats, self.store.cache)
            return 1
        if getattr(args, 'rethreshold_scenes', False):
            from stash_ai.scene_cache import rethreshold as rethreshold_scenes
            self.stats.update(rethreshold_scenes(
                app, self.store, self.scenes, self.settings, self.report,
                markers=args.rethreshold_markers))
        else:
            self.stats.update(rethreshold(
                app, self.store, self.scenes, self.settings, self.report))
        if self.stats['errors']:
            self.report.finish('Thresholds (not applied)', self.stats, self.store.cache)
            return 1
        return None

    def _run_inference(self):
        app, args, store = self.app, self.args, self.store
        kinds = progress.output_kinds(app)
        replacement = {"scene": app.REPLACE_SCENE_TAGS, "markers": app.REPLACE_MARKERS}
        inputs = progress.RunInputs(self.stash, store.cache, self.report)
        if args.merge_only:
            return
        if args.restart:
            progress.request_refresh(
                store, self.scenes, [config.source.key for config in self.settings], kinds)
        progress.preload_status(store)
        for config in self.settings:
            self._run_model(config, kinds, replacement, inputs)

    def _run_model(self, config, kinds, replacement, inputs):
        app, args, store = self.app, self.args, self.store
        key = config.source.key
        tagger = None
        identity = progress.model_identity(app, config, self.catalogs[key])
        try:
            bar = tqdm(self.scenes.items(), desc=key + " inference", unit="scene")
            for scene_id, current in bar:
                self.report.progress(bar, self.stats)
                paths = current["paths"]
                if not paths.get("sprite") or not paths.get("vtt"):
                    self.stats["skipped"] += 1
                    continue
                if app.SKIP_EXISTING_OUTPUTS and not args.restart and merge.has_outputs(app, store, scene_id, key):
                    self.stats["skipped"] += 1
                    continue
                try:
                    store.cache.pressure(force=True)
                    cues, media, signature = inputs.inspect(app, paths, identity)
                    if args.resume and not args.restart and current['publication_reason'] != 'reused':
                        collapsed = progress.reuse_replacement_policy(
                            store, scene_id, key, kinds, signature, replacement)
                        if collapsed is not None:
                            self.stats["replacement_history_collapsed"] += collapsed
                            self.stats["reused"] += 1
                            continue
                    if args.resume and progress.completed(store, scene_id, key, kinds, signature):
                        self.stats["reused"] += 1
                        continue
                    if args.resume and not args.restart:
                        from stash_ai.thresholds import reuse
                        if reuse(app, store, scene_id, config, identity, cues, media.hashes, kinds):
                            if current['publication_reason'] != 'reused':
                                collapsed = progress.collapse_replacement_history(
                                    store, scene_id, key, kinds, replacement)
                                if collapsed is None:
                                    raise RuntimeError(
                                        f'{scene_id}/{key}: threshold cache did not leave a current inference segment.')
                                self.stats["replacement_history_collapsed"] += collapsed
                            self.stats['threshold_cache_reused'] += 1
                            continue
                    if tagger is None:
                        try:
                            tagger = app.load_tagger(config, self.catalogs[key], app.prepare_image)
                        except Exception as exc:
                            self.stats["errors"] += 1
                            self.report.issue(
                                'load model', key, None, exc,
                                app.safe_error(exc, api_key=app.STASH_API_KEY))
                            break
                        # Auto-download may just have supplied missing weight files.
                        identity = progress.model_identity(app, config, self.catalogs[key])
                        signature = progress.digest({
                            "model": identity,
                            "cues": [[c.start, c.end, c.image, list(c.rect)] for c in cues],
                            "media": media.hashes,
                        })
                    if not args.resume:
                        with store.connection:
                            progress.clear_checkpoint(store, scene_id, key)
                    checkpoint = progress.Checkpoint(
                        store, scene_id, key,
                        progress.digest({"input": signature, "kinds": kinds, "scene_frames": 1}),
                        len(cues), len(self.catalogs[key].labels), args.checkpoint_batches)
                    analysis = {}
                    predictions, frames = app.predict_scene(
                        tagger, media, paths, cues, raw_scene=True, checkpoint=checkpoint,
                        prefetch=args.prefetch_batches, analysis=analysis, report=self.report,
                        prepare_workers=getattr(args, 'prepare_workers', 1))
                    outputs = merge.capture_outputs(
                        app, config, self.catalogs[key], cues, predictions, frames, analysis)
                    for segment in outputs.values():
                        segment["meta"].update(
                            inference_signature=signature,
                            frame_input_signature=progress.digest({
                                "cues": [[c.start, c.end, c.image, list(c.rect)] for c in cues],
                                "media": media.hashes,
                            }))
                    progress.finish(store, scene_id, key, outputs, replacement)
                    self.stats["processed"] += 1
                except Exception as exc:
                    self.stats["errors"] += 1
                    self.report.issue(
                        'inference', key, scene_id, exc,
                        app.safe_error(exc, api_key=app.STASH_API_KEY))
        finally:
            if tagger is not None:
                tagger.close()

    def _merge_scenes(self):
        # Profiling records per-stage timings in the parent process, so keep that
        # diagnostic mode serial even when the normal merge uses CPU workers.
        workers = int(getattr(self.args, 'merge_workers', 1))
        if self.app.DRY_RUN or getattr(self.args, 'profile_merge', None) is not None:
            # Dry-run changes live in this process's temporary SQLite overlay,
            # which isolated workers cannot observe. Profiling also needs parent
            # process stage timings, so both modes intentionally remain serial.
            workers = 1
        if workers <= 1 or len(self.scenes) < 2:
            return self._merge_scenes_serial()
        return self._merge_scenes_parallel(workers)

    def _merge_scenes_serial(self):
        app, args, store = self.app, self.args, self.store
        bar = tqdm(self.scenes.items(), desc="Merge to ai:", unit="scene")
        for scene_id, current in bar:
            with self.report.diagnostics.scene(scene_id) as sample:
                try:
                    with stage(self.report, 'publication_check'):
                        reason = (
                            'forced' if getattr(args, 'force_merge', False)
                            else performance.check_reason(
                                store, scene_id, self.merge_key, current['fingerprint']))
                    sample['reason'] = reason
                    if reason == 'reused':
                        sample['status'] = 'reused'
                        self.stats['merge_reused'] += 1
                        self.report.progress(bar, self.stats)
                        continue
                    self.stats['merge_' + reason] += 1
                    diagnostics = Counter()
                    cache_before = store.cache.stats.copy()
                    with self.report.measure('merge_compute'):
                        scene_hits, spans = merge.evidence(
                            app, store, scene_id, self.core_policies, self.scene_policies, diagnostics)
                    with self.report.measure('merge_plan'):
                        result = merge.sync_merged(
                            app, self.stash, self.resolver, store, current, scene_hits, spans,
                            report=self.report, tag_policies=self.scene_policies,
                            publication_key=self.merge_key)
                    self.stats.update(result)
                    sample['statistics'] = dict(result + diagnostics)
                    self.report.record_counts(**{
                        'cache_' + key: value - cache_before[key]
                        for key, value in store.cache.stats.items()
                    })
                    self.stats.update(diagnostics)
                    self.stats["scenes_synced"] += 1
                except Exception as exc:
                    sample['status'] = 'error'
                    self.stats["errors"] += 1
                    self.report.issue(
                        'merge', '', scene_id, exc,
                        app.safe_error(exc, api_key=app.STASH_API_KEY))
                self.report.progress(bar, self.stats)

    def _merge_scenes_parallel(self, workers):
        """Prepare evidence in read-only processes; publish serially in this process."""
        from stash_ai.merge_parallel import iter_parallel_evidence

        app, args, store = self.app, self.args, self.store
        pending = []
        bar = tqdm(total=len(self.scenes), desc="Merge to ai:", unit="scene")
        try:
            # Publication checks stay in the writer process and avoid scheduling
            # expensive evidence replay for scenes whose published fingerprint is current.
            for scene_id, current in self.scenes.items():
                try:
                    with stage(self.report, 'publication_check'):
                        reason = (
                            'forced' if getattr(args, 'force_merge', False)
                            else performance.check_reason(
                                store, scene_id, self.merge_key, current['fingerprint']))
                    if reason == 'reused':
                        self.stats['merge_reused'] += 1
                        bar.update(1)
                        self.report.progress(bar, self.stats)
                        continue
                    self.stats['merge_' + reason] += 1
                    pending.append((scene_id, current, reason))
                except Exception as exc:
                    self.stats["errors"] += 1
                    self.report.issue(
                        'merge', '', scene_id, exc,
                        app.safe_error(exc, api_key=app.STASH_API_KEY))
                    bar.update(1)
                    self.report.progress(bar, self.stats)

            if not pending:
                return

            workers = min(workers, len(pending))
            if workers <= 1:
                # A collection can contain many reused scenes but only one changed
                # scene; avoid process startup overhead in that case.
                for scene_id, current, _ in pending:
                    diagnostics = Counter()
                    try:
                        cache_before = store.cache.stats.copy()
                        with self.report.measure('merge_compute'):
                            scene_hits, spans = merge.evidence(
                                app, store, scene_id, self.core_policies, self.scene_policies, diagnostics)
                        with self.report.measure('merge_plan'):
                            result = merge.sync_merged(
                                app, self.stash, self.resolver, store, current, scene_hits, spans,
                                report=self.report, tag_policies=self.scene_policies,
                                publication_key=self.merge_key)
                        self.stats.update(result)
                        self.report.record_counts(**{
                            'cache_' + key: value - cache_before[key]
                            for key, value in store.cache.stats.items()
                        })
                        self.stats.update(diagnostics)
                        self.stats["scenes_synced"] += 1
                    except Exception as exc:
                        self.stats["errors"] += 1
                        self.report.issue(
                            'merge', '', scene_id, exc,
                            app.safe_error(exc, api_key=app.STASH_API_KEY))
                    bar.update(1)
                    self.report.progress(bar, self.stats)
                return

            # Evidence caches from inference/scan are not shared with spawned
            # processes. Free them before allocating bounded worker-local caches.
            store.cache.clear()
            evidence = iter_parallel_evidence(
                app,
                args.state_db,
                store.scope,
                [scene_id for scene_id, _, _ in pending],
                self.core_policies,
                self.scene_policies,
                workers=workers,
                cache_bytes=int(getattr(args, 'ram_cache_gb', 0) * 2**30),
                memory_budget_bytes=int(getattr(args, 'ram_budget_gb', 32) * 2**30),
            )
            completed = 0
            try:
                for (scene_id, current, _), prepared in zip(pending, evidence):
                    worker_scene, scene_hits, spans, diagnostics, error = prepared
                    try:
                        if worker_scene != scene_id:
                            raise RuntimeError(
                                f"Merge worker returned scene {worker_scene!r} while {scene_id!r} was expected.")
                        if error is not None:
                            raise RuntimeError(error)
                        cache_before = store.cache.stats.copy()
                        with self.report.measure('merge_plan'):
                            result = merge.sync_merged(
                                app, self.stash, self.resolver, store, current, scene_hits, spans,
                                report=self.report, tag_policies=self.scene_policies,
                                publication_key=self.merge_key)
                        self.stats.update(result)
                        self.report.record_counts(**{
                            'cache_' + key: value - cache_before[key]
                            for key, value in store.cache.stats.items()
                        })
                        self.stats.update(diagnostics)
                        self.stats["scenes_synced"] += 1
                        self.stats["merge_parallel_scenes"] += 1
                    except Exception as exc:
                        self.stats["errors"] += 1
                        self.report.issue(
                            'merge', '', scene_id, exc,
                            app.safe_error(exc, api_key=app.STASH_API_KEY))
                    completed += 1
                    bar.update(1)
                    self.report.progress(bar, self.stats)
            except Exception as exc:
                # A broken worker pool is a run-level failure. Do not fall back to
                # a second merge pass after some scenes may already be published.
                remaining = len(pending) - completed
                self.stats["errors"] += max(1, remaining)
                self.report.issue(
                    'merge workers', '', None, exc,
                    app.safe_error(exc, api_key=app.STASH_API_KEY))
                bar.update(remaining)
                self.report.progress(bar, self.stats)
        finally:
            bar.close()

    def _finalize_catalog_key(self, *, refresh=False):
        if not self.use_catalog:
            return
        resolver = self.resolver
        if refresh:
            # Pruning mutates Stash after the resolver's initial catalog snapshot.
            # Refresh only after an actual delete so merge_checks describe the
            # catalog the next run will observe without adding a read to the
            # unchanged hot path.
            resolver = self.app.TagResolver(
                self.stash, self.hierarchy, source=merge.AI,
                create_missing=self.app.CREATE_MISSING)
            resolver.preload()
        final_key = performance.digest([
            self.base_key, performance.catalog_fingerprint(resolver)])
        if final_key != self.merge_key:
            with self.store.connection:
                self.store.connection.execute(
                    'UPDATE merge_checks SET policy=? WHERE scope=? AND policy=?',
                    (final_key, self.store.scope, self.merge_key))

    def _prune_sql(self):
        """Prepare optional discovery before pruning; native checks still authorize deletes."""
        from stash_ai.cleanup import CleanupClient
        from stash_ai.cleanup_sql import SQLCleanup
        try:
            client = CleanupClient(self.stash)
            if client.sql_available:
                sql = SQLCleanup(client, self.args, self.stats, self.report)
                sql.probe()
                # Share one fresh, lightweight catalog between ordinary tags and
                # marker groups, including tags created during this merge.
                sql.catalog = client.catalog()
                self.stats['merge_prune_sql_enabled'] += 1
                return sql
        except Exception as exc:
            self.report.detail('SQL prune unavailable; using GraphQL: ' +
                               self.app.safe_error(exc, api_key=self.app.STASH_API_KEY))
        self.stats['merge_prune_graphql_fallback'] += 1
        return None

    def _prune_outputs(self):
        app, args, stash = self.app, self.args, self.stash
        app.print_hierarchy_summary(self.resolver)
        if self.stats['errors'] or not (args.prune_empty_tags or app.CREATE_MARKERS):
            return False
        pruned_before = self.stats['tags_pruned'] + self.stats['marker_group_tags_pruned']
        sql = self._prune_sql()
        if args.prune_empty_tags:
            from stash_ai.prune import prune_tags
            # Groups use the projected marker counts below, including in dry-run.
            self.stats.update(sql.prune(prefixes=args.prune_prefixes, skip_marker_groups=app.CREATE_MARKERS)
                              if sql is not None else prune_tags(
                                  stash, args.prune_prefixes, report=self.report.detail,
                                  skip_marker_groups=app.CREATE_MARKERS))
        elif sql is not None and app.APPLY_SCENE_TAGS and app.CREATE_MARKERS:
            self.stats.update(sql.prune(self.scene_policies))
        else:
            from stash_ai.exclusions import prune_excluded_definitions
            self.stats.update(prune_excluded_definitions(
                app, stash, self.scene_policies, self.report.detail))
        if app.CREATE_MARKERS:
            from stash_ai.marker_groups import for_resolver, prune_groups
            delta = for_resolver(app, self.resolver).usage_delta
            candidates = sql.marker_group_candidates(delta) if sql is not None else None
            self.stats.update(prune_groups(
                stash, self.report.detail, candidate_ids=candidates, usage_delta=delta))
        pruned_after = self.stats['tags_pruned'] + self.stats['marker_group_tags_pruned']
        return not stash.dry_run and pruned_after > pruned_before
