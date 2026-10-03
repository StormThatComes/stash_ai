"""Merged-output run orchestration.

This module owns the stateful lifecycle of one merged run and calls
merge reconciliation directly; merge algorithms stay function-oriented.
"""
from __future__ import annotations

from functools import partial
from collections import Counter

from tqdm import tqdm

from stash_ai import inference
from stash_ai.inference_inputs import RunInputs
from stash_ai.merge_publish import MergePublisher
from stash_ai import merge
from stash_ai import performance
from stash_ai import progress
from stash_ai.runtime import stage


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
        self.report = args.report
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
                cache_bytes=int(args.ram_cache_gb * 2**30),
                allow_cleanup=args.abandon_pending,
                memory_budget_bytes=int(args.ram_budget_gb * 2**30)) as store:
            self.store = store
            store.report = self.report
            if args.abandon_pending:
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
        stash, args = self.stash, self.args
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
            app, args, self.hierarchy, self.core_policies, self.scene_policies)
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
        if not (args.rethreshold_markers or args.rethreshold_scenes):
            return None
        from stash_ai.thresholds import rethreshold
        if self.stats['errors']:
            self.report.finish('Threshold update aborted: 0 changes applied', self.stats, self.store.cache)
            return 1
        if args.rethreshold_scenes:
            from stash_ai.scene_cache import rethreshold as rethreshold_scenes
            self.stats.update(rethreshold_scenes(
                app, self.store, self.scenes, self.settings, self.report,
                markers=args.rethreshold_markers))
        else:
            self.stats.update(rethreshold(
                app, self.store, self.scenes, self.settings, self.report))
        if self.stats['errors']:
            self.report.finish('Threshold update aborted: 0 changes applied', self.stats, self.store.cache)
            return 1
        return None

    def _run_inference(self):
        app, args, store = self.app, self.args, self.store
        kinds = progress.output_kinds(app)
        replacement = {"scene": app.REPLACE_SCENE_TAGS, "markers": app.REPLACE_MARKERS}
        inputs = RunInputs(self.stash, store.cache, self.report)
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
                    if not args.restart:
                        stale = progress.request_stale_refresh(
                            store, scene_id, key, kinds, signature)
                        self.stats['auto_refresh_requested'] += len(stale)
                    if tagger is None:
                        try:
                            tagger = app.load_tagger(config, self.catalogs[key], partial(inference.prepare_image, app))
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
                    predictions, frames = inference.predict_scene(
                        app, tagger, media, paths, cues, raw_scene=True, checkpoint=checkpoint,
                        prefetch=args.prefetch_batches, analysis=analysis, report=self.report,
                        prepare_workers=args.prepare_workers)
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
        MergePublisher(
            self.app, self.args, self.store, self.stash, self.resolver,
            self.core_policies, self.scene_policies,
            publication_key=self.merge_key, stats=self.stats, report=self.report,
        ).run(self.scenes)

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
            sql = SQLCleanup(client, self.args, self.stats, self.report)
            sql.session.probe()
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
