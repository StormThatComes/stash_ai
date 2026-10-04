"""Merged-output run orchestration.

This module owns the stateful lifecycle of one merged run and calls
merge reconciliation directly; merge algorithms stay function-oriented.
"""
from __future__ import annotations

from collections import Counter

from tqdm import tqdm

from stash_ai.commands import legacy as legacy_commands
from stash_ai.signatures import digest
from stash_ai.inference.inputs import RunInputs
from stash_ai.publication.publisher import MergePublisher
import stash_ai.evidence.labels as merge_labels
import stash_ai.storage.store as merge_store
from stash_ai.publication import fingerprints
import stash_ai.storage.merge_checks as merge_checks
from stash_ai.storage import progress
from stash_ai.runtime.timing import stage
from .merged_model import MergedModelRun


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
        with merge_store.StateStore(
                args.state_db, merge_store.scope_for(app.STASH_URL), app.DRY_RUN,
                cache_bytes=int(args.ram_cache_gb * 2**30),
                allow_cleanup=args.abandon_pending,
                memory_budget_bytes=int(args.ram_budget_gb * 2**30)) as store:
            self.store = store
            store.report = self.report
            if args.abandon_pending:
                from stash_ai.storage.recovery import abandon
                self.stats.update(abandon(store, self.report.write))
            self.stash = app.services.create_stash_client(app.DRY_RUN)
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
        self.resolver = app.services.TagResolver(
            stash, self.hierarchy, source=merge_labels.AI, report=self.report.detail,
            create_missing=app.CREATE_MISSING)
        self.use_catalog = bool(self.hierarchy or app.CREATE_MARKERS)
        if self.use_catalog or not args.scene_id or len(args.scene_id) > 1:
            self.resolver.preload()
        self.base_key = fingerprints.publication_key(
            app, args, self.hierarchy, self.core_policies, self.scene_policies)
        self.merge_key = (
            digest([self.base_key, fingerprints.catalog_fingerprint(self.resolver)])
            if self.use_catalog else self.base_key)
        self.store.preload_sources()

    def _scan_scenes(self):
        app, args = self.app, self.args
        with stage(self.report, 'initial_scan'):
            for current in tqdm(self.stash.iter_scenes(args.scene_id), desc="Read sources", unit="scene"):
                scene_id = str(current["id"])
                try:
                    with stage(self.report, 'scan_fingerprint'):
                        fingerprint = fingerprints.scene_fingerprint(current)
                    publication_reason = merge_checks.check_reason(
                        self.store, scene_id, self.merge_key, fingerprint)
                    if publication_reason == 'reused':
                        self.stats['source_check_reused'] += 1
                    else:
                        with self.report.measure('source_check'):
                            legacy_commands.check_sources(app, self.store, current)
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
                        app.services.safe_error(exc, api_key=app.STASH_API_KEY))

    def _apply_thresholds(self):
        args, app = self.args, self.app
        if not (args.rethreshold_markers or args.rethreshold_scenes):
            return None
        from stash_ai.rethreshold.execution import run
        if self.stats['errors']:
            self.report.finish('Threshold update aborted: 0 changes applied', self.stats, self.store.cache)
            return 1
        self.stats.update(run(
            app, self.store, self.scenes, self.settings, self.report,
            scenes=args.rethreshold_scenes, markers=args.rethreshold_markers))
        if self.stats['errors']:
            self.report.finish('Threshold update aborted: 0 changes applied', self.stats, self.store.cache)
            return 1
        return None

    def _run_inference(self):
        app, args, store = self.app, self.args, self.store
        kinds = progress.output_kinds(app)
        inputs = RunInputs(self.stash, store.cache, self.report)
        if args.merge_only:
            return
        if args.restart:
            progress.request_refresh(
                store, self.scenes, [config.source.key for config in self.settings], kinds)
        progress.preload_status(store)
        for config in self.settings:
            MergedModelRun(
                app, args, store, config, self.catalogs[config.source.key],
                stats=self.stats, report=self.report,
            ).run(self.scenes, kinds, inputs)

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
            resolver = self.app.services.TagResolver(
                self.stash, self.hierarchy, source=merge_labels.AI,
                create_missing=self.app.CREATE_MISSING)
            resolver.preload()
        final_key = digest([
            self.base_key, fingerprints.catalog_fingerprint(resolver)])
        if final_key != self.merge_key:
            with self.store.connection:
                self.store.connection.execute(
                    'UPDATE merge_checks SET policy=? WHERE scope=? AND policy=?',
                    (final_key, self.store.scope, self.merge_key))

    def _prune_sql(self):
        """Prepare optional discovery before pruning; native checks still authorize deletes."""
        from stash_ai.cleanup.client import CleanupClient
        from stash_ai.cleanup.sql import SQLCleanup
        try:
            client = CleanupClient(self.stash)
            sql = SQLCleanup(client, self.args, self.stats, self.report)
            sql.session.probe()
            # Share one fresh, lightweight catalog between ordinary tags and
            # marker groups, including tags created during this merge.
            sql.set_catalog(client.catalog())
            self.stats['merge_prune_sql_enabled'] += 1
            return sql
        except Exception as exc:
            self.report.detail('SQL prune unavailable; using GraphQL: ' +
                               self.app.services.safe_error(exc, api_key=self.app.STASH_API_KEY))
        self.stats['merge_prune_graphql_fallback'] += 1
        return None

    def _prune_outputs(self):
        app, args, stash = self.app, self.args, self.stash
        app.services.print_hierarchy_summary(self.resolver)
        if self.stats['errors'] or not (args.prune_empty_tags or app.CREATE_MARKERS):
            return False
        pruned_before = self.stats['tags_pruned'] + self.stats['marker_group_tags_pruned']
        sql = self._prune_sql()
        if args.prune_empty_tags:
            from stash_ai.cleanup.prune import prune_tags
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
            from stash_ai.publication.groups import for_resolver
            from stash_ai.cleanup.groups import prune_groups
            delta = for_resolver(app, self.resolver).usage_delta
            candidates = sql.discovery.marker_group_candidates(delta) if sql is not None else None
            self.stats.update(prune_groups(
                stash, self.report.detail, candidate_ids=candidates, usage_delta=delta))
        pruned_after = self.stats['tags_pruned'] + self.stats['marker_group_tags_pruned']
        return not stash.dry_run and pruned_after > pruned_before
