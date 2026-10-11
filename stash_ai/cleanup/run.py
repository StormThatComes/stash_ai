"""Cleanup-specific merge reconciliation and stateful cleanup orchestration."""
from __future__ import annotations

from stash_ai.runtime.resources import ResourceMonitor, observe
from stash_ai.runtime.console import progress as console_bar
from stash_ai.runtime.console import activity, phase, skipped, status
from collections import Counter

from tqdm import tqdm

import stash_ai.evidence.labels as merge_labels
import stash_ai.storage.store as merge_store
from stash_ai.models import SOURCES
from stash_ai.cleanup import scenes


class CleanupRun:
    """Own the resources and phases of one cleanup execution."""

    def __init__(self, app, settings, args, core_policies, scene_policies):
        from stash_ai.exclusions import selected_policies
        self.app, self.settings, self.args = app, settings, args
        self.core_policies, self.scene_policies = core_policies, scene_policies
        self.selected = {config.source.key for config in settings}
        self.scoped_tags = selected_policies(scene_policies, self.selected)
        self.stats = Counter()
        self.report = args.report
        self.report.watch_counts(self.stats)
        self.wide = not args.scene_id and self.selected == set(SOURCES)
        self.store = self.stash = self.cleanup_stash = self.resolver = self.sql = None
        self.scene_ids = None

    def _impact(self, **values):
        keys = ('processed', 'changed', 'unchanged', 'missing_scenes', 'errors',
                'markers_deleted', 'marker_tag_removals', 'scene_tag_removals',
                'blacklisted_tags_deleted', 'marker_group_tags_pruned')
        self.report.workload['cleanup'] = {key: self.stats[key] for key in keys}
        status(self.report, cleanup_mode='planned' if self.app.DRY_RUN else 'confirmed',
               cleanup_scenes=' '.join(f'{key}={self.stats[key]}' for key in keys[:5]),
               cleanup_changes=' '.join(f'{key}={self.stats[key]}' for key in keys[5:]), **values)
        if not self.app.DRY_RUN:
            status(self.report, confirmed_changes=' '.join(f'{key}={self.report.counts["mutation_" + key + "_confirmed"]}'
                   for key in ('marker_delete', 'marker_tags', 'scene_tag_removals', 'tag_delete')))

    def run(self):
        app, args = self.app, self.args
        with merge_store.StateStore(args.state_db, merge_store.scope_for(app.STASH_URL), app.DRY_RUN,
                cache_bytes=int(args.ram_cache_gb * 2**30),
                allow_cleanup=self.wide or args.abandon_pending,
                memory_budget_bytes=int(args.ram_budget_gb * 2**30)) as self.store:
            if args.abandon_pending:
                from stash_ai.storage.recovery import abandon
                self.stats.update(abandon(self.store, self.report.write))
            self.stash = self.cleanup_stash = app.services.create_stash_client(app.DRY_RUN)
            self.stash.report = self.report
            self.store.report = self.report
            try:
                with observe(self.report, 'cache', ResourceMonitor(self.report, self.store.cache).sample):
                    self.report.cleanup_pending_observed = bool(self.store.cleanup_pending)
                    self._impact(strategy=args.cleanup_strategy, recovery='resume' if self.report.cleanup_pending_observed else None)
                    self._prepare_strategy()
                    with phase(self.report, 'affected scene reconciliation', errors=lambda: self.stats['errors']):
                        self._cleanup_scenes()
                    self._prune()
                    with phase(self.report, 'cleanup finalize'):
                        self._impact()
            except Exception as exc:
                self.stats['errors'] += 1
                self.report.issue('cleanup', '', None, exc, app.services.safe_error(exc, api_key=app.STASH_API_KEY))
            finally:
                from stash_ai.runtime.cleanup import cleanup
                cleanup([self.stash.close], report=self.report)
            self.report.finish('Cleanup' + (' (planned)' if app.DRY_RUN else ''), self.stats, self.store.cache)
        return int(bool(self.stats["errors"]))

    def _prepare_strategy(self):
        app, args = self.app, self.args
        from stash_ai.cleanup.client import CleanupClient
        self.scene_ids = args.scene_id or None
        strategy = args.cleanup_strategy
        client = CleanupClient(self.stash) if strategy in ('auto', 'sql') else None
        if strategy in ('auto', 'sql'):
            from stash_ai.cleanup.sql import SQLCleanup
            candidate = SQLCleanup(client, args, self.stats, self.report)
            try:
                with phase(self.report, 'cleanup probe'):
                    candidate.session.probe()
                self.sql = candidate
            except Exception as exc:
                if strategy == 'sql' or self.store.cleanup_pending:
                    raise
                self.report.write('SQL cleanup unavailable (' + app.services.safe_error(exc, api_key=app.STASH_API_KEY) +
                                  '); scanning scenes instead.')
                status(self.report, strategy='scene fallback', fallback=type(exc).__name__)
        if self.sql is not None:
            from stash_ai.cleanup.sql import run_sql_cleanup
            self.report.write('Cleanup by indexed SELECT: ID cursors, no global marker counts; writes use Stash API.')
            if self.store.cleanup_pending:
                self.stats['cleanup_resumed'] = 1
            self.scene_ids, self.cleanup_stash = run_sql_cleanup(
                app, self.sql, self.store, self.settings, args, self.core_policies, self.scene_policies, self.stats)
            self._impact(strategy='SQL', recovery=None)
        elif self.store.cleanup_pending:
            raise RuntimeError('Global cleanup is interrupted and the required Stash API is unavailable; '
                               'restore API access and rerun the same cleanup.')
        elif not args.scene_id:
            self.report.write('Cleanup scans scenes; global findSceneMarkers filtering is disabled. '
                              'SQL cleanup is faster for large collections.')

    def _cleanup_scenes(self):
        app = self.app
        # An empty candidate list must not be passed to iter_scenes: its empty
        # list convention means the whole library, not zero scenes.
        if self.scene_ids == []:
            return
        if self.sql is not None:
            # Cleanup uses the same snapshots as merge-only; media paths are unused.
            self.stash.configure_merge_reads(self.report, merge_only=True)
        self.resolver = app.services.TagResolver(self.cleanup_stash, source=merge_labels.AI, report=self.report.detail,
                                        create_missing=app.CREATE_MISSING)
        selection = set(map(str, self.scene_ids)) if self.scene_ids is not None else None
        with console_bar(tqdm, total=len(selection) if selection is not None else None,
                         desc="Affected scene cleanup", unit="scene") as bar:
            for current in self.cleanup_stash.iter_scenes(self.scene_ids):
                scene_id = str(current["id"])
                try:
                    with activity(self.report, 'reconcile cleanup scene', scene=scene_id):
                        self.stats.update(scenes.cleanup_scene(
                            app, self.cleanup_stash, self.resolver, self.store, current, self.selected,
                            self.core_policies, self.scene_policies, self.scoped_tags, self.report))
                    self.stats["processed"] += 1
                except Exception as exc:
                    self.stats["errors"] += 1
                    self.report.issue('cleanup', '', scene_id, exc, app.services.safe_error(exc, api_key=app.STASH_API_KEY))
                bar.update(1)
                self._impact(inspected=bar.n, selected=bar.total)
            self._finish_scene_reads()
            if selection is not None:
                bar.update(len(selection.intersection(self.report.unread_scenes)))
            self._impact(inspected=bar.n, selected=bar.total)

    def _finish_scene_reads(self):
        from stash_ai.stash_api.scene_reads import MissingSceneError
        for scene_id, exc in self.report.unread_scenes.items():
            if isinstance(exc, MissingSceneError):
                self.stats['skipped'] += 1
                self.stats['missing_scenes'] += 1
                self.report.detail(f'Cleanup: scene {scene_id} no longer exists; skipped.')
                continue
            self.stats['errors'] += 1
            self.report.issue('cleanup read', '', scene_id, exc,
                              self.app.services.safe_error(exc, api_key=self.app.STASH_API_KEY))

    def _prune(self):
        if self.stats['errors']:
            skipped(self.report, 'cleanup prune', 'errors')
            return
        with phase(self.report, 'leaf tag prune', errors=lambda: self.stats['errors']):
            self._prune_leaves()
        if self.app.CREATE_MARKERS:
            with phase(self.report, 'marker-group prune', errors=lambda: self.stats['errors']):
                self._prune_groups()
        else:
            skipped(self.report, 'marker-group prune', 'markers disabled')
        self._impact()

    def _prune_leaves(self):
        if self.stats['errors']:
            return
        app, args = self.app, self.args
        from stash_ai.exclusions import prune_excluded_definitions
        if args.prune_empty_tags:
            from stash_ai.cleanup.prune import prune_tags
            self.stats.update(self.sql.prune(prefixes=args.prune_prefixes, skip_marker_groups=app.CREATE_MARKERS)
                              if self.sql is not None else prune_tags(
                                  self.stash, args.prune_prefixes, report=self.report.detail,
                                  skip_marker_groups=app.CREATE_MARKERS))
        elif not self.stats['cleanup_fast_path'] or len({frozenset(v) for v in self.scene_policies.values()}) > 1:
            if self.sql is not None and app.APPLY_SCENE_TAGS and app.CREATE_MARKERS:
                self.stats.update(self.sql.prune(self.scoped_tags))
            else:
                self.stats.update(prune_excluded_definitions(
                    app, self.stash, self.scoped_tags, self.report.detail))

    def _prune_groups(self):
        from stash_ai.publication.groups import for_resolver
        from stash_ai.cleanup.groups import prune_groups
        delta = Counter(getattr(self.cleanup_stash, 'marker_group_usage_delta', {}))
        if self.resolver is not None:
            delta.update(for_resolver(self.app, self.resolver).usage_delta)
        candidates = self.sql.discovery.marker_group_candidates(delta) if self.sql is not None else None
        self.stats.update(prune_groups(
            self.stash, self.report.detail, candidate_ids=candidates, usage_delta=delta))
