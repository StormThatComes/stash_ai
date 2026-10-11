"""Bounded cleanup discovery through Stash's SQL API. Every SQL statement is SELECT.

Writes still use Stash mutations (including hooks and generated-file removal).
Templates are also the dry-run allowlist: no caller-supplied SQL is accepted.
"""
from __future__ import annotations


from stash_ai.runtime.console import progress as console_bar
from stash_ai.runtime.console import phase, status
from tqdm import tqdm

from stash_ai.cleanup.state import begin, finish
from stash_ai.models import SOURCES

from dataclasses import dataclass, field
from stash_ai.cleanup.sql_session import CleanupSqlSession
from stash_ai.cleanup.sql_discovery import SQLDiscovery
from stash_ai.cleanup.sql_mutations import SQLMutationExecutor
from stash_ai.cleanup.sql_preview import PreviewEffects, SQLPreview


@dataclass
class CleanupCatalog:
    records: dict[str, str] = field(default_factory=dict)
    ids_by_name: dict[str, int] = field(default_factory=dict)


class SQLCleanup:
    """Own shared SQL resources; discovery and mutations have explicit collaborators."""
    def __init__(self, client, args, stats, report):
        self.client, self.stash = client, client.stash
        self.stats, self.report = stats, report
        self.session = CleanupSqlSession(self.stash, stats, page_size=args.cleanup_page_size)
        self.catalog = CleanupCatalog()
        self.preview = PreviewEffects()
        self.discovery = SQLDiscovery(self.session, self.catalog)
        self.mutations = SQLMutationExecutor(
            self.session, self.catalog, self.preview, write_size=args.cleanup_write_batch_size)

    def set_catalog(self, records):
        self.catalog.records = records
        self.catalog.ids_by_name = {name: int(identifier) for identifier, name in records.items()}

    def prune(self, policies=None, prefixes=None, *, skip_marker_groups=False):
        from stash_ai.exclusions import excluded_names
        from stash_ai.publication.groups import PREFIX
        from stash_ai.cleanup.prune import prefixes_checked, prune_tags
        names = excluded_names(policies, unused=True) if policies is not None else None
        prefixes = prefixes_checked(prefixes)
        selected = {i for i, n in self.catalog.records.items()
                    if (n in names if names is not None else n.startswith(prefixes))
                    and not (skip_marker_groups and n.startswith(PREFIX))}
        candidates = self.discovery.unused_candidates(selected)
        if not candidates:
            return {}
        # SQL only rejects definitely-used tags. Stash still checks EVERY exposed
        # usage/child counter, including images/groups/etc., before deleting a leaf.
        return prune_tags(self.stash, prefixes, self.report.detail, names=names, candidate_ids=sorted(candidates),
                          skip_marker_groups=skip_marker_groups)


def candidate_names(app, selected, cores, tags):
    from stash_ai.evidence.labels import AI
    scene, primary, extra = set(), set(), set()
    bans = set().union(*(tags[source] for source in selected))
    core_bans = set().union(*(cores[source] for source in selected))
    if app.APPLY_SCENE_TAGS:
        scene.update(AI.prefix + label for label in bans)
    if app.CREATE_MARKERS:
        primary.update(AI.prefix + label for label in bans | core_bans)
        extra.update(AI.prefix + label for label in bans)
    return scene, primary, extra


def run_sql_cleanup(app, sql, store, settings, args, cores, tags, stats):
    """Fast common bans; only source-dependent policies need scene reconciliation."""
    from stash_ai.evidence.labels import AI
    from stash_ai.cleanup.fingerprint import fingerprint as cleanup_fingerprint
    selected = {config.source.key for config in settings}
    wide = not args.scene_id and selected == set(SOURCES)
    global_labels = frozenset.intersection(*tags.values()) if wide else frozenset()
    common_cores = (frozenset.intersection(*cores.values()) - global_labels) if wide and app.CREATE_MARKERS else frozenset()
    names = {AI.prefix + label for label in global_labels}
    core_names = {AI.prefix + label: AI for label in common_cores}
    with phase(sql.report, 'cleanup catalog'):
        catalog = sql.client.catalog()
    sql.set_catalog(catalog)
    # Validate all recovery gates before discovery and before the first write.
    if wide and store.has_pending_publications():
        sql.report.publication_pending_observed = True
        raise RuntimeError('An interrupted merge is pending; run --merge-only before cleanup.')
    kinds = [kind for kind, enabled in [('scene', app.APPLY_SCENE_TAGS), ('markers', app.CREATE_MARKERS)] if enabled] if wide else []
    fingerprint = cleanup_fingerprint(tags, cores, kinds)
    if store.cleanup_pending and store.cleanup_pending != fingerprint:
        raise RuntimeError('Global cleanup is interrupted; rerun the same cleanup mode and exclusion policies. '
                           'For a historical intent, first run python -m stash_ai.tools.cleanup_pending_migration '
                           'with those policies and the same cleanup mode.')
    if wide and (global_labels or common_cores or store.cleanup_pending):
        begin(store, fingerprint, global_labels, common_cores,
              [i for i, n in catalog.items() if n in names], kinds=kinds)
        if not app.DRY_RUN:
            sql.report.cleanup_pending_observed = True
        status(sql.report, recovery='cleanup intent pending')
    remaining_tags = {s: tags[s] - global_labels for s in SOURCES}
    remaining_cores = {s: cores[s] - common_cores - global_labels for s in SOURCES}
    with phase(sql.report, 'cleanup discovery'):
        candidates = set(map(str, args.scene_id)) if args.scene_id else sql.discovery.candidates(
            catalog, *candidate_names(app, selected, remaining_cores, remaining_tags))
    tasks = [(identifier, name) for identifier, name in catalog.items() if name in names or name in core_names]
    sql.preview.names, sql.preview.cores = names, core_names
    with phase(sql.report, 'global cleanup', mode='planned' if app.DRY_RUN else 'apply'), console_bar(tqdm, tasks, desc='Clean excluded tags' + (' (preview)' if app.DRY_RUN else ''), unit='tag', dynamic_ncols=True) as bar:
        sql.mutations.progress = bar
        for identifier, name in bar:
            status(sql.report, tag_index=bar.n + 1, tag_total=bar.total)
            if app.CREATE_MARKERS:
                sql.mutations.delete_primary(identifier, name, core_names.get(name))
            if name in names:
                if app.APPLY_SCENE_TAGS and app.CREATE_MARKERS:
                    sql.mutations.delete_tag(identifier, name)
                else:
                    sql.mutations.remove_associations(identifier, name, markers=app.CREATE_MARKERS)
            bar.set_postfix(deleted=stats['markers_deleted'], removed=stats['marker_tag_removals'] + stats['scene_tag_removals'])
    if store.cleanup_pending:
        with phase(sql.report, 'cleanup state commit'):
            finish(store)
        status(sql.report, recovery=None, local_commit='cleanup committed')
        if not app.DRY_RUN:
            sql.report.cleanup_pending_observed = False
    stats['cleanup_fast_path'] = 1
    stats['cleanup_sql_path'] = 1
    stats['cleanup_candidate_scenes'] = len(candidates)
    view = SQLPreview(sql.stash, app, names, core_names, sql.preview.group_usage_delta) if app.DRY_RUN else sql.stash
    return sorted(candidates), view
