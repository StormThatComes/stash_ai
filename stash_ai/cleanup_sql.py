"""Bounded cleanup discovery through Stash's SQL API. Every SQL statement is SELECT.

Writes still use Stash mutations (including hooks and generated-file removal).
Templates are also the dry-run allowlist: no caller-supplied SQL is accepted.
"""
from __future__ import annotations

import json
from collections import Counter
from itertools import batched

from tqdm import tqdm

from stash_ai import separate_sync
from stash_ai.cleanup import begin, finish
from stash_ai.models import SOURCES

from stash_ai.cleanup_sql_session import CleanupSqlSession


class SQLCleanup:
    def __init__(self, client, args, stats, report):
        self.client, self.stash = client, client.stash
        self.stats, self.report = stats, report
        self.session = CleanupSqlSession(self.stash, stats, page_size=args.cleanup_page_size)
        self.write_size = args.cleanup_write_batch_size
        self.progress = None
        self.catalog = {}
        self.ids_by_name = {}
        self.preview_names, self.preview_cores = set(), {}
        self.group_usage_delta = Counter()

    def tick(self):
        if self.progress is not None:
            self.progress.set_postfix(deleted=self.stats['markers_deleted'],
                removed=self.stats['marker_tag_removals'] + self.stats['scene_tag_removals'], refresh=False)
            self.progress.update(0)

    def delete_primary(self, identifier, name, source=None):
        from stash_ai.marker_groups import PREFIX
        group_ids = [int(i) for i, n in self.catalog.items() if n.startswith(PREFIX)]
        prefix, owner = (source.title_prefix, source.owner_tag) if source else ('', '')
        params = (int(identifier), name, prefix, prefix, prefix, owner)
        for rows in self.session.pages('primary', params, self.session.bounds[0]):
            for group in batched((int(row[0]) for row in rows), self.write_size):
                current = group if self.stash.dry_run else [int(row[0]) for row in
                    self.session.read('primary_check', (*params, json.dumps(group)))]
                if not set(current) <= set(group):
                    raise ValueError('Unexpected cleanup marker IDs; stopped before mutation.')
                self.stats['cleanup_markers_changed_or_gone'] += len(group) - len(current)
                if current and not self.stash.dry_run:
                    ids = list(map(str, current))
                    result = self.session.write('''mutation StashAICleanupDeleteMarkers($ids: [ID!]!) {
                        sceneMarkersDestroy(ids: $ids)
                    }''', {'ids': ids})['sceneMarkersDestroy']
                    if result is not True:
                        raise RuntimeError('Marker deletion was not confirmed; rerun the same cleanup.')
                if self.stash.dry_run and current and group_ids:
                    for tag_id, count in self.session.read('group_usage', (json.dumps(current), json.dumps(group_ids))):
                        self.group_usage_delta[str(tag_id)] -= int(count)
                self.stats['markers_deleted'] += len(current)
                self.tick()

    def remove_associations(self, identifier, name, *, markers):
        key, check, bound = ('extras', 'extra_check', self.session.bounds[1]) if markers else (
            'scenes', 'scene_check', self.session.bounds[2])
        for rows in self.session.pages(key, (int(identifier), name), bound):
            for group in batched(dict.fromkeys(int(row[1]) for row in rows), self.write_size):
                current = group if self.stash.dry_run else [int(row[0]) for row in
                    self.session.read(check, (int(identifier), name, json.dumps(group)))]
                current = list(dict.fromkeys(current))
                if markers and self.stash.dry_run:
                    current = self.preview_survivors(current)
                if not set(current) <= set(group):
                    raise ValueError('Unexpected cleanup association IDs; stopped before mutation.')
                if current and not self.stash.dry_run:
                    field, input_type = ('bulkSceneMarkerUpdate', 'BulkSceneMarkerUpdateInput') if markers else (
                        'bulkSceneUpdate', 'BulkSceneUpdateInput')
                    result = self.session.write('mutation StashAICleanupRemoveTags($input: ' + input_type +
                        '!) { ' + field + '(input: $input) { id } }', {'input': {
                            'ids': list(map(str, current)), 'tag_ids': {'ids': [str(identifier)], 'mode': 'REMOVE'}}})[field]
                    if result is None or {str(row['id']) for row in result} != set(map(str, current)):
                        raise RuntimeError('Tag detachment was not confirmed; rerun the same cleanup.')
                # These count object/tag operations; one object can lose several tags.
                self.stats['marker_tag_removals' if markers else 'scene_tag_removals'] += len(current)
                self.tick()

    def preview_survivors(self, ids):
        """Do not count extra removals on markers already planned for deletion."""
        rows = self.session.read('marker_metadata', (json.dumps(ids),))
        removed = {int(i) for i, name in rows if name in self.preview_names}
        core_groups = {}
        for i, name in rows:
            if name in self.preview_cores:
                core_groups.setdefault(name, []).append(int(i))
        for name, group in core_groups.items():
            source = self.preview_cores[name]
            p = source.title_prefix
            params = (self.ids_by_name[name], name, p, p, p, source.owner_tag, json.dumps(group))
            removed.update(int(row[0]) for row in self.session.read('primary_check', params))
        return [i for i in ids if i not in removed]

    def delete_tag(self, identifier, name):
        if not self.stash.dry_run and self.session.read('tag_check', (json.dumps([int(identifier)]),)) != [[int(identifier), name]]:
            self.stats['cleanup_tags_changed_or_gone'] += 1
            return
        # One tag per mutation: a popular tag can have millions of associations.
        if not self.stash.dry_run:
            result = self.session.write('''mutation StashAICleanupDeleteTags($ids: [ID!]!) {
                tagsDestroy(ids: $ids)
            }''', {'ids': [str(identifier)]})['tagsDestroy']
            if result is not True:
                raise RuntimeError('Tag deletion was not confirmed; rerun the same cleanup.')
        self.stats['blacklisted_tags_deleted'] += 1

    def candidates(self, catalog, scene_names, primary_names, extra_names):
        scenes = set()
        for identifier, name in catalog.items():
            params = (int(identifier),)
            if name in primary_names:
                for rows in self.session.pages('primary_scenes', params, self.session.bounds[0]):
                    scenes.update(str(row[1]) for row in rows)
            if name in extra_names:
                for rows in self.session.pages('extra_scenes', params, self.session.bounds[1]):
                    scenes.update(str(row[1]) for row in rows)
            if name in scene_names:
                for rows in self.session.pages('scenes', (int(identifier), name), self.session.bounds[2]):
                    scenes.update(str(row[1]) for row in rows)
        return scenes

    def unused_candidates(self, ids, *, page_size=200, progress=None):
        """Reject definitely-used tags; native usage/ownership checks still decide deletion."""
        candidates = set()
        for group in batched(sorted(set(ids)), page_size):
            found = {str(row[0]) for row in self.session.read('unused_candidates', (json.dumps(list(map(int, group))),))}
            if not found <= set(group):
                raise ValueError('Unexpected prune candidate IDs; stopped before mutation.')
            candidates.update(found)
            if progress is not None:
                progress.set_postfix(candidates=len(candidates), refresh=False)
                progress.update(len(group))
        return candidates

    def marker_group_candidates(self, usage_delta=None):
        from stash_ai.marker_groups import PREFIX
        selected = {i for i, name in self.catalog.items() if name.startswith(PREFIX)}
        if not selected:
            return []
        with tqdm(total=len(selected), desc='Find empty marker groups', unit='tag', dynamic_ncols=True) as bar:
            candidates = self.unused_candidates(selected, page_size=self.session.page_size, progress=bar)
        if self.stash.dry_run:
            # SQL sees the real database, not planned removals. Let PreviewUsage
            # check groups losing members, even when SQL still finds their tags.
            candidates.update(i for i, change in (usage_delta or {}).items() if change < 0 and i in selected)
        self.stats['marker_group_tags_checked_sql'] += len(selected)
        self.stats['marker_group_tags_candidates'] += len(candidates)
        return sorted(candidates)

    def prune(self, policies=None, prefixes=None, *, skip_marker_groups=False):
        from stash_ai.exclusions import excluded_names
        from stash_ai.marker_groups import PREFIX
        from stash_ai.prune import prefixes_checked, prune_tags
        names = excluded_names(policies, unused=True) if policies is not None else None
        prefixes = prefixes_checked(prefixes)
        selected = {i for i, n in self.catalog.items()
                    if (n in names if names is not None else n.startswith(prefixes))
                    and not (skip_marker_groups and n.startswith(PREFIX))}
        candidates = self.unused_candidates(selected)
        if not candidates:
            return {}
        # SQL only rejects definitely-used tags. Stash still checks EVERY exposed
        # usage/child counter, including images/groups/etc., before deleting a leaf.
        return prune_tags(self.stash, prefixes, self.report.detail, names=names, candidate_ids=sorted(candidates),
                          skip_marker_groups=skip_marker_groups)


def candidate_names(app, selected, cores, tags):
    from stash_ai.merge import AI
    scene, primary, extra = set(), set(), set()
    for source in [AI, *(SOURCES[s] for s in selected)]:
        relevant = selected if source == AI else {source.key}
        bans = set().union(*(tags[s] for s in relevant))
        core_bans = set().union(*(cores[s] for s in relevant))
        if app.APPLY_SCENE_TAGS:
            scene.update(source.prefix + label for label in bans)
        if app.CREATE_MARKERS:
            primary.update(source.prefix + label for label in bans | core_bans)
            extra.update(source.prefix + label for label in bans)
    return scene, primary, extra


class SQLPreview:
    """Overlay global dry-run effects without retaining millions of marker IDs."""
    def __init__(self, stash, app, names, cores, group_usage_delta=None):
        self.stash, self.app, self.names, self.cores = stash, app, names, cores
        self.marker_group_usage_delta = group_usage_delta or Counter()

    def __getattr__(self, name):
        return getattr(self.stash, name)

    def find_scene(self, identifier):
        return self._project(self.stash.find_scene(identifier))

    def find_scene_for_merge(self, identifier):
        return self._project(self.stash.find_scene_for_merge(identifier))

    def _project(self, scene):
        from copy import deepcopy
        scene = deepcopy(scene)
        if self.app.APPLY_SCENE_TAGS:
            scene['tags'] = [t for t in scene.get('tags', []) if t['name'] not in self.names]
        if self.app.CREATE_MARKERS:
            scene['scene_markers'] = [m for m in scene.get('scene_markers', []) if
                m['primary_tag']['name'] not in self.names and not (
                    m['primary_tag']['name'] in self.cores and separate_sync.owned_marker(m, self.cores[m['primary_tag']['name']]))]
            for marker in scene['scene_markers']:
                marker['tags'] = [t for t in marker.get('tags', []) if t['name'] not in self.names]
        return scene

    def iter_scenes(self, ids):
        if not ids:
            return
        for scene in self.stash.iter_scenes(ids):
            yield self._project(scene)


def run_sql_cleanup(app, sql, store, settings, args, cores, tags, stats):
    """Fast common bans; only source-dependent policies need scene reconciliation."""
    from stash_ai.merge import AI
    from stash_ai.progress import digest
    selected = {config.source.key for config in settings}
    wide = not args.scene_id and selected == set(SOURCES)
    global_labels = frozenset.intersection(*tags.values()) if wide else frozenset()
    common_cores = (frozenset.intersection(*cores.values()) - global_labels) if wide and app.CREATE_MARKERS else frozenset()
    sources = [AI, *SOURCES.values()]
    names = {s.prefix + label for s in sources for label in global_labels}
    core_names = {s.prefix + label: s for s in sources for label in common_cores}
    catalog = sql.client.catalog()
    sql.catalog = catalog
    sql.ids_by_name = {name: int(i) for i, name in catalog.items()}
    # Validate all recovery gates before discovery and before the first write.
    if wide and store.has_pending_publications():
        raise RuntimeError('An interrupted merge is pending; run --merge-only before cleanup.')
    fingerprint_data = {'tags': {s: sorted(tags[s]) for s in SOURCES},
                        'cores': {s: sorted(cores[s]) for s in SOURCES},
                        # Preserve resumability of cleanups started with the default empty list.
                        'marker_blocklist': []}
    kinds = [k for k, enabled in [('scene', app.APPLY_SCENE_TAGS), ('markers', app.CREATE_MARKERS)] if enabled] if wide else []
    if kinds != ['scene', 'markers']:
        fingerprint_data['output_kinds'] = kinds
    fingerprint = digest(fingerprint_data)
    if store.cleanup_pending and store.cleanup_pending != fingerprint:
        raise RuntimeError('Global cleanup is interrupted; rerun the same cleanup mode and exclusion policies.')
    if wide and (global_labels or common_cores or store.cleanup_pending):
        begin(store, fingerprint, global_labels, common_cores,
              [i for i, n in catalog.items() if n in names], kinds=kinds)
    remaining_tags = {s: tags[s] - global_labels for s in SOURCES}
    remaining_cores = {s: cores[s] - common_cores - global_labels for s in SOURCES}
    candidates = set(map(str, args.scene_id)) if args.scene_id else sql.candidates(
        catalog, *candidate_names(app, selected, remaining_cores, remaining_tags))
    tasks = [(identifier, name) for identifier, name in catalog.items() if name in names or name in core_names]
    sql.preview_names, sql.preview_cores = names, core_names
    with tqdm(tasks, desc='Clean excluded tags' + (' (preview)' if app.DRY_RUN else ''), unit='tag', dynamic_ncols=True) as bar:
        sql.progress = bar
        for identifier, name in bar:
            if app.CREATE_MARKERS:
                sql.delete_primary(identifier, name, core_names.get(name))
            if name in names:
                if app.APPLY_SCENE_TAGS and app.CREATE_MARKERS:
                    sql.delete_tag(identifier, name)
                else:
                    sql.remove_associations(identifier, name, markers=app.CREATE_MARKERS)
            bar.set_postfix(deleted=stats['markers_deleted'], removed=stats['marker_tag_removals'] + stats['scene_tag_removals'])
    if store.cleanup_pending:
        finish(store)
    stats['cleanup_fast_path'] = 1
    stats['cleanup_sql_path'] = 1
    stats['cleanup_candidate_scenes'] = len(candidates)
    view = SQLPreview(sql.stash, app, names, core_names, sql.group_usage_delta) if app.DRY_RUN else sql.stash
    return sorted(candidates), view
