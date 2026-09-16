"""Bounded cleanup discovery through Stash's SQL API. Every SQL statement is SELECT.

Writes still use Stash mutations (including hooks and generated-file removal).
Templates are also the dry-run allowlist: no caller-supplied SQL is accepted.
"""
from __future__ import annotations

import json
from time import monotonic

from tqdm import tqdm

from stash_cleanup import batches, begin, finish
from stash_taggers import SOURCES

READ_OPERATION = '''mutation StashAICleanupReadSQL($sql: String!, $args: [Any]) {
    querySQL(sql: $sql, args: $args) { columns rows }
}'''

# tag_id indexes in Stash also contain rowid. For relation tables, paginate by
# rowid, NOT scene_marker_id/scene_id: the latter can sort the full hit set again
# on every page. No COUNT, OFFSET, recursive hierarchy or marker relation resolvers.
OWNER = """AND (? = '' OR (substr(m.title, 1, length(?)) = ? AND EXISTS (
    SELECT 1 FROM scene_markers_tags own CROSS JOIN tags ot ON ot.id = own.tag_id
    WHERE own.scene_marker_id = m.id AND ot.name = ? COLLATE BINARY)))"""
PRIMARY = """FROM scene_markers m JOIN tags t ON t.id = m.primary_tag_id
    WHERE m.primary_tag_id = ? AND t.name = ? COLLATE BINARY """ + OWNER
SQL = {
    'probe': '''SELECT m.id, m.primary_tag_id, m.scene_id, m.title, t.name,
        mt.rowid, mt.tag_id, mt.scene_marker_id, st.rowid, st.tag_id, st.scene_id
        FROM scene_markers m, tags t, scene_markers_tags mt, scenes_tags st WHERE 0''',
    'json': 'SELECT value FROM json_each(?)',
    'bounds': '''SELECT (SELECT coalesce(max(id), 0) FROM scene_markers),
        (SELECT coalesce(max(rowid), 0) FROM scene_markers_tags),
        (SELECT coalesce(max(rowid), 0) FROM scenes_tags)''',
    'primary': 'SELECT m.id ' + PRIMARY + ' AND m.id > ? AND m.id <= ? ORDER BY m.id LIMIT ?',
    'primary_check': 'SELECT m.id ' + PRIMARY + ' AND m.id IN (SELECT value FROM json_each(?))',
    'primary_scenes': '''SELECT m.id, m.scene_id FROM scene_markers m
        WHERE m.primary_tag_id = ? AND m.id > ? AND m.id <= ? ORDER BY m.id LIMIT ?''',
    'extra_scenes': '''SELECT mt.rowid, m.scene_id FROM scene_markers_tags mt
        JOIN scene_markers m ON m.id = mt.scene_marker_id
        WHERE mt.tag_id = ? AND mt.rowid > ? AND mt.rowid <= ? ORDER BY mt.rowid LIMIT ?''',
    'extras': '''SELECT mt.rowid, mt.scene_marker_id FROM scene_markers_tags mt
        JOIN tags t ON t.id = mt.tag_id WHERE mt.tag_id = ? AND t.name = ? COLLATE BINARY
        AND mt.rowid > ? AND mt.rowid <= ? ORDER BY mt.rowid LIMIT ?''',
    'scenes': '''SELECT st.rowid, st.scene_id FROM scenes_tags st
        JOIN tags t ON t.id = st.tag_id WHERE st.tag_id = ? AND t.name = ? COLLATE BINARY
        AND st.rowid > ? AND st.rowid <= ? ORDER BY st.rowid LIMIT ?''',
    'extra_check': '''SELECT mt.scene_marker_id FROM scene_markers_tags mt
        JOIN tags t ON t.id = mt.tag_id WHERE mt.tag_id = ? AND t.name = ? COLLATE BINARY
        AND mt.scene_marker_id IN (SELECT value FROM json_each(?))''',
    'scene_check': '''SELECT st.scene_id FROM scenes_tags st
        JOIN tags t ON t.id = st.tag_id WHERE st.tag_id = ? AND t.name = ? COLLATE BINARY
        AND st.scene_id IN (SELECT value FROM json_each(?))''',
    'tag_check': 'SELECT id, name FROM tags WHERE id IN (SELECT value FROM json_each(?))',
    'marker_metadata': '''SELECT m.id, t.name FROM scene_markers m JOIN tags t ON t.id=m.primary_tag_id
        WHERE m.id IN (SELECT value FROM json_each(?))''',
    'unused_candidates': '''SELECT t.id FROM tags t WHERE t.id IN (SELECT value FROM json_each(?))
        AND NOT EXISTS (SELECT 1 FROM scene_markers m WHERE m.primary_tag_id=t.id)
        AND NOT EXISTS (SELECT 1 FROM scene_markers_tags mt WHERE mt.tag_id=t.id)
        AND NOT EXISTS (SELECT 1 FROM scenes_tags st WHERE st.tag_id=t.id)''',
}


def is_read_only_cleanup(query, variables):
    """Only the exact SELECT templates above may pass the dry-run write guard."""
    return (query == READ_OPERATION and isinstance(variables, dict)
            and variables.get('sql') in SQL.values())


class SQLCleanup:
    def __init__(self, client, args, stats, report):
        self.client, self.stash = client, client.stash
        self.stats, self.report = stats, report
        self.page_size = args.cleanup_page_size
        self.write_size = args.cleanup_write_batch_size
        self.bounds = None
        self.progress = None
        self.catalog = {}
        self.ids_by_name = {}
        self.preview_names, self.preview_cores = set(), {}

    def tick(self):
        if self.progress is not None:
            self.progress.set_postfix(deleted=self.stats['markers_deleted'],
                removed=self.stats['marker_tag_removals'] + self.stats['scene_tag_removals'], refresh=False)
            self.progress.update(0)

    def read(self, key, params=()):
        started = monotonic()
        result = self.stash.gql(READ_OPERATION, {'sql': SQL[key], 'args': list(params)})['querySQL']
        self.stats['cleanup_sql_reads'] += 1
        self.stats['cleanup_sql_read_ms'] += round((monotonic() - started) * 1000)
        rows = result.get('rows')
        if not isinstance(rows, list) or any(not isinstance(row, list) for row in rows):
            raise ValueError('Invalid SQL cleanup response; no write attempted for this batch.')
        self.stats['cleanup_rows_read'] += len(rows)
        return rows

    def probe(self):
        self.read('probe')
        if self.read('json', ('[1]',)) != [[1]]:
            raise RuntimeError('Stash SQL JSON support is incompatible with cleanup.')
        self.bounds = self.read('bounds')[0]

    def write(self, query, variables):
        started = monotonic()
        self.stats['cleanup_api_writes'] += 1
        try:
            return self.stash.gql(query, variables)
        finally:
            self.stats['cleanup_api_write_ms'] += round((monotonic() - started) * 1000)

    def pages(self, key, params, bound):
        cursor = 0
        while True:
            rows = self.read(key, (*params, cursor, bound, self.page_size))
            if not rows:
                return
            previous = cursor
            for row in rows:
                identifier = int(row[0])
                if identifier <= previous or identifier > bound:
                    raise ValueError('Stash returned an invalid cleanup cursor; stopped before mutation.')
                previous = identifier
            # Capture the cursor BEFORE yielding: the caller may delete these rows.
            cursor = previous
            yield rows
            if len(rows) < self.page_size:
                return

    def delete_primary(self, identifier, name, source=None):
        prefix, owner = (source.title_prefix, source.owner_tag) if source else ('', '')
        params = (int(identifier), name, prefix, prefix, prefix, owner)
        for rows in self.pages('primary', params, self.bounds[0]):
            for group in batches([int(row[0]) for row in rows], self.write_size):
                current = group if self.stash.dry_run else [int(row[0]) for row in
                    self.read('primary_check', (*params, json.dumps(group)))]
                if not set(current) <= set(group):
                    raise ValueError('Unexpected cleanup marker IDs; stopped before mutation.')
                self.stats['cleanup_markers_changed_or_gone'] += len(group) - len(current)
                if current and not self.stash.dry_run:
                    ids = list(map(str, current))
                    if self.client.bulk_markers:
                        result = self.write('''mutation StashAICleanupDeleteMarkers($ids: [ID!]!) {
                            sceneMarkersDestroy(ids: $ids)
                        }''', {'ids': ids})['sceneMarkersDestroy']
                        if result is not True:
                            raise RuntimeError('Marker deletion was not confirmed; rerun the same cleanup.')
                    else:
                        for value in ids:
                            result = self.write('''mutation StashAICleanupDeleteMarker($id: ID!) {
                                sceneMarkerDestroy(id: $id)
                            }''', {'id': value})['sceneMarkerDestroy']
                            if result is not True:
                                raise RuntimeError('Marker deletion was not confirmed; rerun the same cleanup.')
                self.stats['markers_deleted'] += len(current)
                self.tick()

    def remove_associations(self, identifier, name, *, markers):
        key, check, bound = ('extras', 'extra_check', self.bounds[1]) if markers else (
            'scenes', 'scene_check', self.bounds[2])
        for rows in self.pages(key, (int(identifier), name), bound):
            for group in batches(list(dict.fromkeys(int(row[1]) for row in rows)), self.write_size):
                current = group if self.stash.dry_run else [int(row[0]) for row in
                    self.read(check, (int(identifier), name, json.dumps(group)))]
                current = list(dict.fromkeys(current))
                if markers and self.stash.dry_run:
                    current = self.preview_survivors(current)
                if not set(current) <= set(group):
                    raise ValueError('Unexpected cleanup association IDs; stopped before mutation.')
                if current and not self.stash.dry_run:
                    field, input_type = ('bulkSceneMarkerUpdate', 'BulkSceneMarkerUpdateInput') if markers else (
                        'bulkSceneUpdate', 'BulkSceneUpdateInput')
                    result = self.write('mutation StashAICleanupRemoveTags($input: ' + input_type +
                        '!) { ' + field + '(input: $input) { id } }', {'input': {
                            'ids': list(map(str, current)), 'tag_ids': {'ids': [str(identifier)], 'mode': 'REMOVE'}}})[field]
                    if result is None or {str(row['id']) for row in result} != set(map(str, current)):
                        raise RuntimeError('Tag detachment was not confirmed; rerun the same cleanup.')
                # These count object/tag operations; one object can lose several tags.
                self.stats['marker_tag_removals' if markers else 'scene_tag_removals'] += len(current)
                self.tick()

    def preview_survivors(self, ids):
        """Do not count extra removals on markers already planned for deletion."""
        rows = self.read('marker_metadata', (json.dumps(ids),))
        removed = {int(i) for i, name in rows if name in self.preview_names}
        core_groups = {}
        for i, name in rows:
            if name in self.preview_cores:
                core_groups.setdefault(name, []).append(int(i))
        for name, group in core_groups.items():
            source = self.preview_cores[name]
            p = source.title_prefix
            params = (self.ids_by_name[name], name, p, p, p, source.owner_tag, json.dumps(group))
            removed.update(int(row[0]) for row in self.read('primary_check', params))
        return [i for i in ids if i not in removed]

    def delete_tag(self, identifier, name):
        if not self.stash.dry_run and self.read('tag_check', (json.dumps([int(identifier)]),)) != [[int(identifier), name]]:
            self.stats['cleanup_tags_changed_or_gone'] += 1
            return
        # One tag per mutation: a popular tag can have millions of associations.
        if not self.stash.dry_run:
            if self.client.bulk_tags:
                result = self.write('''mutation StashAICleanupDeleteTags($ids: [ID!]!) {
                    tagsDestroy(ids: $ids)
                }''', {'ids': [str(identifier)]})['tagsDestroy']
            else:
                result = self.write('''mutation StashAICleanupDeleteTag($input: TagDestroyInput!) {
                    tagDestroy(input: $input)
                }''', {'input': {'id': str(identifier)}})['tagDestroy']
            if result is not True:
                raise RuntimeError('Tag deletion was not confirmed; rerun the same cleanup.')
        self.stats['blacklisted_tags_deleted'] += 1

    def candidates(self, catalog, scene_names, primary_names, extra_names):
        scenes = set()
        for identifier, name in catalog.items():
            params = (int(identifier),)
            if name in primary_names:
                for rows in self.pages('primary_scenes', params, self.bounds[0]):
                    scenes.update(str(row[1]) for row in rows)
            if name in extra_names:
                for rows in self.pages('extra_scenes', params, self.bounds[1]):
                    scenes.update(str(row[1]) for row in rows)
            if name in scene_names:
                for rows in self.pages('scenes', (int(identifier), name), self.bounds[2]):
                    scenes.update(str(row[1]) for row in rows)
        return scenes

    def prune(self, policies=None, prefixes=None):
        from stash_exclusions import excluded_names
        from stash_prune import prefixes_checked, prune_tags
        names = excluded_names(policies, unused=True) if policies is not None else None
        prefixes = prefixes_checked(prefixes)
        selected = {i for i, n in self.catalog.items() if (n in names if names is not None else n.startswith(prefixes))}
        candidates = set()
        for group in batches(sorted(selected), 200):
            candidates.update(str(row[0]) for row in self.read('unused_candidates', (json.dumps(list(map(int,group))),)))
        if not candidates <= selected:
            raise ValueError('Unexpected prune candidate IDs; stopped before mutation.')
        if not candidates:
            return {}
        # SQL only rejects definitely-used tags. Stash still checks EVERY exposed
        # usage/child counter, including images/groups/etc., before deleting a leaf.
        return prune_tags(self.stash, prefixes, self.report.detail, names=names, candidate_ids=sorted(candidates))


def candidate_names(app, selected, cores, tags):
    from stash_merge import AI
    scene, primary, extra = set(), set(), set()
    for source in [AI, *(SOURCES[s] for s in selected)]:
        relevant = selected if source == AI else {source.key}
        bans = set().union(*(tags[s] for s in relevant))
        core_bans = set().union(*(cores[s] for s in relevant))
        if app.APPLY_SCENE_TAGS:
            scene.update(source.prefix + label for label in bans)
        if app.CREATE_MARKERS:
            primary.update(source.prefix + label for label in bans | core_bans | app.MARKER_TAG_BLOCKLIST)
            extra.update(source.prefix + label for label in bans | app.MARKER_TAG_BLOCKLIST)
    return scene, primary, extra


class SQLPreview:
    """Overlay global dry-run effects without retaining millions of marker IDs."""
    def __init__(self, stash, app, names, cores):
        self.stash, self.app, self.names, self.cores = stash, app, names, cores

    def __getattr__(self, name):
        return getattr(self.stash, name)

    def find_scene(self, identifier):
        from copy import deepcopy
        scene = deepcopy(self.stash.find_scene(identifier))
        if self.app.APPLY_SCENE_TAGS:
            scene['tags'] = [t for t in scene.get('tags', []) if t['name'] not in self.names]
        if self.app.CREATE_MARKERS:
            scene['scene_markers'] = [m for m in scene.get('scene_markers', []) if
                m['primary_tag']['name'] not in self.names and not (
                    m['primary_tag']['name'] in self.cores and self.app.owned_marker(m, self.cores[m['primary_tag']['name']]))]
            for marker in scene['scene_markers']:
                marker['tags'] = [t for t in marker.get('tags', []) if t['name'] not in self.names]
        return scene

    def iter_scenes(self, ids):
        for identifier in ids:
            yield self.find_scene(identifier)


def run_sql_cleanup(app, sql, store, settings, args, cores, tags, stats):
    """Fast common bans; only source-dependent policies need scene reconciliation."""
    from stash_merge import AI
    from stash_progress import digest
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
                        'marker_blocklist': sorted(app.MARKER_TAG_BLOCKLIST)}
    kinds = [k for k, enabled in [('scene', app.APPLY_SCENE_TAGS), ('markers', app.CREATE_MARKERS)] if enabled] if wide else []
    if kinds != ['scene', 'markers']:
        fingerprint_data['output_kinds'] = kinds
    fingerprint = digest(fingerprint_data)
    if store.cleanup_pending and store.cleanup_pending != fingerprint:
        raise RuntimeError('Global cleanup is interrupted; rerun the same cleanup mode and exclusion policies.')
    # Older APIs without bulk REMOVE can still use SQL to discover affected scenes.
    direct = wide and ((app.APPLY_SCENE_TAGS and app.CREATE_MARKERS) or
        (app.CREATE_MARKERS and sql.client.bulk_marker_tags) or
        (app.APPLY_SCENE_TAGS and sql.client.bulk_scene_tags))
    if not direct:
        if store.cleanup_pending:
            raise RuntimeError('Interrupted cleanup needs the same bulk API capabilities before it can resume.')
        global_labels, common_cores, names, core_names = frozenset(), frozenset(), set(), {}
    if direct and (global_labels or common_cores or store.cleanup_pending):
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
    view = SQLPreview(sql.stash, app, names, core_names) if app.DRY_RUN else sql.stash
    return sorted(candidates), view
