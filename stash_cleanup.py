"""Tag-directed cleanup and compact, lazy publication masks. No model/media IO."""
from __future__ import annotations

from copy import deepcopy
from collections import Counter

from tqdm import tqdm

from stash_taggers import SOURCES

PAGE_SIZE = 1000
TAG_BATCH_SIZE = 100
WRITE_BATCH_SIZE = 200


def batches(values, size):
    values = list(values)
    for offset in range(0, len(values), size):
        yield values[offset:offset + size]


def full_scope(app, settings, args):
    return (not args.scene_id and app.APPLY_SCENE_TAGS and app.CREATE_MARKERS
            and {config.source.key for config in settings} == set(SOURCES))


def initialize(store, previous_version):
    from stash_merge import decode
    store.connection.executescript('''
        CREATE TABLE IF NOT EXISTS cleanup_state (
            scope TEXT PRIMARY KEY, generation INTEGER NOT NULL, data BLOB NOT NULL,
            pending TEXT);
    ''')
    if previous_version < 5:
        store.connection.execute('PRAGMA user_version=5')
    row = store.connection.execute(
        'SELECT generation,data,pending FROM cleanup_state WHERE scope=?', (store.scope,)).fetchone()
    store.cleanup_generation, store.cleanup_masks, store.cleanup_pending = (
        (row[0], decode(row[1]), row[2]) if row else (0, {}, None))


def save_state(store):
    from stash_merge import encode
    with store.connection:
        store.connection.execute('''INSERT INTO cleanup_state VALUES (?,?,?,?)
            ON CONFLICT(scope) DO UPDATE SET generation=excluded.generation,
            data=excluded.data,pending=excluded.pending''',
            (store.scope, store.cleanup_generation, encode(store.cleanup_masks), store.cleanup_pending))
    store.cache.clear()


def begin(store, fingerprint, global_labels, core_labels, tag_ids, *, kinds=('scene', 'markers')):
    if store.cleanup_pending and store.cleanup_pending != fingerprint:
        raise RuntimeError('Global cleanup is interrupted; rerun the same --models all cleanup mode '
                           'with the same exclusion policies before changing them.')
    if not store.cleanup_pending:
        store.cleanup_generation += 1
    store.cleanup_pending = fingerprint
    scope = '' if set(kinds) == {'scene', 'markers'} else ('marker_' if 'markers' in kinds else 'scene_')
    if scope:
        # Old executables do not understand output-specific lazy masks.
        if store.connection.execute('PRAGMA user_version').fetchone()[0] < 6:
            store.connection.execute('PRAGMA user_version=6')
    for kind, values in ((scope + 'global', global_labels), ('core', core_labels), (scope + 'ids', tag_ids)):
        store.cleanup_masks.setdefault(kind, {}).update(
            (str(value), store.cleanup_generation) for value in values)
    # Commit intent before the first API mutation. A retry discovers remaining
    # objects again instead of replaying IDs that may already have been deleted.
    save_state(store)


def finish(store):
    store.cleanup_pending = None
    save_state(store)


def mask_snapshot(store, snapshot, kind):
    """Apply only cleanups newer than this publication; keep raw scores intact."""
    if snapshot is None or snapshot.get('cleanup_generation', 0) >= store.cleanup_generation:
        return snapshot
    generation = snapshot.get('cleanup_generation', 0)
    abandoned = getattr(store, 'abandoned_cleanups', {})
    active = {key: {label for label, revision in values.items() if revision > generation and revision not in abandoned}
              for key, values in store.cleanup_masks.items()}
    prefix = 'marker_' if kind == 'markers' else 'scene_'
    blocked = active.get('global', set()) | active.get(prefix + 'global', set())
    cores = active.get('core', set())
    ids = active.get('ids', set()) | active.get(prefix + 'ids', set())
    result = {**snapshot, 'cleanup_generation': store.cleanup_generation}
    if any(revision > generation and kind in kinds for revision, kinds in abandoned.items()):
        result['recovery_required'] = True
    contributions = []
    for item in snapshot.get('contributions', []):
        if item['tag'] in blocked or (kind == 'markers' and item['tag'] in cores):
            continue
        item = dict(item)
        if kind == 'markers':
            for field in ('extra_tags', 'extra_ids'):
                if field in item:
                    excluded = blocked if field == 'extra_tags' else ids
                    item[field] = [value for value in item[field] if str(value) not in excluded]
            if 'extra_support' in item:
                item['extra_support'] = {tag: proof for tag, proof in item['extra_support'].items() if tag not in blocked}
            if 'extra_evidence' in item:
                item['extra_evidence'] = [proof for proof in item['extra_evidence'] if proof[0] not in blocked]
        if item.get('build_options'):
            options = dict(item['build_options'])
            fields = (('tag_exclusions', blocked), ('core_exclusions', cores)) if kind == 'markers' else (
                ('scene_exclusions', blocked),)
            for field, labels in fields:
                options[field] = {source: sorted(set(options.get(field, {}).get(source, [])) | labels)
                                  for source in SOURCES}
            item['build_options'] = options
        contributions.append(item)
    result['contributions'] = contributions
    if 'owned_ids' in result:
        result['owned_ids'] = [identifier for identifier in result['owned_ids'] if str(identifier) not in ids]
    return result


def stamp_snapshot(store, snapshot):
    snapshot = {key: value for key, value in snapshot.items() if key != 'recovery_required'}
    return {**snapshot, 'cleanup_generation': store.cleanup_generation} if store.cleanup_generation else snapshot


class CleanupClient:
    def __init__(self, stash):
        self.stash = stash
        data = stash.gql('''query StashAICleanupCapabilities {
            query: __type(name: "Query") { fields { name args { name } } }
            mutation: __type(name: "Mutation") { fields { name } }
            marker_filter: __type(name: "SceneMarkerFilterType") { inputFields { name } }
            scene_filter: __type(name: "SceneFilterType") { inputFields { name } }
            criterion: __type(name: "HierarchicalMultiCriterionInput") { inputFields { name } }
            bulk_marker: __type(name: "BulkSceneMarkerUpdateInput") { inputFields { name } }
            bulk_scene: __type(name: "BulkSceneUpdateInput") { inputFields { name } }
            bulk_ids: __type(name: "BulkUpdateIds") { inputFields { name } }
        }''')
        queries = {f['name']: {arg['name'] for arg in f.get('args', [])}
                   for f in (data.get('query') or {}).get('fields', [])}
        mutations = {f['name'] for f in (data.get('mutation') or {}).get('fields', [])}
        self.bulk_tags = 'tagsDestroy' in mutations
        self.bulk_markers = 'sceneMarkersDestroy' in mutations
        self.sql_available = 'querySQL' in mutations
        bulk_ids = {'ids', 'mode'} <= {f['name'] for f in (data.get('bulk_ids') or {}).get('inputFields', [])}
        self.bulk_marker_tags = (bulk_ids and 'bulkSceneMarkerUpdate' in mutations and
            {'ids', 'tag_ids'} <= {f['name'] for f in (data.get('bulk_marker') or {}).get('inputFields', [])})
        self.bulk_scene_tags = (bulk_ids and 'bulkSceneUpdate' in mutations and
            {'ids', 'tag_ids'} <= {f['name'] for f in (data.get('bulk_scene') or {}).get('inputFields', [])})
        self.supported = (
            {'ids', 'filter'} <= queries.get('findTags', set())
            and {'ids', 'filter', 'scene_marker_filter'} <= queries.get('findSceneMarkers', set())
            and {'scene_filter', 'filter'} <= queries.get('findScenes', set())
            and all('tags' in {f['name'] for f in (data.get(key) or {}).get('inputFields', [])}
                    for key in ('marker_filter', 'scene_filter'))
            and {'depth', 'value', 'modifier'} <= {f['name'] for f in (data.get('criterion') or {}).get('inputFields', [])}
            and (self.bulk_tags or 'tagDestroy' in mutations)
            and (self.bulk_markers or 'sceneMarkerDestroy' in mutations))

    def catalog(self):
        catalog = {}
        with tqdm(desc='Read cleanup tags', unit='tag', dynamic_ncols=True) as bar:
            page = 1
            while True:
                count = 'count' if page == 1 else ''
                response = self.stash.gql('''query StashAICleanupTags($filter: FindFilterType!) {
                    findTags(filter: $filter) { ''' + count + ''' tags { id name } }
                }''', {'filter': self.filter(page)})['findTags']
                for tag in response['tags']:
                    if not isinstance(tag.get('name'), str) or not tag.get('id'):
                        raise ValueError('Incomplete tag catalog; cleanup stopped.')
                    catalog[str(tag['id'])] = tag['name']
                bar.total = max(response.get('count', bar.total or 0), bar.n + len(response['tags']))
                bar.update(len(response['tags']))
                if len(response['tags']) < PAGE_SIZE:
                    return catalog
                page += 1

    @staticmethod
    def filter(page=1, size=None):
        return dict(page=page, per_page=PAGE_SIZE if size is None else size, sort='id', direction='ASC')

    @staticmethod
    def criterion(ids):
        # Explicit depth zero: matching a blocked parent must not select children.
        return {'tags': {'value': list(ids), 'modifier': 'INCLUDES', 'depth': 0}}

    def matching(self, ids, *, markers):
        query = ('''query StashAICleanupMarkers($filter: FindFilterType!, $where: SceneMarkerFilterType!) {
            findSceneMarkers(filter: $filter, scene_marker_filter: $where) {
                COUNT_FIELD scene_markers { id title primary_tag { id name } tags { id name } scene { id } }
            }
        }''' if markers else '''query StashAICleanupScenes($filter: FindFilterType!, $where: SceneFilterType!) {
            findScenes(filter: $filter, scene_filter: $where) { COUNT_FIELD scenes { id } }
        }''')
        key, field = ('findSceneMarkers', 'scene_markers') if markers else ('findScenes', 'scenes')
        description = 'Find affected markers' if markers else 'Find affected scenes'
        with tqdm(desc=description, unit='marker' if markers else 'scene', dynamic_ncols=True) as bar:
            for group in batches(sorted(ids), TAG_BATCH_SIZE):
                page = 1
                while True:
                    response = self.stash.gql(query.replace('COUNT_FIELD', 'count' if page == 1 else ''),
                                              {'filter': self.filter(page), 'where': self.criterion(group)})[key]
                    rows = response[field]
                    # Counts across tag groups may overlap; the caller deduplicates IDs.
                    if page == 1:
                        bar.total = (bar.total or 0) + response['count']
                    bar.total = max(bar.total or 0, bar.n + len(rows))
                    bar.update(len(rows))
                    for row in rows:
                        yield row
                    if len(rows) < PAGE_SIZE:
                        break
                    page += 1

    def delete_markers(self, ids, eligible, stats):
        description = 'Preview blocked markers' if self.stash.dry_run else 'Delete blocked markers'
        with tqdm(total=len(ids), desc=description, unit='marker', dynamic_ncols=True) as bar:
            for group in batches(sorted(ids), WRITE_BATCH_SIZE):
                if self.stash.dry_run:
                    current = group
                else:
                    rows = self.stash.gql('''query StashAICleanupMarkerBatch($ids: [ID!]!, $filter: FindFilterType!) {
                        findSceneMarkers(ids: $ids, filter: $filter) {
                            scene_markers { id title primary_tag { id name } tags { id name } scene { id } }
                        }
                    }''', {'ids': group, 'filter': self.filter(size=len(group))})['findSceneMarkers']['scene_markers']
                    if any(str(row['id']) not in group for row in rows):
                        raise ValueError('Unexpected marker IDs returned by Stash; cleanup stopped.')
                    current = [str(row['id']) for row in rows if eligible(row)]
                    stats['cleanup_markers_changed_or_gone'] += len(group) - len(current)
                    if current and self.bulk_markers:
                        result = self.stash.gql('''mutation StashAICleanupDeleteMarkers($ids: [ID!]!) {
                            sceneMarkersDestroy(ids: $ids)
                        }''', {'ids': current})['sceneMarkersDestroy']
                        if result is not True:
                            raise RuntimeError('Marker deletion was not confirmed; rerun the same cleanup to reconcile.')
                    elif current:
                        for identifier in current:
                            self.stash.delete_marker(identifier)
                stats['markers_deleted'] += len(current)
                bar.update(len(group))

    def delete_tags(self, tags, stats):
        description = 'Preview blacklisted tags' if self.stash.dry_run else 'Delete blacklisted tags'
        with tqdm(total=len(tags), desc=description, unit='tag', dynamic_ncols=True) as bar:
            for group in batches(sorted(tags), WRITE_BATCH_SIZE):
                if self.stash.dry_run:
                    current = group
                else:
                    rows = self.stash.gql('''query StashAICleanupTagBatch($ids: [ID!]!, $filter: FindFilterType!) {
                        findTags(ids: $ids, filter: $filter) { tags { id name } }
                    }''', {'ids': group, 'filter': self.filter(size=len(group))})['findTags']['tags']
                    if any(str(row['id']) not in group for row in rows):
                        raise ValueError('Unexpected tag IDs returned by Stash; cleanup stopped.')
                    current = [str(row['id']) for row in rows if row['name'] == tags[str(row['id'])]]
                    stats['cleanup_tags_changed_or_gone'] += len(group) - len(current)
                    if current and self.bulk_tags:
                        result = self.stash.gql('''mutation StashAICleanupDeleteTags($ids: [ID!]!) {
                            tagsDestroy(ids: $ids)
                        }''', {'ids': current})['tagsDestroy']
                        if result is not True:
                            raise RuntimeError('Tag deletion was not confirmed; rerun the same cleanup to reconcile.')
                    elif current:
                        for identifier in current:
                            result = self.stash.gql('''mutation StashAICleanupDeleteTag($input: TagDestroyInput!) {
                                tagDestroy(input: $input)
                            }''', {'input': {'id': identifier}})['tagDestroy']
                            if result is not True:
                                raise RuntimeError('Tag deletion was not confirmed; rerun the same cleanup to reconcile.')
                stats['blacklisted_tags_deleted'] += len(current)
                bar.update(len(group))


class CleanupPreview:
    """Present simulated global deletions to later, scoped dry-run reconciliation."""
    def __init__(self, stash, tags, markers, group_usage_delta=None):
        self.stash, self.tags, self.markers = stash, set(tags), set(markers)
        self.marker_group_usage_delta = group_usage_delta or Counter()

    def __getattr__(self, name):
        return getattr(self.stash, name)

    def find_scene(self, identifier):
        scene = deepcopy(self.stash.find_scene(identifier))
        scene['tags'] = [tag for tag in scene.get('tags', []) if str(tag['id']) not in self.tags]
        scene['scene_markers'] = [m for m in scene.get('scene_markers', []) if str(m['id']) not in self.markers]
        for marker in scene['scene_markers']:
            marker['tags'] = [tag for tag in marker.get('tags', []) if str(tag['id']) not in self.tags]
        return scene

    def iter_scenes(self, ids):
        for identifier in ids:
            yield self.find_scene(identifier)


def global_cleanup(app, client, store, core_policies, tag_policies, stats):
    """Return only scenes needing source-specific reconciliation and its API view."""
    from stash_merge import AI
    from stash_progress import digest
    global_labels = frozenset.intersection(*(tag_policies[source] for source in SOURCES))
    common_cores = frozenset.intersection(*(core_policies[source] for source in SOURCES)) - global_labels
    sources = [AI, *SOURCES.values()]
    names = {source.prefix + label for source in sources for label in global_labels}
    core_names = {source.prefix + label: source for source in sources for label in common_cores}
    remaining_tags = frozenset.union(*(tag_policies[s] - global_labels for s in SOURCES))
    remaining_cores = frozenset.union(*(core_policies[s] - common_cores - global_labels for s in SOURCES))
    scene_names = {source.prefix + label for source in sources for label in remaining_tags}
    marker_names = scene_names | {source.prefix + label for source in sources
                                 for label in remaining_cores | app.MARKER_TAG_BLOCKLIST}
    fingerprint = digest({'tags': {s: sorted(tag_policies[s]) for s in SOURCES},
                          'cores': {s: sorted(core_policies[s]) for s in SOURCES},
                          'marker_blocklist': sorted(app.MARKER_TAG_BLOCKLIST)})
    if store.cleanup_pending and store.cleanup_pending != fingerprint:
        raise RuntimeError('Global cleanup is interrupted; rerun --models all --cleanup-only '
                           'with the same exclusion policies before changing them.')
    if store.has_pending_publications():
        raise RuntimeError('An interrupted merge is pending; run --merge-only before cleanup.')
    catalog = client.catalog()
    tags = {identifier: name for identifier, name in catalog.items() if name in names}
    if global_labels or common_cores or store.cleanup_pending:
        begin(store, fingerprint, global_labels, common_cores, tags)

    def eligible(marker):
        name = marker['primary_tag']['name']
        return name in names or (name in core_names and app.owned_marker(marker, core_names[name]))

    marker_ids, scenes = set(), set()
    group_usage_delta = Counter()
    matching_names = names | set(core_names) | marker_names
    find_ids = {identifier for identifier, name in catalog.items() if name in matching_names}
    # Complete pagination before mutation to avoid skipping rows as pages shrink.
    for marker in client.matching(find_ids, markers=True) if find_ids else ():
        if eligible(marker):
            if client.stash.dry_run and str(marker['id']) not in marker_ids:
                from stash_marker_groups import PREFIX
                group_usage_delta.subtract({str(t['id']) for t in marker.get('tags', []) if t['name'].startswith(PREFIX)})
            marker_ids.add(str(marker['id']))
        elif any(tag['name'] in marker_names for tag in [marker['primary_tag'], *marker.get('tags', [])]):
            scenes.add(str(marker['scene']['id']))
    scene_tag_ids = {identifier for identifier, name in catalog.items() if name in scene_names}
    for scene in client.matching(scene_tag_ids, markers=False) if scene_tag_ids else ():
        scenes.add(str(scene['id']))
    client.delete_markers(marker_ids, eligible, stats)
    # Stash removes scene/extra-tag associations and hierarchy edges by cascade.
    # Children are not deleted, and primary-tag references were handled above.
    client.delete_tags(tags, stats)
    if store.cleanup_pending:
        finish(store)
    stats['cleanup_candidate_scenes'] = len(scenes)
    stats['cleanup_fast_path'] = 1
    view = CleanupPreview(client.stash, tags, marker_ids, group_usage_delta) if client.stash.dry_run else client.stash
    return sorted(scenes), view
