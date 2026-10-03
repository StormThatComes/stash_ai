"""Shared cleanup tag catalog and lazy publication masks. No model/media IO."""
from __future__ import annotations

from tqdm import tqdm

from stash_ai.state_codec import encode, decode
from stash_ai.taggers import SOURCES

PAGE_SIZE = 1000


def initialize(store, previous_version):
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
