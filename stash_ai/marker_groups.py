"""Derived marker groups. Never model evidence, scene tags, or marker ownership."""

from stash_ai.publication import markers as marker_ops
from collections import Counter
from copy import deepcopy
import json


ROOT = 'ai_markers'
PREFIX = 'm:'
# User-facing spelling is intentional and is part of the persisted naming contract.
OCCURRENCE_SUFFIX = ':occurance'
DESCRIPTION = '[Stash AI marker group v1] '
ROOT_DESCRIPTION = '[Stash AI marker groups v1] Organizational root.'


def group_name(label, occurrence=False):
    if not isinstance(label, str) or not label or label != label.strip() or '\n' in label or '\r' in label:
        raise ValueError('Marker groups require a non-empty raw core label.')
    return PREFIX + label + (OCCURRENCE_SUFFIX if occurrence else '')


def group_description(label, occurrence=False):
    return DESCRIPTION + json.dumps({'core': label, 'occurrence': occurrence},
                                   ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def managed_group(tag):
    """A prefix alone never authorizes changing or deleting somebody else's tag."""
    description = tag.get('description') or ''
    if not description.startswith(DESCRIPTION):
        return False
    try:
        data = json.loads(description[len(DESCRIPTION):])
        return (isinstance(data, dict) and set(data) == {'core', 'occurrence'}
                and type(data['occurrence']) is bool
                and tag.get('name') == group_name(data['core'], data['occurrence']))
    except (ValueError, TypeError, KeyError):
        return False


class MarkerGroups:
    def __init__(self, app, resolver):
        self.app, self.resolver, self.stash = app, resolver, resolver.stash
        self.resolved = {}
        self.root = None
        self.usage_delta = Counter()

    def _create(self, name, description, parents):
        if not self.app.CREATE_MISSING:
            raise ValueError(f'Missing marker group tag {name!r}; enable CREATE_MISSING.')
        if self.stash.dry_run:
            tag = dict(id='dry-run:' + name, name=name, description=description,
                       parents=parents, ignore_auto_tag=True)
            if self.resolver.report:
                self.resolver.report(f'[DRY] Create marker group tag {name!r}.')
        else:
            tag = self.stash.create_tag(name, description=description,
                parent_ids=[str(p['id']) for p in parents], ignore_auto_tag=True)
        return self.resolver._remember(tag)

    def _root(self):
        if self.root is None:
            # An existing user-created organizational root can be reused; never
            # claim or delete it, or change its other hierarchy relationships.
            self.root = self.resolver._find(ROOT) or self._create(ROOT, ROOT_DESCRIPTION, [])
        return self.root

    def resolve(self, label, occurrence=False):
        key = label, occurrence
        if key in self.resolved:
            return self.resolved[key]
        root = self._root()
        name, description = group_name(label, occurrence), group_description(label, occurrence)
        tag = self.resolver._find(name)
        if tag is None:
            tag = self._create(name, description, [root])
        else:
            # The suffix can collide with a literal raw label ending in
            # ':occurance'. Fail explicitly instead of mixing the two groups.
            if not managed_group(tag) or tag.get('description') != description:
                raise ValueError(f'Marker group name collision: {name!r}. '
                                 'The existing tag belongs to another label/type or is not managed by Stash AI.')
            wanted = {str(p['id']) for p in tag.get('parents', [])} | {str(root['id'])}
            if wanted != {str(p['id']) for p in tag.get('parents', [])} or not tag.get('ignore_auto_tag'):
                if self.stash.dry_run:
                    if self.resolver.report:
                        self.resolver.report(f'[DRY] Repair marker group hierarchy/settings for {name!r}.')
                    parents = {str(p['id']): p for p in tag.get('parents', [])}
                    parents[str(root['id'])] = root
                    tag = {**tag, 'parents': list(parents.values()), 'ignore_auto_tag': True}
                else:
                    tag = self.stash.find_tag_by_id(str(tag['id']))
                    if not managed_group(tag) or tag.get('description') != description:
                        raise ValueError(f'Marker group ownership changed for {name!r}.')
                    wanted = {str(p['id']) for p in tag.get('parents', [])} | {str(root['id'])}
                    tag = self.stash.update_tag_organization(str(tag['id']), sorted(wanted), True)
                self.resolver._remember(tag)
        self.resolved[key] = str(tag['id'])
        return self.resolved[key]

    def owns(self, tag):
        if not tag.get('name', '').startswith(PREFIX):
            return False
        full = self.resolver._find(tag['name'])
        return bool(full and str(full['id']) == str(tag['id']) and managed_group(full))

    def owns_id(self, identifier):
        name = self.resolver.name_for_id(str(identifier))
        return bool(name and self.owns({'id': str(identifier), 'name': name}))

    def record_preview(self, scene, creates, updates, deletes):
        if not self.stash.dry_run:
            return
        previous = {str(m['id']): m for m in scene.get('scene_markers', [])}
        def count(ids, change):
            for identifier in set(map(str, ids)):
                if self.owns_id(identifier):
                    self.usage_delta[identifier] += change
        for payload in creates:
            count(payload['tag_ids'], 1)
        for payload in updates:
            count((t['id'] for t in previous[payload['id']].get('tags', [])), -1)
            count(payload['tag_ids'], 1)
        for identifier in deletes:
            count((t['id'] for t in previous[str(identifier)].get('tags', [])), -1)


def for_resolver(app, resolver):
    if not hasattr(resolver, 'marker_groups'):
        resolver.marker_groups = MarkerGroups(app, resolver)
    return resolver.marker_groups


class PreviewUsage:
    """Apply planned membership deltas to native, global tag usage counters."""
    def __init__(self, stash, delta):
        self.stash, self.delta = stash, delta

    def __getattr__(self, name):
        return getattr(self.stash, name)

    def gql(self, query, variables=None):
        data = self.stash.gql(query, variables)
        if not any(name in query for name in ('StashAIPruneCandidates', 'StashAIPruneCatalog', 'StashAIPruneTag(')):
            return data
        data = deepcopy(data)
        tags = data.get('findTags', {}).get('tags', [])
        if data.get('findTag'):
            tags = [data['findTag']]
        for tag in tags:
            if type(tag.get('scene_marker_count')) is int:
                tag['scene_marker_count'] = max(0, tag['scene_marker_count'] + self.delta.get(str(tag['id']), 0))
        return data


def prune_groups(stash, report=print, *, candidate_ids=None, usage_delta=None):
    """Only owned, globally unused leaf groups are pruned; the root is retained."""
    from stash_ai.cleanup.prune import prune_tags
    if candidate_ids is None:
        candidate_ids = [str(t['id']) for t in stash.iter_tags() if t['name'].startswith(PREFIX)]
    if not candidate_ids:
        return Counter()
    view = PreviewUsage(stash, usage_delta) if stash.dry_run and usage_delta else stash
    stats = Counter(prune_tags(view, [PREFIX], report=report, candidate_ids=candidate_ids))
    return Counter(marker_group_tags_pruned=stats['tags_pruned'],
                   marker_group_tags_retained=stats['tags_retained_after_refresh'])


def run_backfill(app, args):
    """Repair group membership on existing owned markers, without source replay."""
    from tqdm import tqdm
    from stash_ai.merge import AI
    from stash_ai.storage.store import StateStore, scope_for
    from stash_ai.occurrences import OCCURRENCE_TAG
    stats, report = Counter(), args.report
    with StateStore(args.state_db, scope_for(app.STASH_URL), app.DRY_RUN) as store:
        if store.has_pending_publications():
            raise RuntimeError('An interrupted merge is pending; run --merge-only before synchronizing marker groups.')
        stash = app.create_stash_client(app.DRY_RUN)
        try:
            resolver = app.TagResolver(stash, source=AI, report=report.detail, create_missing=app.CREATE_MISSING)
            resolver.preload()
            groups = for_resolver(app, resolver)
            for row in tqdm(stash.iter_scenes(args.scene_id), desc='Marker groups', unit='scene'):
                try:
                    scene = stash.find_scene(str(row['id']))
                    updates = []
                    for marker in scene.get('scene_markers', []):
                        core = marker['primary_tag']['name']
                        if not marker_ops.owned_marker(marker, AI) or not core.startswith(AI.prefix):
                            continue
                        occurrence = any(t['name'] == OCCURRENCE_TAG for t in marker.get('tags', []))
                        desired = groups.resolve(core[len(AI.prefix):], occurrence)
                        tags = {str(t['id']) for t in marker.get('tags', []) if not groups.owns(t)} | {desired}
                        if tags == {str(t['id']) for t in marker.get('tags', [])}:
                            stats['markers_unchanged'] += 1
                            continue
                        payload = dict(id=str(marker['id']), scene_id=str(scene['id']), title=marker['title'],
                            seconds=marker['seconds'], primary_tag_id=str(marker['primary_tag']['id']), tag_ids=sorted(tags))
                        payload['end_seconds'] = marker.get('end_seconds')
                        updates.append(payload)
                    groups.record_preview(scene, [], updates, [])
                    for payload in updates:
                        if stash.dry_run:
                            report.detail(f"[DRY] Update marker group for marker {payload['id']} in scene {scene['id']}.")
                        else:
                            stash.update_marker(payload)
                    stats['markers_updated'] += len(updates)
                    stats['processed'] += 1
                except Exception as exc:
                    stats['errors'] += 1
                    report.issue('marker groups', '', row['id'], exc, app.safe_error(exc, api_key=app.STASH_API_KEY))
            if not stats['errors']:
                stats.update(prune_groups(stash, report.detail, usage_delta=groups.usage_delta))
        finally:
            stash.close()
    report.finish('Marker groups' + (' (planned)' if app.DRY_RUN else ''), stats)
    return int(bool(stats['errors']))
