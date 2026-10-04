"""Derived marker groups. Never model evidence, scene tags, or marker ownership."""

from collections import Counter
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
