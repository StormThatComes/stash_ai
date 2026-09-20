"""Shared offline helpers; no test collection or fixture registration."""
from copy import deepcopy

import pytest
import yaml

import stash_ai as app
from tests.helpers.stash import Stash


class FastStash(Stash):
    """Models Stash's primary-tag restriction, cascades and filtered pagination."""
    def __init__(self, dry=False, bulk=True):
        super().__init__(dry)
        self.scenes = {'1': self.scene}
        self.images = []
        self.bulk, self.calls, self.scene_reads = bulk, [], []
        self.fail_after = None
        self.before_batch = None
        self.last_tag = 0
        self.last_marker = 1000

    def tag(self, name):
        if name not in self.tags:
            self.last_tag = max(self.last_tag, max((int(t['id']) for t in self.tags.values()), default=0)) + 1
            self.tags[name] = dict(id=str(self.last_tag), name=name, parents=[], description=None)
        return self.tags[name]

    def find_scene(self, identifier):
        self.scene_reads.append(str(identifier))
        return deepcopy(self.scenes[str(identifier)])

    def iter_scenes(self, identifiers=None):
        for identifier in identifiers or self.scenes:
            yield self.find_scene(identifier)

    def update_scene_tags(self, identifier, ids):
        assert not self.dry_run
        self.writes.append(('scene', ids))
        self.scenes[str(identifier)]['tags'] = [self.find_tag_by_id(i) for i in ids]

    def create_marker(self, payload):
        assert not self.dry_run
        self.writes.append(('create', payload))
        self.last_marker = max(self.last_marker, max((int(m['id']) for m in self.marker_rows()
                               if str(m['id']).isdigit()), default=0)) + 1
        self.scenes[str(payload['scene_id'])]['scene_markers'].append(self.record(payload, str(self.last_marker)))

    def update_marker(self, payload):
        assert not self.dry_run
        self.writes.append(('update', payload))
        scene = self.scenes[str(payload['scene_id'])]
        scene['scene_markers'] = [self.record(payload, m['id']) if str(m['id']) == payload['id'] else m
                                  for m in scene['scene_markers']]

    def delete_marker(self, identifier):
        assert not self.dry_run
        self.writes.append(('delete', identifier))
        for scene in self.scenes.values():
            scene['scene_markers'] = [m for m in scene['scene_markers'] if str(m['id']) != identifier]

    def marker_rows(self):
        return [dict(deepcopy(marker), scene={'id': str(scene['id'])})
                for scene in self.scenes.values() for marker in scene['scene_markers']]

    def remove_tags(self, identifiers):
        assert not self.dry_run
        identifiers = set(identifiers)
        assert all(row['primary_tag']['id'] not in identifiers for row in self.marker_rows())
        for identifier in identifiers:
            tag = self.find_tag_by_id(identifier)
            self.writes.append(('delete_tag', tag['name']))
            del self.tags[tag['name']]
        for scene in self.scenes.values():
            scene['tags'] = [t for t in scene['tags'] if t['id'] not in identifiers]
            for marker in scene['scene_markers']:
                marker['tags'] = [t for t in marker['tags'] if t['id'] not in identifiers]
        for tag in self.tags.values():
            tag['parents'] = [t for t in tag['parents'] if t['id'] not in identifiers]
        for image in self.images:
            image['tags'] = [t for t in image['tags'] if t['id'] not in identifiers]

    def gql(self, query, variables=None):
        op = query.split('StashAI', 1)[1].split('(', 1)[0].split()[0]
        self.calls.append((op, deepcopy(variables)))
        if op == 'CleanupCapabilities':
            queries = {'findTags': ['ids', 'filter'], 'findSceneMarkers': ['ids', 'filter', 'scene_marker_filter'],
                       'findScenes': ['scene_filter', 'filter']}
            mutations = ['tagDestroy', 'sceneMarkerDestroy'] + (['tagsDestroy', 'sceneMarkersDestroy'] if self.bulk else [])
            return dict(query={'fields': [{'name': k, 'args': [{'name': a} for a in args]} for k, args in queries.items()]},
                mutation={'fields': [{'name': name} for name in mutations]},
                marker_filter={'inputFields': [{'name': 'tags'}]}, scene_filter={'inputFields': [{'name': 'tags'}]},
                criterion={'inputFields': [{'name': name} for name in ['value', 'modifier', 'depth']]})
        if op in {'CleanupTags', 'CleanupTagBatch'}:
            if self.before_batch and op == 'CleanupTagBatch':
                self.before_batch(self)
                self.before_batch = None
            rows = list(self.tags.values())
            if 'ids' in variables:
                rows = [row for row in rows if row['id'] in variables['ids']]
            key, field = 'findTags', 'tags'
        elif op in {'CleanupMarkers', 'CleanupMarkerBatch'}:
            rows = self.marker_rows()
            if 'ids' in variables:
                rows = [row for row in rows if row['id'] in variables['ids']]
            else:
                criterion = variables['where']['tags']
                assert criterion['modifier'] == 'INCLUDES' and criterion['depth'] == 0
                rows = [row for row in rows if any(tag['id'] in criterion['value']
                        for tag in [row['primary_tag'], *row['tags']])]
            key, field = 'findSceneMarkers', 'scene_markers'
        elif op == 'CleanupScenes':
            criterion = variables['where']['tags']
            assert criterion['depth'] == 0 and criterion['modifier'] == 'INCLUDES'
            rows = [scene for scene in self.scenes.values() if any(t['id'] in criterion['value'] for t in scene['tags'])]
            key, field = 'findScenes', 'scenes'
        elif op == 'CleanupDeleteMarkers':
            for identifier in variables['ids']:
                self.delete_marker(identifier)
            if self.fail_after == op:
                self.fail_after = None
                raise RuntimeError('Connection lost after committed marker batch')
            return {'sceneMarkersDestroy': True}
        elif op in {'CleanupDeleteTags', 'CleanupDeleteTag'}:
            ids = variables['ids'] if 'ids' in variables else [variables['input']['id']]
            self.remove_tags(ids)
            if self.fail_after == op:
                self.fail_after = None
                raise RuntimeError('Connection lost after committed tag batch')
            return {'tagsDestroy' if 'ids' in variables else 'tagDestroy': True}
        else:
            return super().gql(query, variables)
        rows.sort(key=lambda row: str(row['id']))
        f = variables['filter']; offset = (f['page'] - 1) * f['per_page']
        return {key: {'count': len(rows), field: deepcopy(rows[offset:offset + f['per_page']])}}


def options(tmp_path, labels=(), cores=(), by_model=None, *, output='merged'):
    tags, core = tmp_path/'tags.yml', tmp_path/'cores.yml'
    tags.write_text(yaml.safe_dump(dict(version=1, excluded_tags=list(labels), by_model=by_model or {})))
    core.write_text(yaml.safe_dump(dict(version=1, excluded_core_tags=list(cores))))
    return ['--models', 'all', '--output-mode', output, '--no-hierarchy', '--cleanup-only',
            '--cleanup-strategy', 'legacy',
            '--state-db', str(tmp_path/'state.sqlite3'), '--tag-exclusions', str(tags),
            '--marker-core-exclusions', str(core)]


def install(monkeypatch, stash):
    monkeypatch.setattr(app, 'StashClient', lambda *_: stash)
    for name in ('load_tagger', 'load_vocabulary', 'fetch_cues', 'iter_sprite_frames'):
        monkeypatch.setattr(app, name, lambda *args: pytest.fail('cleanup accessed model or media assets'))
