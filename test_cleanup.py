"""Offline API-contract, scale and recovery tests for tag-directed cleanup."""
from copy import deepcopy
import sqlite3

import pytest
import yaml

import stash_ai as app
import stash_cleanup as cleanup
import stash_merge as merge
from stash_taggers import WD
from test_multimodel import Stash
from test_progress import pipeline, reset
from test_merge import span, hit


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


@pytest.mark.parametrize('output', ['merged', 'separate'])
def test_global_cleanup_of_9000_scenes_never_fetches_scenes_or_predictions(tmp_path, monkeypatch, capsys, output):
    stash = FastStash()
    blocked, human = stash.tag('ai:blocked'), stash.tag('blocked')
    for i in range(1, 9001):
        stash.scenes[str(i)] = dict(id=str(i), tags=[blocked, human], scene_markers=[])
    stash.scene = stash.scenes['1']
    stash.marker(merge.AI, 'blocked', 'global-manual', manual=True)
    keep = stash.marker(merge.AI, 'keep', 'keep')
    keep['tags'] += [stash.tag('op:blocked'), human]
    keep.update(seconds=10.125, end_seconds=63.625)
    stash.marker(WD, 'core_only', 'owned-core')
    stash.marker(WD, 'core_only', 'manual-core', manual=True)
    parent, child = stash.tag('jt:blocked'), stash.tag('jt:allowed_child')
    child['parents'] = [deepcopy(parent)]
    stash.images.append({'tags': [parent, human]})
    install(monkeypatch, stash)
    monkeypatch.setattr(merge.StateStore, 'records', lambda *a, **k: pytest.fail('global cleanup decoded results'))
    assert app.main(options(tmp_path, ['blocked'], ['core_only'], output=output)) == 0
    assert not stash.scene_reads
    assert all([t['name'] for t in s['tags']] == ['blocked'] for s in stash.scenes.values())
    assert {m['id'] for m in stash.scene['scene_markers']} == {'keep', 'manual-core'}
    assert (keep['seconds'], keep['end_seconds']) == (10.125, 63.625)
    assert {t['name'] for t in keep['tags']} == {merge.AI.owner_tag, 'blocked'}
    assert child['name'] in stash.tags and child['parents'] == []
    assert [t['name'] for t in stash.images[0]['tags']] == ['blocked']
    assert 'wd:core_only' in stash.tags
    assert len(stash.calls) < 15
    assert 'cleanup_candidate_scenes=0' in capsys.readouterr().out
    stash.writes.clear()
    assert app.main(options(tmp_path, ['blocked'], ['core_only'], output=output)) == 0
    assert stash.writes == [] and not stash.scene_reads


@pytest.mark.parametrize('bulk', [False, True])
def test_pagination_finishes_before_deletion_and_deduplicates_markers(tmp_path, monkeypatch, bulk):
    monkeypatch.setattr(cleanup, 'PAGE_SIZE', 3)
    monkeypatch.setattr(cleanup, 'TAG_BATCH_SIZE', 2)
    monkeypatch.setattr(cleanup, 'WRITE_BATCH_SIZE', 2)
    stash = FastStash(bulk=bulk)
    for i in range(17):
        marker = stash.marker(merge.AI, 'ban'+str(i % 5), str(i))
        marker['tags'] += [stash.tag('op:ban'+str(j)) for j in range(5)]
    install(monkeypatch, stash)
    assert app.main(options(tmp_path, ['ban'+str(i) for i in range(5)])) == 0
    assert not stash.scene['scene_markers']
    removed = [identifier for op, identifier in stash.writes if op == 'delete']
    assert len(removed) == len(set(removed)) == 17
    ops = [op for op, _ in stash.calls]
    if bulk:
        assert max(i for i, op in enumerate(ops) if op == 'CleanupMarkers') < ops.index('CleanupDeleteMarkers')
        assert len([op for op in ops if op == 'CleanupDeleteMarkers']) == 9
    assert not stash.scene_reads


def test_global_dry_run_changes_neither_stash_nor_existing_state(tmp_path, monkeypatch):
    stash = FastStash(dry=True)
    stash.marker(merge.AI, 'blocked', '1')
    stash.scene['tags'] = [stash.tag('op:blocked')]
    args = options(tmp_path, ['blocked'])
    path = tmp_path/'state.sqlite3'
    with merge.StateStore(path, merge.scope_for(app.STASH_URL)):
        pass
    before = deepcopy(stash.scenes), deepcopy(stash.tags), path.read_bytes(), path.stat().st_mtime_ns
    install(monkeypatch, stash)
    assert app.main([*args, '--dry-run']) == 0
    assert (stash.scenes, stash.tags, path.read_bytes(), path.stat().st_mtime_ns) == before
    assert stash.writes == []


@pytest.mark.parametrize('failure', ['CleanupDeleteMarkers', 'CleanupDeleteTags'])
def test_resume_reconciles_committed_batches_without_replaying_deleted_ids(tmp_path, monkeypatch, failure):
    stash = FastStash()
    stash.marker(merge.AI, 'blocked', '1')
    stash.scene['tags'] = [stash.tag('op:blocked')]
    stash.fail_after = failure
    install(monkeypatch, stash)
    args = options(tmp_path, ['blocked'])
    assert app.main(args) == 1
    path = tmp_path/'state.sqlite3'
    with pytest.raises(RuntimeError, match='Global cleanup is interrupted'):
        merge.StateStore(path, merge.scope_for(app.STASH_URL))
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT pending FROM cleanup_state').fetchone()[0]
    assert app.main(args) == 0
    assert not stash.scene['scene_markers'] and not stash.scene['tags']
    assert len([op for op, _ in stash.writes if op == 'delete']) == 1
    with merge.StateStore(path, merge.scope_for(app.STASH_URL)) as store:
        assert store.cleanup_generation == 1 and store.cleanup_pending is None


def test_changed_policy_during_interrupted_cleanup_is_rejected_before_mutations(tmp_path, monkeypatch):
    stash = FastStash(); stash.marker(merge.AI, 'blocked', '1')
    stash.fail_after = 'CleanupDeleteMarkers'
    install(monkeypatch, stash)
    assert app.main(options(tmp_path, ['blocked'])) == 1
    stash.writes.clear()
    assert app.main(options(tmp_path, ['different'])) == 1
    assert not stash.writes and 'ai:blocked' in stash.tags
    assert app.main(options(tmp_path, ['blocked'])) == 0


def test_pending_merge_blocks_global_deletion_before_first_mutation(tmp_path, monkeypatch):
    stash = FastStash(); stash.marker(merge.AI, 'blocked', '1')
    install(monkeypatch, stash)
    args = options(tmp_path, ['blocked'])
    with merge.StateStore(tmp_path/'state.sqlite3', merge.scope_for(app.STASH_URL)) as store:
        store.prepare('1', {'markers': {'contributions': []}})
    assert app.main(args) == 1
    assert not stash.writes and 'ai:blocked' in stash.tags


@pytest.mark.parametrize('temporal', [False, True])
def test_publication_masks_are_lazy_and_unbanning_restores_raw_evidence(pipeline, tmp_path, monkeypatch, temporal):
    p = pipeline
    monkeypatch.setattr(app, 'DEFAULT_TEMPORAL', temporal)
    assert app.main(['--models', 'all', *p.args[2:]]) == 0
    stash = FastStash()
    stash.tags, stash.scene = deepcopy(p.stash.tags), deepcopy(p.stash.scene)
    stash.scenes = {'1': stash.scene}
    monkeypatch.setattr(app, 'StashClient', lambda *_: stash)
    def blobs():
        with sqlite3.connect(p.path) as db:
            return (db.execute('SELECT data FROM results ORDER BY model,kind').fetchall(),
                    db.execute('SELECT published,pending FROM snapshots ORDER BY scene,kind').fetchall())
    before = blobs()
    args = options(tmp_path, ['common'])
    with monkeypatch.context() as patch:
        patch.setattr(merge, 'decode', lambda *_: pytest.fail('cleanup decompressed per-scene evidence'))
        assert app.main(args) == 0
    assert blobs() == before
    with merge.StateStore(p.path, merge.scope_for(app.STASH_URL), cache_bytes=2**20) as store:
        for kind in merge.KINDS:
            snapshot, _ = store.snapshot('1', kind)
            assert all(item['tag'] != 'common' for item in snapshot['contributions'])
            assert all('common' not in item.get('extra_tags', []) for item in snapshot['contributions'])
            assert store.snapshot('1', kind)[0] == snapshot
    options(tmp_path)  # Remove the ban; a normal merge may restore it from raw scores.
    merge_args = ['--models', 'all', *p.args[2:], '--merge-only', '--tag-exclusions', str(tmp_path/'tags.yml'),
                  '--marker-core-exclusions', str(tmp_path/'cores.yml')]
    monkeypatch.setattr(app, 'load_tagger', lambda *_: pytest.fail('unban reran inference'))
    assert app.main(merge_args) == 0
    assert 'ai:common' in {t['name'] for t in stash.scene['tags']}
    with merge.StateStore(p.path, merge.scope_for(app.STASH_URL)) as store:
        assert any(item['tag'] == 'common' for item in store.snapshot('1', 'scene')[0]['contributions'])
    assert blobs()[0] == before[0]
    assert app.main(options(tmp_path, ['common'])) == 0
    with merge.StateStore(p.path, merge.scope_for(app.STASH_URL)) as store:
        assert store.cleanup_generation == 2
        assert not any(item['tag'] == 'common' for item in store.snapshot('1', 'scene')[0]['contributions'])


def test_source_specific_cleanup_only_revisits_matching_scene_and_splits_bridge(tmp_path, monkeypatch):
    stash = FastStash()
    for i in range(2, 100):
        stash.scenes[str(i)] = dict(id=str(i), tags=[stash.tag('ai:blocked')], scene_markers=[])
    with merge.StateStore(tmp_path/'state.sqlite3', merge.scope_for(app.STASH_URL)) as store:
        merge.sync_merged(app, stash, app.TagResolver(stash, source=merge.AI), store, stash.find_scene('1'),
            [hit('wd'), hit('oppai')], [span('joytag', 10, 20), span('oppai', 20, 30), span('wd', 30, 40)])
    install(monkeypatch, stash)
    stash.scene_reads.clear()
    assert app.main(options(tmp_path, ['blocked'], by_model={'oppai': ['common']})) == 0
    assert set(stash.scene_reads) == {'1'}
    assert sorted((m['seconds'], m['end_seconds']) for m in stash.scene['scene_markers']) == [(10, 20), (30, 40)]
    assert 'ai:common' in stash.tags
    assert all(not s['tags'] for key, s in stash.scenes.items() if key != '1')


def test_tag_renamed_after_catalog_is_not_deleted(tmp_path, monkeypatch):
    stash = FastStash(); stash.tag('ai:blocked')
    def rename(client):
        tag = client.tags.pop('ai:blocked'); tag['name'] = 'ai:renamed'
        client.tags[tag['name']] = tag
    stash.before_batch = rename
    install(monkeypatch, stash)
    assert app.main(options(tmp_path, ['blocked'])) == 0
    assert 'ai:renamed' in stash.tags and not stash.writes


def test_partial_scope_keeps_compatible_scene_cleanup(pipeline, tmp_path, monkeypatch):
    p = pipeline
    assert app.main(p.args) == 0
    install(monkeypatch, p.stash)
    args = options(tmp_path, ['common'])
    assert app.main([*args, '--scene-id', '1']) == 0
    # The existing older-API fixture would fail any bulk cleanup operation.
    assert not any(t['name'] == 'ai:common' for t in p.stash.scene['tags'])


def test_lazy_global_mask_keeps_frozen_exclusions_during_later_source_cleanup(pipeline, tmp_path, monkeypatch):
    p = pipeline
    monkeypatch.setattr(app, 'DEFAULT_TEMPORAL', True)
    assert app.main(['--models', 'all', *p.args[2:]]) == 0
    stash = FastStash()
    stash.tags, stash.scene = deepcopy(p.stash.tags), deepcopy(p.stash.scene)
    stash.scenes = {'1': stash.scene}
    install(monkeypatch, stash)
    args = options(tmp_path, ['extra'])
    assert app.main(args) == 0
    options(tmp_path)
    (tmp_path/'cores.yml').write_text(yaml.safe_dump(dict(version=1, excluded_core_tags=[], by_model={'wd': ['common']})))
    assert app.main(args) == 0
    assert stash.scene['scene_markers']
    assert all(tag['name'] != 'ai:extra' for marker in stash.scene['scene_markers']
               for tag in [marker['primary_tag'], *marker['tags']])
    with merge.StateStore(p.path, merge.scope_for(app.STASH_URL)) as store:
        spans = store.snapshot('1', 'markers')[0]['contributions']
        assert spans and any(item.get('build_options') for item in spans)
        for item in spans:
            if item.get('build_options'):
                assert all('extra' in values for values in item['build_options']['tag_exclusions'].values())
