"""Offline API-contract, scale and recovery tests for SQL discovery and shared cleanup state."""
from copy import deepcopy
import sqlite3

import pytest
import yaml

from stash_ai import cli as app
from stash_ai import merge_sync
from stash_ai import cleanup
from stash_ai import merge
from stash_ai.models import WD
from tests.helpers.evidence import span
from tests.helpers.evidence import hit


from tests.helpers.cleanup_sql import SQLStash
from tests.helpers.cleanup import options
from tests.helpers.cleanup import install
pytestmark = pytest.mark.usefixtures("reset_progress")


@pytest.mark.parametrize('output', ['merged', 'separate'])
def test_global_cleanup_of_9000_scenes_never_fetches_scenes_or_predictions(tmp_path, monkeypatch, capsys, output):
    stash = SQLStash()
    blocked, human = stash.tag('ai:blocked'), stash.tag('blocked')
    for i in range(1, 9001):
        stash.scenes[str(i)] = dict(id=str(i), tags=[blocked, human], scene_markers=[])
    stash.scene = stash.scenes['1']
    stash.marker(merge.AI, 'blocked', '1', manual=True)
    keep = stash.marker(merge.AI, 'keep', '2')
    keep['tags'] += [stash.tag('op:blocked'), human]
    keep.update(seconds=10.125, end_seconds=63.625)
    stash.marker(WD, 'core_only', '3')
    stash.marker(WD, 'core_only', '4', manual=True)
    parent, child = stash.tag('jt:blocked'), stash.tag('jt:allowed_child')
    child['parents'] = [deepcopy(parent)]
    stash.images.append({'tags': [parent, human]})
    install(monkeypatch, stash)
    monkeypatch.setattr(merge.StateStore, 'records', lambda *a, **k: pytest.fail('global cleanup decoded results'))
    assert app.main(options(tmp_path, ['blocked'], ['core_only'], output=output)) == 0
    assert not stash.scene_reads
    assert all([t['name'] for t in s['tags']] == ['blocked'] for s in stash.scenes.values())
    assert {m['id'] for m in stash.scene['scene_markers']} == {'2', '4'}
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


def test_cursor_deletion_handles_overlapping_tags_once(tmp_path, monkeypatch):
    monkeypatch.setattr(cleanup, 'PAGE_SIZE', 3)
    stash = SQLStash()
    for i in range(1, 18):
        marker = stash.marker(merge.AI, 'ban'+str(i % 5), str(i))
        marker['tags'] += [stash.tag('op:ban'+str(j)) for j in range(5)]
    install(monkeypatch, stash)
    assert app.main([*options(tmp_path, ['ban'+str(i) for i in range(5)]),
                     '--cleanup-page-size', '3', '--cleanup-write-batch-size', '2']) == 0
    assert not stash.scene['scene_markers']
    removed = [identifier for op, identifier in stash.writes if op == 'delete']
    assert len(removed) == len(set(removed)) == 17
    assert all(len(v['ids']) <= 2 for op, v in stash.calls if op == 'CleanupDeleteMarkers')
    assert not stash.scene_reads


def test_global_dry_run_changes_neither_stash_nor_existing_state(tmp_path, monkeypatch):
    stash = SQLStash(dry=True)
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
    stash = SQLStash()
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
    stash = SQLStash(); stash.marker(merge.AI, 'blocked', '1')
    stash.fail_after = 'CleanupDeleteMarkers'
    install(monkeypatch, stash)
    assert app.main(options(tmp_path, ['blocked'])) == 1
    stash.writes.clear()
    assert app.main(options(tmp_path, ['different'])) == 1
    assert not stash.writes and 'ai:blocked' in stash.tags
    assert app.main(options(tmp_path, ['blocked'])) == 0


def test_pending_merge_blocks_global_deletion_before_first_mutation(tmp_path, monkeypatch):
    stash = SQLStash(); stash.marker(merge.AI, 'blocked', '1')
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
    stash = SQLStash()
    stash.tags, stash.scene = deepcopy(p.stash.tags), deepcopy(p.stash.scene)
    stash.scenes = {'1': stash.scene}
    monkeypatch.setattr(app, 'create_stash_client', lambda *_: stash)
    def blobs():
        with sqlite3.connect(p.path) as db:
            return (db.execute('SELECT data FROM results ORDER BY model,kind').fetchall(),
                    db.execute('SELECT published,pending FROM snapshots ORDER BY scene,kind').fetchall())
    before = blobs()
    args = options(tmp_path, ['common'])
    with monkeypatch.context() as patch:
        patch.setattr(merge.StateStore, 'unpack', lambda *_: pytest.fail('cleanup decompressed per-scene evidence'))
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
    stash = SQLStash()
    for i in range(2, 100):
        stash.scenes[str(i)] = dict(id=str(i), tags=[stash.tag('ai:blocked')], scene_markers=[])
    with merge.StateStore(tmp_path/'state.sqlite3', merge.scope_for(app.STASH_URL)) as store:
        merge_sync.sync_merged(app, stash, app.TagResolver(stash, source=merge.AI), store, stash.find_scene('1'),
            [hit('wd'), hit('oppai')], [span('joytag', 10, 20), span('oppai', 20, 30), span('wd', 30, 40)])
    install(monkeypatch, stash)
    stash.scene_reads.clear()
    assert app.main(options(tmp_path, ['blocked'], by_model={'oppai': ['common']})) == 0
    assert set(stash.scene_reads) == {'1'}
    assert sorted((m['seconds'], m['end_seconds']) for m in stash.scene['scene_markers']) == [(10, 20), (30, 40)]
    assert 'ai:common' in stash.tags
    assert all(not s['tags'] for key, s in stash.scenes.items() if key != '1')


def test_tag_renamed_after_catalog_is_not_deleted(tmp_path, monkeypatch):
    stash = SQLStash(); stash.tag('ai:blocked')
    def rename(client, key):
        if key == 'tag_check':
            tag = client.tags.pop('ai:blocked'); tag['name'] = 'ai:renamed'
            client.tags[tag['name']] = tag
            client.before_sql = None
    stash.before_sql = rename
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
    stash = SQLStash()
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
