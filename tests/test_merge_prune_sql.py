"""Post-merge pruning limits native usage reads without weakening deletion checks."""
from copy import deepcopy
import json

import pytest

from stash_ai import cli as app
from stash_ai import cleanup_sql as fast
from stash_ai import marker_groups as groups
from stash_ai import merge
from tests.helpers.cleanup import install, options
from tests.helpers.cleanup_sql import SQLStash
from tests.helpers.evidence import span
from tests.helpers.stash import Stash


pytestmark = pytest.mark.usefixtures('reset_progress')


class PruneStash(SQLStash):
    def __init__(self, dry=False):
        super().__init__(dry)
        self.usage_reads = []
        self.catalog_reads = 0

    def gql(self, query, variables=None):
        if 'StashAIPruneCatalog' in query:
            pytest.fail('SQL post-merge pruning fetched the full usage catalog')
        if 'StashAIPruneCandidates' in query:
            self.usage_reads.append(list(variables['ids']))
            return Stash.gql(self, query, variables)
        if 'StashAICleanupTags' in query:
            self.catalog_reads += 1
        return super().gql(query, variables)


def managed(stash, label):
    root = stash.tag(groups.ROOT)
    tag = stash.tag(groups.group_name(label))
    tag.update(description=groups.group_description(label), parents=[root])
    return tag


def command(tmp_path, **kwargs):
    return [arg for arg in options(tmp_path, **kwargs) if arg != '--cleanup-only'] + ['--merge-only']


@pytest.mark.parametrize('explicit', [False, True])
def test_merge_prefilters_used_groups_and_shares_one_lightweight_catalog(tmp_path, monkeypatch, capsys, explicit):
    stash = PruneStash()
    root, primary = stash.tag(groups.ROOT), stash.tag('manual core')
    # Cross SQL/catalog page boundaries without quadratic fixture ID allocation.
    for i in range(2001):
        label, identifier = f'used_{i}', str(i + 3)
        tag = dict(id=identifier, name=groups.group_name(label), parents=[root],
                   description=groups.group_description(label))
        stash.tags[tag['name']] = tag
        stash.scene['scene_markers'].append(dict(id=str(i + 1), title='Manual', seconds=i,
            end_seconds=i + 1, primary_tag=primary, tags=[tag]))
    orphan = managed(stash, 'orphan')
    ordinary = stash.tag('op:empty')
    install(monkeypatch, stash)
    args = command(tmp_path, by_model={'oppai': ['empty']})
    assert app.main(args + (['--prune-empty-tags'] if explicit else [])) == 0
    assert stash.usage_reads == [[ordinary['id']], [orphan['id']]]
    assert stash.catalog_reads == 3  # one ID/name catalog, paginated at 1000
    assert len(stash.scene['scene_markers']) == 2001
    assert orphan['name'] not in stash.tags and ordinary['name'] not in stash.tags
    scans = [json.loads(params[0]) for key, params in stash.sql_calls if key == 'unused_candidates']
    assert max(map(len, scans)) == 2000
    output = capsys.readouterr().out
    assert 'merge_prune_sql_enabled=1' in output
    assert 'marker_group_tags_checked_sql=2002' in output
    assert 'marker_group_tags_candidates=1' in output


@pytest.mark.parametrize('dry', [False, True])
@pytest.mark.parametrize('shared', [False, True])
def test_merge_prunes_groups_losing_markers_and_preserves_global_usage(tmp_path, monkeypatch, capsys, dry, shared):
    stash = PruneStash(dry)
    tag = managed(stash, 'old')
    stash.marker(merge.AI, 'old', '1')['tags'].append(tag)
    if shared:
        other = deepcopy(stash.scene)
        other['id'], other['scene_markers'][0]['id'] = '2', '2'
        stash.scenes['2'] = other
    with merge.StateStore(tmp_path/'state.sqlite3', merge.scope_for(app.STASH_URL)) as state:
        state.put('1', 'wd', {'markers': dict(mode='spans', items=[], meta={'origin': 'legacy'})})
        state.publish('1', {'markers': {'contributions': [span('wd', 0, 2, 'old')]}})
    before = deepcopy((stash.tags, stash.scenes))
    install(monkeypatch, stash)
    args = [*command(tmp_path), '--scene-id', '1']
    assert app.main(args + (['--dry-run'] if dry else [])) == 0
    output = capsys.readouterr().out
    assert ('marker_group_tags_pruned=1' in output) == (not shared)
    if dry:
        assert stash.usage_reads == [[tag['id']]]
        assert (stash.tags, stash.scenes) == before and not stash.writes
    else:
        assert not stash.scene['scene_markers']
        assert (tag['name'] in stash.tags) == shared
        assert stash.usage_reads == ([] if shared else [[tag['id']]])


def test_merge_prune_keeps_native_usage_children_and_group_ownership_checks(tmp_path, monkeypatch):
    stash = PruneStash()
    empty, image, parent = (managed(stash, label) for label in ('empty', 'image', 'parent'))
    stash.images.append({'tags': [image]})
    stash.tag('manual child')['parents'] = [parent]
    manual = stash.tag('m:manual')
    primary, extra, scene = (managed(stash, label) for label in ('primary', 'extra', 'scene'))
    stash.scene['scene_markers'] = [dict(id='1', title='Manual', seconds=0, end_seconds=2,
        primary_tag=primary, tags=[extra])]
    stash.scene['tags'] = [scene]
    install(monkeypatch, stash)
    assert app.main(command(tmp_path)) == 0
    assert set(stash.usage_reads[0]) == {t['id'] for t in (empty, image, parent, manual)}
    assert empty['name'] not in stash.tags
    assert all(t['name'] in stash.tags for t in (image, parent, manual, primary, extra, scene))


def test_merge_prune_rechecks_new_usage_before_deleting(tmp_path, monkeypatch, capsys):
    stash = PruneStash()
    tag = managed(stash, 'changed')
    original = stash.gql

    def gql(query, variables=None):
        if 'query StashAIPruneTag(' in query:
            stash.images.append({'tags': [tag]})
        return original(query, variables)

    stash.gql = gql
    install(monkeypatch, stash)
    assert app.main(command(tmp_path)) == 0
    assert tag['name'] in stash.tags and not stash.writes
    assert 'marker_group_tags_retained=1' in capsys.readouterr().out


@pytest.mark.parametrize('failure', ['probe', 'catalog'])
def test_merge_prune_falls_back_before_discovery_writes(tmp_path, monkeypatch, capsys, failure):
    stash = SQLStash()
    tag = managed(stash, 'orphan')
    original = stash.gql

    def gql(query, variables=None):
        if failure == 'probe' and query == fast.READ_OPERATION:
            raise RuntimeError('SQL disabled')
        if failure == 'catalog' and 'StashAICleanupTags' in query:
            raise RuntimeError('Catalog unavailable')
        return original(query, variables)

    stash.gql = gql
    install(monkeypatch, stash)
    assert app.main(command(tmp_path)) == 0
    assert tag['name'] not in stash.tags
    assert 'merge_prune_graphql_fallback=1' in capsys.readouterr().out


def test_merge_prune_rejects_unexpected_sql_candidates_before_native_reads(tmp_path, monkeypatch):
    stash = PruneStash()
    managed(stash, 'orphan')
    outside = stash.tag('manual')
    original = stash.gql

    def gql(query, variables=None):
        if query == fast.READ_OPERATION and variables['sql'] == fast.SQL['unused_candidates']:
            return {'querySQL': {'columns': ['id'], 'rows': [[int(outside['id'])]]}}
        return original(query, variables)

    stash.gql = gql
    install(monkeypatch, stash)
    with pytest.raises(ValueError, match='Unexpected prune candidate'):
        app.main(command(tmp_path))
    assert not stash.usage_reads and not stash.writes


def test_explicit_prefix_pruning_removes_newly_empty_parents(tmp_path, monkeypatch):
    stash = PruneStash()
    parent = stash.tag('op:parent')
    leaf = stash.tag('op:leaf')
    leaf['parents'] = [{'id': parent['id']}]
    outside = stash.tag('wd:empty')
    install(monkeypatch, stash)
    assert app.main([*command(tmp_path), '--prune-empty-tags', '--prune-prefixes', 'op:',
                     '--scene-tags-only']) == 0
    assert ('delete_tag', leaf['name']) in stash.writes
    assert ('delete_tag', parent['name']) in stash.writes
    assert outside['name'] in stash.tags


def test_failed_merge_does_not_discover_or_prune_tags(tmp_path, monkeypatch):
    stash = PruneStash()
    managed(stash, 'orphan')
    stash.marker(merge.AI, 'missing state', '1')
    install(monkeypatch, stash)
    assert app.main(command(tmp_path)) == 1
    assert not stash.sql_calls and not stash.catalog_reads and not stash.usage_reads
    assert not stash.writes


def test_pruned_catalog_is_finalized_for_next_merge(tmp_path, monkeypatch, capsys):
    stash = PruneStash()
    orphan = managed(stash, 'orphan')
    install(monkeypatch, stash)
    args = command(tmp_path)

    assert app.main(args) == 0
    assert orphan['name'] not in stash.tags
    capsys.readouterr()

    monkeypatch.setattr(
        merge, 'evidence',
        lambda *a, **k: pytest.fail('catalog pruning forced an unnecessary second merge'),
    )
    stash.writes.clear()
    assert app.main(args) == 0
    output = capsys.readouterr().out
    assert 'merge_reused=1' in output
    assert not stash.writes


def test_unchanged_merge_skips_native_counters_for_used_groups(tmp_path, monkeypatch, capsys):
    stash = PruneStash()
    tag = managed(stash, 'used')
    stash.marker(merge.AI, 'used', '1', manual=True)['tags'].append(tag)
    install(monkeypatch, stash)
    args = command(tmp_path)
    assert app.main(args) == 0
    capsys.readouterr()
    assert app.main(args) == 0
    output = capsys.readouterr().out
    assert 'merge_reused=1' in output
    assert 'marker_group_tags_candidates=0' in output
    assert not stash.usage_reads and not stash.writes
