"""SQL group prefilter scalability, preview effects and final API safeguards."""
from copy import deepcopy
import json

import pytest

from stash_ai import cli as app
from stash_ai.cleanup import sql as fast
from stash_ai.cleanup.sql_session import READ_OPERATION, SQL
from stash_ai.publication import groups as groups
import stash_ai.evidence.labels as merge_labels
import stash_ai.storage.store as merge_store
from tests.helpers.cleanup import install, options
from tests.helpers.cleanup_sql import SQLStash
from tests.helpers.evidence import span
from tests.helpers.stash import Stash


pytestmark = pytest.mark.usefixtures('reset_progress')


class CountedStash(SQLStash):
    def __init__(self):
        super().__init__()
        self.usage_reads = []

    def gql(self, query, variables=None):
        if 'StashAIPruneCandidates' in query:
            self.usage_reads.append(list(variables['ids']))
            # The shared base helper computes counters only for requested IDs.
            return Stash.gql(self, query, variables)
        return super().gql(query, variables)


def managed(stash, label, occurrence=False):
    root = stash.tag(groups.ROOT)
    tag = stash.tag(groups.group_name(label, occurrence))
    tag.update(description=groups.group_description(label, occurrence), parents=[{'id': root['id']}])
    return tag


def command(tmp_path, **kwargs):
    return [*options(tmp_path, **kwargs), '--cleanup-strategy', 'sql']


def test_17000_groups_fetch_counters_only_for_two_old_orphans(tmp_path, monkeypatch, capsys):
    stash = CountedStash()
    root = stash.tag(groups.ROOT)
    primary, owner = stash.tag('ai:kept'), stash.tag(merge_labels.AI.owner_tag)
    group_ids = set()
    # Populate directly: fixture ID allocation scans all existing tags.
    for i in range(16998):
        label, identifier = f'kept_{i}', str(i + 4)
        tag = dict(id=identifier, name=groups.group_name(label), parents=[{'id': root['id']}],
                   description=groups.group_description(label))
        stash.tags[tag['name']] = tag
        group_ids.add(identifier)
        stash.scene['scene_markers'].append(dict(id=str(i + 1), title=merge_labels.AI.title_prefix + 'kept',
            seconds=0, end_seconds=2, primary_tag=primary, tags=[owner, tag]))
    orphans = {managed(stash, 'old', occurrence)['id'] for occurrence in (False, True)}
    group_ids.update(orphans)
    install(monkeypatch, stash)
    assert app.main([*command(tmp_path), '--prune-empty-tags']) == 0
    assert stash.usage_reads == [sorted(orphans)]
    scans = [json.loads(params[0]) for key, params in stash.sql_calls if key == 'unused_candidates']
    group_scans = [batch for batch in scans if set(map(str, batch)) & group_ids]
    assert len(group_scans) == 9 and max(map(len, group_scans)) == 2000
    assert {str(i) for batch in group_scans for i in batch} == group_ids
    assert 'm:old' not in stash.tags and 'm:old:occurance' not in stash.tags
    assert len(stash.scene['scene_markers']) == 16998
    output = capsys.readouterr().out
    assert 'marker_group_tags_checked_sql=17000' in output
    assert 'marker_group_tags_candidates=2' in output


@pytest.mark.parametrize('dry', [False, True])
@pytest.mark.parametrize('shared', [False, True])
def test_scoped_cleanup_preview_includes_groups_losing_members(tmp_path, monkeypatch, capsys, dry, shared):
    stash = CountedStash()
    tag = managed(stash, 'blocked')
    marker = stash.marker(merge_labels.AI, 'blocked', '1')
    marker['tags'].append(tag)
    if shared:
        second = deepcopy(stash.scene)
        second['id'] = '2'
        second['scene_markers'][0]['id'] = '2'
        stash.scenes['2'] = second
    with merge_store.StateStore(tmp_path/'state.sqlite3', merge_store.scope_for(app.STASH_URL)) as state:
        state.publish('1', {'markers': {'contributions': [span('wd', 0, 2, 'blocked')]}})
    stash.dry_run = dry
    before = deepcopy((stash.tags, stash.scenes))
    install(monkeypatch, stash)
    args = [*command(tmp_path, cores=['blocked']), '--scene-id', '1']
    assert app.main(args + (['--dry-run'] if dry else [])) == 0
    output = capsys.readouterr().out
    if not shared:
        assert 'marker_group_tags_pruned=1' in output
    if dry:
        assert stash.usage_reads == [[tag['id']]]
        assert (stash.tags, stash.scenes) == before and not stash.writes
    else:
        assert ('m:blocked' in stash.tags) == shared
        assert not stash.scene['scene_markers']
        assert stash.usage_reads == ([] if shared else [[tag['id']]])


def test_prefilter_keeps_native_checks_for_images_children_and_ownership(tmp_path, monkeypatch):
    stash = CountedStash()
    empty, image_tag, parent = (managed(stash, label) for label in ('empty', 'image', 'parent'))
    stash.images = [{'tags': [image_tag]}]
    stash.tag('manual child')['parents'] = [{'id': parent['id']}]
    manual = stash.tag('m:manual')
    primary, scene_tag = managed(stash, 'primary'), managed(stash, 'scene')
    stash.scene['scene_markers'] = [dict(id='1', title='Manual', seconds=0, primary_tag=primary, tags=[])]
    stash.scene['tags'] = [scene_tag]
    install(monkeypatch, stash)
    assert app.main(command(tmp_path)) == 0
    assert set(stash.usage_reads[0]) == {t['id'] for t in (empty, image_tag, parent, manual)}
    assert 'm:empty' not in stash.tags
    assert all(t['name'] in stash.tags for t in (image_tag, parent, manual, primary, scene_tag))
    assert groups.ROOT in stash.tags


def test_new_usage_before_final_refresh_prevents_deletion(tmp_path, monkeypatch, capsys):
    stash = CountedStash()
    tag = managed(stash, 'changed')
    original = stash.gql

    def gql(query, variables=None):
        if 'query StashAIPruneTag(' in query:
            stash.images.append({'tags': [tag]})
        return original(query, variables)

    stash.gql = gql
    install(monkeypatch, stash)
    assert app.main(command(tmp_path)) == 0
    assert 'm:changed' in stash.tags and not stash.writes
    assert 'marker_group_tags_retained=1' in capsys.readouterr().out


def test_all_used_groups_skip_full_counter_reads(tmp_path, monkeypatch, capsys):
    stash = CountedStash()
    tag = managed(stash, 'used')
    stash.marker(merge_labels.AI, 'allowed', '1')['tags'].append(tag)
    install(monkeypatch, stash)
    assert app.main(command(tmp_path)) == 0
    assert not stash.usage_reads and not stash.writes
    output = capsys.readouterr()
    assert 'Find empty marker groups' in output.err
    assert 'Check tag usage' not in output.err
    assert 'marker_group_tags_candidates=0' in output.out


def test_unexpected_sql_candidate_stops_before_group_deletion(tmp_path, monkeypatch):
    stash = CountedStash()
    managed(stash, 'empty')
    outside = stash.tag('manual')
    original = stash.gql

    def gql(query, variables=None):
        if query == READ_OPERATION and variables['sql'] == SQL['unused_candidates']:
            return {'querySQL': {'columns': ['id'], 'rows': [[int(outside['id'])]]}}
        return original(query, variables)

    stash.gql = gql
    install(monkeypatch, stash)
    assert app.main(command(tmp_path)) == 1
    assert not stash.usage_reads and not stash.writes
