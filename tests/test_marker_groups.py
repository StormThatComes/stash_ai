"""Marker group lifecycle, native API safety and backfill without inference."""
from copy import deepcopy
import json

import pytest

from stash_ai import cli as app
from stash_ai.publication import merged as merge_sync
from stash_ai import marker_groups as groups
from stash_ai import merge
from stash_ai.occurrences import OCCURRENCE_TAG
from stash_ai.prune import REQUIRED_COUNTS, prune_tags
from stash_ai.models import WD
from tests.helpers.cleanup import FastStash
from tests.helpers.cleanup import install
from tests.helpers.cleanup import options
from tests.helpers.cleanup_sql import SQLStash
from tests.helpers.evidence import span
from tests.helpers.stash import Stash
from tests.helpers.prune import PruneStash


pytestmark = pytest.mark.usefixtures("reset_progress")


def manager(stash):
    return groups.for_resolver(app, app.TagResolver(stash, source=merge.AI))


def attach(stash, marker, label, occurrence=False):
    identifier = manager(stash).resolve(label, occurrence)
    marker['tags'].append(stash.find_tag_by_id(identifier))
    if occurrence:
        marker['tags'].append(stash.tag(OCCURRENCE_TAG))
    return identifier


def group_names(marker):
    return {t['name'] for t in marker['tags'] if t['name'].startswith(groups.PREFIX)}


def backfill_args(tmp_path, *extra):
    return ['--sync-marker-groups-only', '--state-db', str(tmp_path/'state.sqlite3'), *extra]


def saved_marker_sources(tmp_path):
    with merge.StateStore(tmp_path/'state.sqlite3', merge.scope_for(app.STASH_URL)) as state:
        state.publish('1', {'markers': {'contributions': [span('wd', 0, 2, 'blocked')]}})


@pytest.mark.parametrize('use_end', [False, True])
def test_one_group_per_primary_independent_of_extra_labels_or_time_policy(tmp_path, monkeypatch, use_end):
    stash = Stash()
    monkeypatch.setattr(app, 'MARKER_USE_END_TIME', use_end)
    with merge.StateStore(tmp_path/'state', 'fixture') as state:
        resolver = app.TagResolver(stash, source=merge.AI)
        contributions = [span('wd', 0, 5, 'abc', extras=['def']), span('joytag', 10, 15, 'abc')]
        merge_sync.sync_merged(app, stash, resolver, state, stash.find_scene('1'), [], contributions)
        assert len(stash.scene['scene_markers']) == 2
        assert all(group_names(m) == {'m:abc'} for m in stash.scene['scene_markers'])
        assert 'm:def' not in stash.tags and 'm:abc:occurance' not in stash.tags
        assert all(not t['name'].startswith('m:') for t in stash.scene['tags'])
        root, group = stash.tags['ai_markers'], stash.tags['m:abc']
        assert [t['id'] for t in group['parents']] == [root['id']]
        assert root['ignore_auto_tag'] and group['ignore_auto_tag']
        assert groups.managed_group(group)
        assert 'm:abc' not in json.dumps(state.snapshot('1', 'markers')[0]['contributions'])
        stash.writes.clear()
        merge_sync.sync_merged(app, stash, resolver, state, stash.find_scene('1'), [], contributions)
        assert not stash.writes


def test_union_and_split_drop_obsolete_managed_groups_but_keep_manual_extras(tmp_path):
    stash = Stash()
    with merge.StateStore(tmp_path/'state', 'fixture') as state:
        resolver = app.TagResolver(stash, source=merge.AI)
        base = [span('wd', 0, 5, 'abc'), span('joytag', 10, 15, 'abc')]
        def sync(contributions):
            merge_sync.sync_merged(app, stash, resolver, state, stash.find_scene('1'), [], contributions)
        sync(base)
        stale = groups.for_resolver(app, resolver).resolve('old')
        for marker in stash.scene['scene_markers']:
            marker['tags'] += [stash.find_tag_by_id(stale), stash.tag('human'), stash.tag('m:manual')]
        sync([*base, span('oppai', 5, 10, 'abc')])
        assert len(stash.scene['scene_markers']) == 1
        sync(base)
        assert len(stash.scene['scene_markers']) == 2
        for marker in stash.scene['scene_markers']:
            assert group_names(marker) == {'m:abc', 'm:manual'}
            assert 'human' in {t['name'] for t in marker['tags']}
        assert groups.prune_groups(stash, lambda _: None)['marker_group_tags_pruned'] == 1
        assert 'm:old' not in stash.tags and 'm:manual' in stash.tags


@pytest.mark.parametrize('dry', [False, True])
def test_repair_adds_direct_root_preserving_other_parents_and_disables_autotag(dry):
    stash = Stash(); identifier = manager(stash).resolve('abc', True)
    manual = stash.tag('human category')
    tag = stash.tags['m:abc:occurance']
    tag.update(parents=[manual], ignore_auto_tag=False)
    stash.dry_run = dry; stash.writes.clear()
    resolver = app.TagResolver(stash, source=merge.AI)
    assert groups.for_resolver(app, resolver).resolve('abc', True) == identifier
    repaired = resolver.records['m:abc:occurance']
    assert {p['name'] for p in repaired['parents']} == {'ai_markers', 'human category'}
    assert repaired['ignore_auto_tag']
    assert bool(stash.writes) != dry
    assert tag['ignore_auto_tag'] != dry


@pytest.mark.parametrize('existing', ['manual', 'ambiguous_suffix'])
def test_name_collision_never_adopts_an_unrelated_group(existing):
    stash = Stash()
    if existing == 'manual':
        stash.tag('m:abc:occurance')
    else:
        manager(stash).resolve('abc:occurance')
    before = deepcopy(stash.tags['m:abc:occurance'])
    with pytest.raises(ValueError, match='name collision'):
        manager(stash).resolve('abc', True)
    assert stash.tags['m:abc:occurance'] == before


def test_changed_ownership_on_refresh_prevents_hierarchy_repair(monkeypatch):
    stash = Stash(); identifier = manager(stash).resolve('abc')
    stash.tags['m:abc']['parents'] = []
    resolver = app.TagResolver(stash, source=merge.AI); resolver.preload()
    stash.tags['m:abc']['description'] = 'Now maintained manually'
    stash.writes.clear()
    with pytest.raises(ValueError, match='ownership changed'):
        groups.for_resolver(app, resolver).resolve('abc')
    assert not stash.writes and stash.tags['m:abc']['id'] == identifier


def test_missing_tags_respect_create_missing_switch(monkeypatch):
    stash = Stash()
    monkeypatch.setattr(app, 'CREATE_MISSING', False)
    with pytest.raises(ValueError, match='CREATE_MISSING'):
        manager(stash).resolve('abc')
    assert not stash.writes


@pytest.mark.parametrize('end', [None, 2])
def test_backfill_uses_semantic_type_preserves_identity_and_ignores_manual_markers(tmp_path, monkeypatch, end):
    stash = FastStash()
    regular = stash.marker(merge.AI, 'abc', '1'); regular['end_seconds'] = end
    occurrence = stash.marker(merge.AI, 'abc', '2'); occurrence['tags'].append(stash.tag(OCCURRENCE_TAG))
    occurrence.update(title='[AI] custom title', seconds=12.125, end_seconds=None)
    manual = stash.marker(merge.AI, 'abc', '3', manual=True)
    legacy = stash.marker(WD, 'abc', '4')
    for marker in (regular, occurrence): marker['tags'].append(stash.tag('human'))
    before = deepcopy(stash.scene)
    install(monkeypatch, stash)
    monkeypatch.setattr(app, 'load_model_settings', lambda *a: pytest.fail('backfill loaded model configuration'))
    assert app.main(backfill_args(tmp_path, '--no-hierarchy')) == 0
    after = stash.scene['scene_markers']
    assert group_names(after[0]) == {'m:abc'}
    assert group_names(after[1]) == {'m:abc:occurance'}
    assert after[2:] == [manual, legacy] and stash.scene['tags'] == before['tags']
    for old, new in zip(before['scene_markers'], after):
        for field in ('id', 'title', 'seconds', 'end_seconds', 'primary_tag'):
            assert new[field] == old[field]
        assert {t['id'] for t in old['tags']} <= {t['id'] for t in new['tags']}
    stash.writes.clear()
    assert app.main(backfill_args(tmp_path)) == 0
    assert not stash.writes


@pytest.mark.parametrize('dry', [False, True])
def test_backfill_replaces_core_and_type_groups_and_prunes_only_unused(tmp_path, monkeypatch, capsys, dry):
    stash = FastStash()
    marker = stash.marker(merge.AI, 'abc', '1')
    attach(stash, marker, 'abc', True)
    # Simulate a core/type edit after publication; the old group is still attached.
    marker['primary_tag'] = stash.tag('ai:def')
    marker['tags'] = [t for t in marker['tags'] if t['name'] != OCCURRENCE_TAG]
    marker['tags'].append(stash.tag('m:human'))
    stash.dry_run = dry; stash.writes.clear()
    before = deepcopy((stash.tags, stash.scenes))
    install(monkeypatch, stash)
    assert app.main(backfill_args(tmp_path, *(['--dry-run'] if dry else []))) == 0
    assert 'marker_group_tags_pruned=1' in capsys.readouterr().out
    if dry:
        assert (stash.tags, stash.scenes) == before and not stash.writes
        assert not (tmp_path/'state.sqlite3').exists()
    else:
        assert group_names(stash.scene['scene_markers'][0]) == {'m:def', 'm:human'}
        assert 'm:abc:occurance' not in stash.tags and groups.ROOT in stash.tags


def test_backfill_failure_keeps_orphans_for_retry_and_does_not_mutate_raw_evidence(tmp_path, monkeypatch):
    stash = FastStash()
    for identifier in ('1', '2'): stash.marker(merge.AI, 'abc', identifier)
    manager(stash).resolve('orphan')
    install(monkeypatch, stash)
    path = tmp_path/'state.sqlite3'
    with merge.StateStore(path, merge.scope_for(app.STASH_URL)) as state:
        state.put('1', 'wd', {'markers': {'mode': 'frames', 'meta': {}, 'items': []}})
        raw = state.records('1', 'markers')
    update = stash.update_marker
    def fail_second(payload):
        if payload['id'] == '2': raise RuntimeError('interrupted')
        update(payload)
    monkeypatch.setattr(stash, 'update_marker', fail_second)
    assert app.main(backfill_args(tmp_path)) == 1
    assert 'm:orphan' in stash.tags and group_names(stash.scene['scene_markers'][0]) == {'m:abc'}
    monkeypatch.setattr(stash, 'update_marker', update)
    assert app.main(backfill_args(tmp_path)) == 0
    assert 'm:orphan' not in stash.tags
    with merge.StateStore(path, merge.scope_for(app.STASH_URL), True) as state:
        assert state.records('1', 'markers') == raw


@pytest.mark.parametrize('extra', [['--output-mode', 'separate'], ['--scene-tags-only'], ['--restart'], ['--prune-empty-tags']])
def test_backfill_rejects_conflicting_modes(tmp_path, extra):
    with pytest.raises(SystemExit) as exc:
        app.main(backfill_args(tmp_path, *extra))
    assert exc.value.code == 2


@pytest.mark.parametrize('counter', sorted(REQUIRED_COUNTS - {'child_count'} | {'studio_count', 'group_count', 'movie_count'}))
def test_group_pruning_keeps_global_usage_manual_names_and_root(counter):
    stash = PruneStash()
    root = stash.add('ai_markers')
    used = stash.add('m:abc', [root], description=groups.group_description('abc'), **{counter: 1})
    empty = stash.add('m:def:occurance', [root], description=groups.group_description('def', True))
    manual = stash.add('m:manual', [root])
    stats = prune_tags(stash, report=lambda _: None)
    assert stats['tags_pruned'] == 1 and stash.deleted == [empty]
    assert set(stash.tags) == {root, used, manual}


def test_group_pruning_refresh_rechecks_ownership():
    stash = PruneStash()
    identifier = stash.add('m:abc', description=groups.group_description('abc'))
    def change(stash, identifier): stash.tags[identifier]['description'] = 'manual'
    stash.before_refresh = change
    assert prune_tags(stash, report=lambda _: None)['tags_retained_after_refresh'] == 1
    assert identifier in stash.tags


@pytest.mark.parametrize('strategy', ['sql', 'scenes'])
@pytest.mark.parametrize('dry', [False, True])
@pytest.mark.parametrize('explicit_prune', [False, True])
def test_cleanup_removes_both_groups_with_matching_dry_run_counts(tmp_path, monkeypatch, capsys, strategy, dry, explicit_prune):
    stash = SQLStash() if strategy == 'sql' else FastStash()
    saved_marker_sources(tmp_path)
    for identifier, occurrence in [('1', False), ('2', True)]:
        attach(stash, stash.marker(merge.AI, 'blocked', identifier), 'blocked', occurrence)
    stash.dry_run = dry; stash.writes.clear()
    before = deepcopy((stash.tags, stash.scenes))
    install(monkeypatch, stash)
    args = [*options(tmp_path, cores=['blocked']), '--cleanup-strategy', strategy]
    if dry: args.append('--dry-run')
    if explicit_prune: args.append('--prune-empty-tags')
    assert app.main(args) == 0
    assert 'marker_group_tags_pruned=2' in capsys.readouterr().out
    if dry:
        assert (stash.tags, stash.scenes) == before and not stash.writes
        if strategy == 'sql': assert any(key == 'group_usage' for key, _ in stash.sql_calls)
    else:
        assert not stash.scene['scene_markers']
        assert 'm:blocked' not in stash.tags and 'm:blocked:occurance' not in stash.tags
        assert 'ai_markers' in stash.tags


@pytest.mark.parametrize('strategy', ['sql', 'scenes'])
def test_scoped_cleanup_keeps_group_used_on_another_scene(tmp_path, monkeypatch, strategy):
    stash = SQLStash() if strategy == 'sql' else FastStash()
    saved_marker_sources(tmp_path)
    marker = stash.marker(merge.AI, 'blocked', '1'); attach(stash, marker, 'blocked')
    second = deepcopy(stash.scene); second['id'] = '2'; second['scene_markers'][0]['id'] = '2'
    stash.scenes['2'] = second
    install(monkeypatch, stash)
    assert app.main([*options(tmp_path, cores=['blocked']), '--cleanup-strategy', strategy, '--scene-id', '1']) == 0
    assert not stash.scene['scene_markers'] and stash.scenes['2'] == second
    assert 'm:blocked' in stash.tags


def test_scene_tag_only_cleanup_does_not_touch_groups(tmp_path, monkeypatch):
    stash = FastStash()
    attach(stash, stash.marker(merge.AI, 'blocked', '1'), 'blocked')
    before = deepcopy(stash.scene['scene_markers'])
    install(monkeypatch, stash)
    args = options(tmp_path, cores=['blocked'])
    args[args.index('--cleanup-only')] = '--cleanup-scene-tags-only'
    assert app.main(args) == 0
    assert stash.scene['scene_markers'] == before and 'm:blocked' in stash.tags


@pytest.mark.parametrize('change', ['hierarchy', 'deleted'])
def test_merge_cache_repairs_groups_with_no_category_hierarchy(pipeline, monkeypatch, change):
    p = pipeline
    assert app.main(p.args) == 0
    tag = p.stash.tags['m:common']
    if change == 'hierarchy': tag['parents'] = []
    else:
        del p.stash.tags['m:common']
        for marker in p.stash.scene['scene_markers']:
            marker['tags'] = [t for t in marker['tags'] if t['id'] != tag['id']]
    p.stash.writes.clear()
    monkeypatch.setattr(app, 'load_tagger', lambda *a: pytest.fail('group repair loaded model weights'))
    assert app.main([*p.args, '--merge-only']) == 0
    assert p.stash.tags['m:common']['parents'][0]['name'] == 'ai_markers'
    assert all('m:common' in group_names(m) for m in p.stash.scene['scene_markers'] if m['primary_tag']['name'] == 'ai:common')
    p.stash.writes.clear()
    assert app.main([*p.args, '--merge-only']) == 0
    assert not p.stash.writes
