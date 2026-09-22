"""Abandonment across partial API success, old state formats and new policies."""
from copy import deepcopy
import sqlite3

import pytest

import stash_ai as app
import stash_cleanup as cleanup
import stash_merge as merge
import stash_recovery as recovery
from stash_occurrences import OCCURRENCE_TAG
from tests.helpers.cleanup import FastStash
from tests.helpers.cleanup import install
from tests.helpers.cleanup import options
from tests.helpers.cleanup_sql import SQLStash
from tests.helpers.evidence import span
from tests.helpers.evidence import hit
from tests.helpers.evidence import empty_policies


pytestmark = pytest.mark.usefixtures("reset_progress")


def state_path(tmp_path):
    return tmp_path/'state.sqlite3'


def state(tmp_path, **kwargs):
    return merge.StateStore(state_path(tmp_path), merge.scope_for(app.STASH_URL), **kwargs)


def seed(tmp_path, stash, spans=None):
    spans = spans if spans is not None else [span('wd', 0, 2, 'abc'), span('wd', 10, 12, 'abc'), span('wd', 20, 22, 'def')]
    hits = [hit('wd', label) for label in sorted({s['tag'] for s in spans})]
    with state(tmp_path) as store:
        store.put('1', 'wd', {
            'scene': dict(mode='scene', items=hits, meta={'origin': 'legacy'}),
            'markers': dict(mode='spans', items=spans, meta={'origin': 'legacy'})})
        merge.sync_merged(app, stash, app.TagResolver(stash, source=merge.AI), store, stash.find_scene('1'), hits, spans)


def raw_results(tmp_path):
    with sqlite3.connect(state_path(tmp_path)) as db:
        return db.execute('SELECT * FROM results ORDER BY scope,scene,model,kind').fetchall()


def command(tmp_path, *extra):
    return ['--abandon-pending', '--state-db', str(state_path(tmp_path)), *extra]


def test_standalone_abandons_without_stash_models_or_source_changes(tmp_path, monkeypatch):
    stash = FastStash(); seed(tmp_path, stash)
    with state(tmp_path) as store:
        store.prepare('1', {'scene': {'owned_ids': ['999'], 'contributions': [hit('wd', 'new')]}})
        cleanup.begin(store, 'old-policy', {'abc'}, set(), [stash.tags['ai:abc']['id']])
    before = raw_results(tmp_path)
    for name in ('create_stash_client', 'load_model_settings', 'load_vocabulary', 'load_tagger'):
        monkeypatch.setattr(app, name, lambda *a: pytest.fail('standalone accessed Stash/model assets'))
    assert app.main(command(tmp_path)) == 0
    with state(tmp_path) as store:
        assert not store.cleanup_pending and not store.has_pending_publications()
        assert store.connection.execute('PRAGMA user_version').fetchone() == (7,)
        assert recovery.publications(store, '1', 'scene')[0]['owned_ids'] == ['999']
        assert store.snapshot('1', 'markers')[0]['recovery_required']
        assert any(i['tag'] == 'abc' for i in store.snapshot('1', 'markers')[0]['contributions'])
    assert raw_results(tmp_path) == before
    assert app.main(command(tmp_path)) == 0
    with state(tmp_path) as store:
        assert store.cleanup_generation == 1
        assert len(recovery.publications(store, '1', 'scene')) == 1


@pytest.mark.parametrize('dry', [False, True])
@pytest.mark.parametrize('strategy', ['sql', 'scenes'])
def test_changed_cleanup_does_not_finish_old_ban_or_restore_deleted_markers(tmp_path, monkeypatch, dry, strategy):
    stash = SQLStash(); seed(tmp_path, stash)
    before_raw = raw_results(tmp_path)
    install(monkeypatch, stash)
    stash.fail_after = 'CleanupDeleteMarkers'
    old = [*options(tmp_path, ['abc']), '--cleanup-strategy', 'sql', '--cleanup-write-batch-size', '1']
    assert app.main(old) == 1
    assert [m['seconds'] for m in stash.scene['scene_markers'] if m['primary_tag']['name'] == 'ai:abc'] == [10]
    stash.dry_run = dry; stash.writes.clear()
    before_stash = deepcopy((stash.tags, stash.scenes))
    before_db = state_path(tmp_path).read_bytes()
    new = [*options(tmp_path, ['def']), '--cleanup-strategy', strategy, '--abandon-pending']
    if dry: new.append('--dry-run')
    assert app.main(new) == 0
    if dry:
        assert (stash.tags, stash.scenes) == before_stash and not stash.writes
        assert state_path(tmp_path).read_bytes() == before_db
    else:
        assert [(m['primary_tag']['name'], m['seconds']) for m in stash.scene['scene_markers']] == [('ai:abc', 10)]
        assert 'ai:abc' in stash.tags and 'ai:def' not in stash.tags
        # Later scoped cleanup observes the surviving old outputs, even when
        # the replacement fast cleanup did not have to read any scenes.
        assert app.main([*options(tmp_path), '--cleanup-strategy', 'scenes']) == 0
        with state(tmp_path) as store:
            published, pending = store.snapshot('1', 'markers')
            assert pending is None and not published.get('recovery_required')
            assert [(s['tag'], s['start']) for s in published['contributions']] == [('abc', 10)]
        assert [m['seconds'] for m in stash.scene['scene_markers']] == [10]
    assert raw_results(tmp_path) == before_raw


@pytest.mark.parametrize('strategy', ['sql', 'scenes'])
def test_partial_extra_removal_is_observed_without_restoring_or_spreading_it(tmp_path, monkeypatch, strategy):
    stash = SQLStash()
    seed(tmp_path, stash, [span('wd', 0, 2, 'core', extras=['abc', 'def']), span('wd', 10, 12, 'core', extras=['abc', 'def'])])
    human = stash.tag('human')
    for marker in stash.scene['scene_markers']: marker['tags'].append(human)
    install(monkeypatch, stash)
    stash.fail_after = 'CleanupRemoveTags'
    old = options(tmp_path, ['abc']); old[old.index('--cleanup-only')] = '--cleanup-markers-only'
    assert app.main([*old, '--cleanup-strategy', 'sql', '--cleanup-write-batch-size', '1']) == 1
    assert ['ai:abc' in {t['name'] for t in m['tags']} for m in stash.scene['scene_markers']] == [False, True]
    new = options(tmp_path, ['def']); new[new.index('--cleanup-only')] = '--cleanup-markers-only'
    assert app.main([*new, '--cleanup-strategy', strategy, '--abandon-pending']) == 0
    assert app.main([*options(tmp_path), '--cleanup-strategy', 'scenes']) == 0
    for marker, had_abc in zip(stash.scene['scene_markers'], [False, True]):
        names = {t['name'] for t in marker['tags']}
        assert ('ai:abc' in names) == had_abc
        assert 'ai:def' not in names and 'human' in names
    assert [m['seconds'] for m in stash.scene['scene_markers']] == [0, 10]


@pytest.mark.parametrize('dry', [False, True])
def test_abandoned_merge_keeps_scene_association_ownership_after_lost_response(tmp_path, monkeypatch, dry):
    stash = FastStash(); stash.scene['tags'] = [stash.tag('human')]
    actual = stash.update_scene_tags
    def lost(*args):
        actual(*args)
        raise RuntimeError('lost response after commit')
    with state(tmp_path) as store:
        monkeypatch.setattr(stash, 'update_scene_tags', lost)
        with pytest.raises(RuntimeError):
            merge.sync_merged(app, stash, app.TagResolver(stash, source=merge.AI), store,
                              stash.find_scene('1'), [hit('wd', 'abc')], [])
    monkeypatch.setattr(stash, 'update_scene_tags', actual)
    before = state_path(tmp_path).read_bytes()
    stash.dry_run = dry; stash.writes.clear()
    with state(tmp_path, dry_run=dry) as store:
        assert recovery.abandon(store, lambda _: None)['publications_abandoned'] == 2
        assert not store.has_pending_publications()
        # New desired policy has no AI output. 'abc' must not become a manual tag.
        merge.sync_merged(app, stash, app.TagResolver(stash, source=merge.AI), store, stash.find_scene('1'), [], [])
        assert not recovery.publications(store, '1', 'scene')
    if dry:
        assert state_path(tmp_path).read_bytes() == before and not stash.writes
    else:
        assert [t['name'] for t in stash.scene['tags']] == ['human']


def test_repeated_abandonment_retains_all_applied_scene_ownership(tmp_path):
    stash = FastStash(); human = stash.tag('human'); stash.scene['tags'] = [human]
    with state(tmp_path) as store:
        for name in ('abc', 'def'):
            tag = stash.tag('ai:' + name); stash.scene['tags'].append(tag)
            store.prepare('1', {'scene': {'owned_ids': [tag['id']], 'contributions': [hit('wd', name)]}})
            recovery.abandon(store, lambda _: None)
        assert len(recovery.publications(store, '1', 'scene')) == 2
        merge.sync_merged(app, stash, app.TagResolver(stash, source=merge.AI), store, stash.find_scene('1'), [], [])
        assert stash.scene['tags'] == [human]
        assert not recovery.publications(store, '1', 'scene')


@pytest.mark.parametrize('has_end', [False, True])
def test_abandoned_partial_merge_cleanup_preserves_observed_range_or_occurrence(tmp_path, monkeypatch, has_end):
    stash = FastStash(); stash.has_end_time = has_end
    seed(tmp_path, stash, [span('wd', 0, 2, 'core')])
    old = deepcopy(stash.scene['scene_markers'][0])
    point = {**span('wd', 8, None, 'core'), 'derived': 'occurrence'}
    with state(tmp_path) as store:
        store.prepare('1', {'markers': {'contributions': [point]}})
        # A new occurrence was created, but the old range was not yet removed.
        created = stash.marker(merge.AI, 'core', '2000')
        created.update(seconds=8, end_seconds=None)
        created['tags'].append(stash.tag(OCCURRENCE_TAG))
        recovery.abandon(store, lambda _: None)
        observed, changed = recovery.reconcile(app, stash, store, stash.find_scene('1'), 'markers', store.snapshot('1', 'markers')[0])
        assert changed
        assert {(s['start'], s['end'], s.get('derived')) for s in observed['contributions']} == {(0, 2, None), (8, None, 'occurrence')}
        assert not recovery.publications(store, '1', 'markers')
    assert stash.scene['scene_markers'][0] == old


def test_unrecognized_survivor_keeps_recovery_evidence_and_requests_current_merge(tmp_path):
    stash = FastStash(); seed(tmp_path, stash)
    with state(tmp_path) as store:
        store.prepare('1', {'markers': {'contributions': [span('wd', 0, 2, 'abc')]}})
        recovery.abandon(store, lambda _: None)
        stash.scene['scene_markers'][0]['seconds'] = 1.234
        with pytest.raises(RuntimeError, match='current YAML'):
            recovery.reconcile(app, stash, store, stash.find_scene('1'), 'markers', store.snapshot('1', 'markers')[0])
        assert recovery.publications(store, '1', 'markers')


@pytest.mark.parametrize('version', [5, 6])
def test_old_pending_cleanup_state_can_be_abandoned_without_losing_earlier_masks(tmp_path, version):
    with state(tmp_path) as store:
        store.publish('1', {'scene': {'owned_ids': ['1', '2'], 'contributions': [hit('wd', 'old'), hit('wd', 'new')]}})
        cleanup.begin(store, 'finished', {'old'}, set(), ['1']); cleanup.finish(store)
        cleanup.begin(store, 'interrupted', {'new'}, set(), ['2'])
        store.connection.execute(f'PRAGMA user_version={version}')
    # These are precisely the recovery data available in pre-abandon releases.
    with sqlite3.connect(state_path(tmp_path)) as db:
        db.execute('DROP TABLE recovery_publications'); db.execute('DROP TABLE recovery_cleanups')
    with state(tmp_path, allow_cleanup=True) as store:
        recovery.abandon(store, lambda _: None)
        snapshot = store.snapshot('1', 'scene')[0]
        assert [i['tag'] for i in snapshot['contributions']] == ['new']
        assert snapshot['owned_ids'] == ['2'] and snapshot['recovery_required']
        cleanup.begin(store, 'next', {'new'}, set(), ['2'], kinds=('scene',))
        assert store.connection.execute('PRAGMA user_version').fetchone() == (7,)
        assert not store.snapshot('1', 'scene')[0]['contributions']


def test_recovery_metadata_survives_reopen_and_preview_clear_is_temporary(tmp_path):
    with state(tmp_path) as store:
        store.prepare('1', {'scene': {'owned_ids': ['123'], 'contributions': []}})
        recovery.abandon(store, lambda _: None)
    before = state_path(tmp_path).read_bytes()
    with state(tmp_path, dry_run=True) as store:
        assert store.preview_source and recovery.publications(store, '1', 'scene')
        store.publish('1', {'scene': {'owned_ids': ['123'], 'contributions': []}})
        assert not recovery.publications(store, '1', 'scene')
    assert state_path(tmp_path).read_bytes() == before
    with state(tmp_path) as store:
        assert recovery.publications(store, '1', 'scene')


def test_abandon_refuses_active_writer_and_does_not_change_another_stash_scope(tmp_path):
    with merge.StateStore(state_path(tmp_path), 'other') as other:
        other.prepare('2', {'scene': {'owned_ids': [], 'contributions': []}})
    with state(tmp_path) as store:
        store.prepare('1', {'markers': {'contributions': []}})
        with pytest.raises(RuntimeError, match='Another merged-output process'):
            state(tmp_path, allow_cleanup=True)
        recovery.abandon(store, lambda _: None)
    with merge.StateStore(state_path(tmp_path), 'other') as other:
        assert other.has_pending_publications()


def test_merge_with_new_yaml_can_restore_only_ai_evidence_after_abandon(tmp_path, monkeypatch):
    stash = SQLStash(); seed(tmp_path, stash)
    install(monkeypatch, stash)
    stash.fail_after = 'CleanupDeleteMarkers'
    assert app.main([*options(tmp_path, ['abc']), '--cleanup-strategy', 'sql']) == 1
    before_raw = raw_results(tmp_path)
    new = options(tmp_path, ['def']); new[new.index('--cleanup-only')] = '--merge-only'
    assert app.main([*new, '--abandon-pending']) == 0
    assert [m['seconds'] for m in stash.scene['scene_markers']] == [0, 10]
    assert all(m['primary_tag']['name'] == 'ai:abc' for m in stash.scene['scene_markers'])
    assert raw_results(tmp_path) == before_raw


def test_abandoned_cleanup_can_be_replaced_and_interrupted_again(tmp_path, monkeypatch):
    stash = SQLStash(); seed(tmp_path, stash)
    install(monkeypatch, stash)
    for label in ('abc', 'def'):
        stash.fail_after = 'CleanupDeleteMarkers'
        assert app.main([*options(tmp_path, [label]), '--cleanup-strategy', 'sql', '--abandon-pending']) == 1
    with state(tmp_path, allow_cleanup=True) as store:
        assert store.cleanup_pending and store.cleanup_generation == 2
        assert set(store.abandoned_cleanups) == {1}
    assert app.main(command(tmp_path)) == 0
    with state(tmp_path) as store:
        assert not store.cleanup_pending and set(store.abandoned_cleanups) == {1, 2}


@pytest.mark.parametrize('flags', [
    ['--output-mode', 'separate'], ['--download-models'], ['--restart'], ['--sync-marker-groups-only'],
    ['--prune-empty-tags-only'], ['--scene-id', '1'], ['--markers-only'], ['--validate-hierarchy']])
def test_incompatible_cli_modes_are_rejected_before_state_changes(tmp_path, flags):
    with pytest.raises(SystemExit) as error: app.main(command(tmp_path, *flags))
    assert error.value.code == 2 and not state_path(tmp_path).exists()


def test_no_pending_standalone_dry_run_does_not_create_state(tmp_path):
    assert app.main(command(tmp_path, '--dry-run')) == 0
    assert not state_path(tmp_path).exists()


@pytest.mark.parametrize('applied', [False, True])
def test_same_interval_selects_observed_extras_from_old_or_pending_plan(tmp_path, applied):
    stash = FastStash()
    seed(tmp_path, stash, [span('wd', 0, 2, 'core', extras=['old'])])
    with state(tmp_path) as store:
        store.prepare('1', {'markers': {'contributions': [span('wd', 0, 2, 'core', extras=['new'])]}})
        if applied:
            marker = stash.scene['scene_markers'][0]
            marker['tags'] = [t for t in marker['tags'] if t['name'] != 'ai:old'] + [stash.tag('ai:new')]
        recovery.abandon(store, lambda _: None)
        observed, _ = recovery.reconcile(app, stash, store, stash.find_scene('1'), 'markers', store.snapshot('1', 'markers')[0])
        assert observed['contributions'][0]['extra_tags'] == ['new' if applied else 'old']


def test_local_abandonment_is_atomic_including_schema_version(tmp_path):
    with state(tmp_path) as store:
        store.prepare('1', {'scene': {'owned_ids': ['1'], 'contributions': []}})
        cleanup.begin(store, 'old', {'abc'}, set(), ['1'])
        version = store.connection.execute('PRAGMA user_version').fetchone()
        store.connection.execute("CREATE TRIGGER fail_archive BEFORE INSERT ON recovery_publications "
                                 "BEGIN SELECT RAISE(ABORT,'injected archive failure'); END")
        with pytest.raises(sqlite3.IntegrityError, match='injected archive failure'):
            recovery.abandon(store, lambda _: None)
        assert store.cleanup_pending == 'old' and store.has_pending_publications()
        assert store.connection.execute('PRAGMA user_version').fetchone() == version
        assert not recovery.publications(store, '1', 'scene')


@pytest.mark.parametrize('applied', [False, True])
def test_marker_title_selects_observed_model_votes_before_source_cleanup(tmp_path, applied):
    stash = FastStash()
    seed(tmp_path, stash, [span('wd', 0, 2, 'core')])
    with state(tmp_path) as store:
        store.prepare('1', {'markers': {'contributions': [span('wd', 0, 2, 'core'), span('oppai', 0, 2, 'core')]}})
        if applied: stash.scene['scene_markers'][0]['title'] = '[AI] core [wd+op]'
        recovery.abandon(store, lambda _: None)
        observed, _ = recovery.reconcile(app, stash, store, stash.find_scene('1'), 'markers', store.snapshot('1', 'markers')[0])
        cores = empty_policies(); cores['wd'] = frozenset({'core'})
        kept = merge.cleanup_support(app, store, '1', observed['contributions'], {'wd'}, cores, empty_policies())
        assert {s['source'] for s in kept} == ({'oppai'} if applied else set())
