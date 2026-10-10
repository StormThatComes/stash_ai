"""Spelling migration keeps tag identities and never claims unrelated tags."""
from copy import deepcopy

import pytest

from tests.helpers.runtime import runtime as app
from tests.helpers.stash import Stash
from tests.helpers.cleanup import install
from stash_ai.publication import groups
from stash_ai.evidence.labels import AI
from stash_ai.storage.store import StateStore, scope_for
from stash_ai.stash_api.client import StashClient
from stash_ai.commands.arguments import parse_args

pytestmark = pytest.mark.usefixtures('reset_progress')


def legacy(stash, core='abc'):
    tag = stash.tag('m:' + core + ':occurance')
    tag.update(description=groups.group_description(core, True), ignore_auto_tag=False,
               parents=[stash.tag('manual parent')])
    return tag


def argv(tmp_path, *extra):
    return ['--rename-occurrence-groups', '--state-db', str(tmp_path/'state.sqlite3'),
            '--no-run-history', *extra]


def test_rename_preserves_id_metadata_and_every_assignment_and_is_idempotent(tmp_path, monkeypatch, capsys):
    stash = Stash(); old = legacy(stash); before = deepcopy(old)
    stash.scene['tags'] = [old]
    stash.scene['scene_markers'] = [dict(id='77', title='Occurrence example', seconds=12,
        end_seconds=None, primary_tag=stash.tag('ai:abc'), tags=[old])]
    stash.tag('child')['parents'] = [old]
    manual = stash.tag('m:manual:occurance')
    regular = stash.tag('m:literal:occurance')
    regular['description'] = groups.group_description('literal:occurance', False)
    malformed = stash.tag('m:broken:occurance'); malformed['description'] = groups.DESCRIPTION + '{}'
    unrelated = deepcopy([manual, regular, malformed])
    geometry = {k: deepcopy(v) for k,v in stash.scene['scene_markers'][0].items() if k != 'tags'}
    install(monkeypatch, stash)
    monkeypatch.setattr(app.services, 'load_model_settings', lambda *a: pytest.fail('loaded models'))
    assert app.main(argv(tmp_path)) == 0
    tag = stash.tags['m:abc:occurrence']
    assert {k:v for k,v in tag.items() if k != 'name'} == {k:v for k,v in before.items() if k != 'name'}
    assert stash.scene['tags'][0] is tag and stash.scene['scene_markers'][0]['tags'][0] is tag
    assert stash.tags['child']['parents'][0] is tag
    assert {k:v for k,v in stash.scene['scene_markers'][0].items() if k != 'tags'} == geometry
    assert [manual, regular, malformed] == unrelated
    assert stash.writes == [('rename_tag', tag['id'], tag['name'])]
    assert 'm:abc:occurance' in capsys.readouterr().out
    assert app.main(argv(tmp_path)) == 0 and len(stash.writes) == 1


def test_dry_run_prints_changes_without_any_database_or_stash_writes(tmp_path, monkeypatch, capsys):
    stash = Stash(dry=True); legacy(stash); before = deepcopy(stash.tags)
    install(monkeypatch, stash)
    arguments = argv(tmp_path, '--dry-run')
    arguments.remove('--no-run-history')
    assert app.main(arguments) == 0
    assert stash.tags == before and not stash.writes
    assert '[DRY] Rename tag' in capsys.readouterr().out
    assert not (tmp_path/'state.sqlite3').exists()
    assert not list(tmp_path.glob('*.runs.sqlite3'))


@pytest.mark.parametrize('owned', [False, True])
def test_destination_collision_blocks_entire_preflight(tmp_path, monkeypatch, capsys, owned):
    stash = Stash(); legacy(stash, 'first'); legacy(stash, 'conflict')
    target = stash.tag('m:conflict:occurrence')
    if owned:
        target['description'] = groups.group_description('conflict', True)
    before = deepcopy(stash.tags)
    install(monkeypatch, stash)
    assert app.main(argv(tmp_path)) == 1
    assert not stash.writes and stash.tags == before
    assert 'no tags renamed' in capsys.readouterr().out


@pytest.mark.parametrize('race', ['ownership', 'source_name', 'destination'])
def test_fresh_reads_detect_tag_changes_before_mutation(monkeypatch, race):
    stash = Stash(); tag = deepcopy(legacy(stash))
    if race == 'ownership':
        stash.tags[tag['name']]['description'] = 'manual now'
    elif race == 'source_name':
        stash.tags[tag['name']]['name'] = 'manual now'
    else:
        stash.tag('m:abc:occurrence')
    with pytest.raises(ValueError, match='changed|collision'):
        groups.rename_legacy_group(stash, tag)
    assert not stash.writes


def test_retry_after_ambiguous_commit_finishes_remaining_groups(tmp_path, monkeypatch):
    stash = Stash(); legacy(stash, 'first'); legacy(stash, 'second'); install(monkeypatch, stash)
    original = stash.rename_tag
    def commit_then_fail(*args):
        original(*args)
        raise RuntimeError('Connection lost after commit')
    monkeypatch.setattr(stash, 'rename_tag', commit_then_fail)
    assert app.main(argv(tmp_path)) == 1
    assert 'm:first:occurrence' in stash.tags and 'm:second:occurance' in stash.tags
    monkeypatch.setattr(stash, 'rename_tag', original)
    assert app.main(argv(tmp_path)) == 0
    assert 'm:second:occurrence' in stash.tags and len(stash.writes) == 2


@pytest.mark.parametrize('dry', [False, True])
def test_normal_resolution_reuses_legacy_id_and_preview_mutation(dry):
    stash = Stash(dry); old = deepcopy(legacy(stash))
    # Avoid unrelated root/hierarchy creation in this test.
    root = stash.tag(groups.ROOT)
    stash.tags[old['name']].update(parents=[root], ignore_auto_tag=True)
    resolver = app.services.TagResolver(stash, source=AI); resolver.preload()
    from types import SimpleNamespace
    rows = []
    resolver.plan_writer = SimpleNamespace(write=rows.append)
    manager = groups.for_resolver(app, resolver)
    assert manager.resolve('abc', True) == old['id']
    assert resolver.records['m:abc:occurrence']['id'] == old['id']
    assert manager.resolve('abc', True) == old['id']
    if dry:
        assert not stash.writes and old['name'] in stash.tags
        assert rows == [dict(type='tag', operation='update', id=old['id'],
            payload=dict(id=old['id'], name='m:abc:occurrence'), reason='occurrence_group_spelling')]
    else:
        assert stash.writes == [('rename_tag', old['id'], 'm:abc:occurrence')]


def test_pending_publication_blocks_rename(tmp_path, monkeypatch):
    stash = Stash(); legacy(stash); install(monkeypatch, stash)
    with StateStore(tmp_path/'state.sqlite3', scope_for(app.STASH_URL)) as store:
        store.prepare('1', {'markers': {'contributions': []}})
    with pytest.raises(RuntimeError, match='interrupted merge'):
        app.main(argv(tmp_path))
    assert not stash.writes


@pytest.mark.parametrize('option', [['--scene-id','1'], ['--models','wd'], ['--restart'],
    ['--markers-only'], ['--profile-merge','unused.jsonl'], ['--abandon-pending'], ['--no-hierarchy']])
def test_rename_rejects_ignored_modifiers(tmp_path, option):
    with pytest.raises(SystemExit):
        parse_args(app, argv(tmp_path, *option))


def test_native_rename_sends_only_id_and_name_and_checks_confirmation(monkeypatch):
    client = object.__new__(StashClient); calls = []
    def gql(query, variables):
        calls.append((query, variables))
        return {'tagUpdate': dict(id='7', name='m:abc:occurrence')}
    monkeypatch.setattr(client, 'gql', gql)
    assert client.rename_tag('7', 'm:abc:occurrence')['id'] == '7'
    assert calls[0][1] == {'input': {'id':'7', 'name':'m:abc:occurrence'}}
    monkeypatch.setattr(client, 'gql', lambda *a: {'tagUpdate': None})
    with pytest.raises(RuntimeError, match='did not confirm'):
        client.rename_tag('7', 'm:abc:occurrence')
