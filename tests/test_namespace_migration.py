"""Fresh inference replaces source-owned output without a persistent import ledger."""
from functools import partial
from stash_ai.tools.namespace_migration import main as namespace_main
from copy import deepcopy
import sqlite3

import pytest

from stash_ai import db
from stash_ai.evidence.labels import AI
from stash_ai.models import WD, JOYTAG
from stash_ai.storage.store import StateStore, scope_for
from tests.helpers.evidence import inject_records
from tests.helpers.runtime import runtime as app
migrate_namespace = partial(namespace_main, runtime=app)
from tests.test_result_thresholds import outputs

pytestmark = pytest.mark.usefixtures("reset_merge")


def seed_source(p):
    p.scores['wd'][:] = 0
    p.scores['wd'][:, 0] = .95
    p.stash.scene['tags'] = [p.stash.tag('wd:common'), p.stash.tag('jt:obsolete'), p.stash.tag('human')]
    for source, identifier in ((WD, '100'), (JOYTAG, '101')):
        marker = p.stash.marker(source, 'common', identifier)
        marker['end_seconds'] = 10
        marker['tags'].append(p.stash.tag('manual-' + identifier))
    manual = p.stash.marker(WD, 'common', 'manual', manual=True)
    manual.update(seconds=50, end_seconds=52)
    return deepcopy(manual)


def test_transition_infers_fresh_preserves_manual_data_and_marker_ids_and_is_one_time(pipeline):
    p = pipeline
    manual = seed_source(p)
    assert migrate_namespace([*p.args]) == 0
    assert len(p.loaded) == 1
    assert {tag['name'] for tag in p.stash.scene['tags']} == {'ai:common', 'human'}
    owned = [marker for marker in p.stash.scene['scene_markers'] if marker['primary_tag']['name'] == 'ai:common']
    assert len(owned) == 1 and owned[0]['id'] in {'100', '101'}
    assert (owned[0]['seconds'], owned[0]['end_seconds']) == (0, 10)
    assert {'manual-100', 'manual-101', AI.owner_tag} <= {tag['name'] for tag in owned[0]['tags']}
    assert manual in p.stash.scene['scene_markers']
    assert 'wd:common' in p.stash.tags
    with StateStore(p.path, scope_for(app.STASH_URL)) as store:
        assert not store.connection.execute("SELECT 1 FROM sqlite_master WHERE name='legacy_imports'").fetchone()
        assert store.model_record('1', 'markers', 'wd')['mode'] == 'frames'
        assert store.model_record('1', 'markers', 'wd')['meta']['origin'] == 'inference'
    before = deepcopy(p.stash.scene)
    p.stash.writes.clear()
    assert migrate_namespace([*p.args]) == 0
    assert len(p.loaded) == 1 and not p.stash.writes and p.stash.scene == before
    assert app.main([*p.args, '--merge-only']) == 0
    assert len(p.loaded) == 1
    assert db.check_database(p.path, scope_for(app.STASH_URL))['ok']


def test_ordinary_cli_does_not_import_migration_tools(pipeline, monkeypatch):
    import builtins
    original = builtins.__import__
    def checked(name, *args, **kwargs):
        if name.startswith('stash_ai.tools'):
            pytest.fail('ordinary runtime imported migration tools')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', checked)
    assert app.main(pipeline.args) == 0



def test_transition_dry_run_keeps_stash_and_source_database_unchanged(pipeline):
    p = pipeline
    seed_source(p)
    with StateStore(p.path, scope_for(app.STASH_URL)):
        pass
    before = p.path.read_bytes(), deepcopy(p.stash.scene)
    p.stash.dry_run = True
    assert migrate_namespace([*p.args, '--dry-run']) == 0
    assert (p.path.read_bytes(), p.stash.scene) == before
    assert not p.stash.writes


def test_unresolved_replacement_preserves_source_objects(pipeline, monkeypatch):
    p = pipeline
    seed_source(p)
    p.stash.tag(AI.owner_tag)
    monkeypatch.setattr(app, 'CREATE_MISSING', False)
    before = deepcopy(p.stash.scene)
    assert migrate_namespace([*p.args]) == 0
    assert p.stash.scene['scene_markers'] == before['scene_markers']
    assert 'wd:common' in {tag['name'] for tag in p.stash.scene['tags']}
    monkeypatch.setattr(app, 'CREATE_MISSING', True)
    assert migrate_namespace([*p.args]) == 0
    assert any(marker['id'] == '100' and marker['primary_tag']['name'] == 'ai:common'
               for marker in p.stash.scene['scene_markers'])


@pytest.mark.parametrize('mode,untouched', [('--markers-only', 'tags'), ('--scene-tags-only', 'scene_markers')])
def test_partial_transition_preserves_other_output(pipeline, mode, untouched):
    p = pipeline
    seed_source(p)
    before = deepcopy(p.stash.scene[untouched])
    assert migrate_namespace([*p.args, mode]) == 0
    assert p.stash.scene[untouched] == before


def test_failed_fresh_inference_keeps_old_objects_and_blocks_publication(pipeline):
    p = pipeline
    seed_source(p)
    before = deepcopy(p.stash.scene)
    p.fail['wd'] = 2
    with pytest.raises(KeyboardInterrupt):
        migrate_namespace([*p.args])
    assert p.stash.scene == before and not p.stash.writes
    p.fail.clear()
    assert migrate_namespace([*p.args]) == 0


@pytest.mark.parametrize('operation', ['update_marker', 'update_scene_tags', 'delete_marker'])
@pytest.mark.parametrize('lost_response', [False, True])
def test_transition_recovers_after_each_partial_publication(pipeline, monkeypatch, operation, lost_response):
    p = pipeline
    manual = seed_source(p)
    actual = getattr(p.stash, operation)
    calls = 0

    def fail(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            if lost_response:
                actual(*args)
            raise RuntimeError('interrupted transition')
        return actual(*args)

    with monkeypatch.context() as interrupted:
        interrupted.setattr(p.stash, operation, fail)
        assert migrate_namespace([*p.args]) == 1
    with StateStore(p.path, scope_for(app.STASH_URL)) as store:
        assert store.has_pending_publication('1')
    assert migrate_namespace([*p.args]) == 0
    assert manual in p.stash.scene['scene_markers']
    assert len([marker for marker in p.stash.scene['scene_markers'] if marker['primary_tag']['name'] == 'ai:common']) == 1
    assert {tag['name'] for tag in p.stash.scene['tags']} == {'human', 'ai:common'}
    with StateStore(p.path, scope_for(app.STASH_URL)) as store:
        assert not store.has_pending_publication('1')




@pytest.mark.parametrize('flag', ['--import-legacy-only', '--prune-legacy-imports', '--migrate-legacy-only', '--migrate-point-markers-only'])
def test_retired_import_commands_are_not_parser_aliases(flag):
    with pytest.raises(SystemExit):
        app.main([flag])


@pytest.mark.parametrize('flag', ['--resume', '--merge-only', '--abandon-pending', '--db-update'])
def test_namespace_tool_rejects_non_inference_modes_before_runtime_setup(monkeypatch, flag):
    monkeypatch.setattr(app.services, 'load_model_settings', lambda *args: pytest.fail('model settings loaded'))
    monkeypatch.setattr(app.services, 'create_stash_client', lambda *args: pytest.fail('Stash opened'))
    with pytest.raises(SystemExit):
        migrate_namespace([flag])
