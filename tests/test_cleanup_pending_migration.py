"""Offline fingerprint conversion preserves the interrupted cleanup's intent."""
import sqlite3

import pytest

from stash_ai.cleanup.fingerprint import fingerprint
from stash_ai.models import SOURCES
from stash_ai.runtime.context import RuntimeContext
from stash_ai.signatures import digest
from stash_ai.storage.codec import encode
from stash_ai.storage.store import StateStore, scope_for
from stash_ai.tools.cleanup_pending_migration import main, migrate


def policies():
    return {s: ['blocked', s] for s in SOURCES}, {s: ['core'] for s in SOURCES}


def historical(tags, cores, kinds):
    value = {'tags': {s: sorted(tags[s]) for s in SOURCES},
             'cores': {s: sorted(cores[s]) for s in SOURCES}, 'marker_blocklist': []}
    if kinds != ['scene', 'markers']:
        value['output_kinds'] = kinds
    return digest(value)


def seed(path, scope, kinds):
    tags, cores = policies()
    with StateStore(path, scope) as store:
        with store.connection:
            store.connection.execute('INSERT INTO cleanup_state VALUES (?,?,?,?)',
                (scope, 7, encode({'global': {'blocked': 3}, 'core': {'core': 7}}),
                 historical(tags, cores, kinds)))
            store.connection.execute('INSERT INTO recovery_cleanups VALUES (?,?,?)',
                                     (scope, 3, '["markers"]'))


@pytest.mark.parametrize('kinds', [['scene', 'markers'], ['markers'], ['scene']])
def test_conversion_changes_only_pending_in_selected_scope_and_is_idempotent(tmp_path, kinds):
    path = tmp_path/'state.sqlite3'
    seed(path, 'first', kinds)
    seed(path, 'second', kinds)
    with sqlite3.connect(path) as db:
        before = db.execute('SELECT * FROM cleanup_state ORDER BY scope').fetchall()
        recovery = db.execute('SELECT * FROM recovery_cleanups').fetchall()
    tags, cores = policies()
    raw = path.read_bytes()
    with StateStore(path, 'first', dry_run=True, allow_cleanup=True) as store:
        assert migrate(store, tags, cores, kinds, dry_run=True)
        assert store.cleanup_pending == historical(tags, cores, kinds)
    assert path.read_bytes() == raw
    with StateStore(path, 'first', allow_cleanup=True) as store:
        assert migrate(store, tags, cores, kinds)
        assert store.cleanup_pending == fingerprint(tags, cores, kinds)
        assert not migrate(store, tags, cores, kinds)
    with sqlite3.connect(path) as db:
        after = db.execute('SELECT * FROM cleanup_state ORDER BY scope').fetchall()
        assert after[0][:3] == before[0][:3]
        assert after[1] == before[1]
        assert db.execute('SELECT * FROM recovery_cleanups').fetchall() == recovery


@pytest.mark.parametrize('damage', ['tags', 'cores', 'scope'])
def test_conversion_rejects_different_per_model_policies_or_output_scope(tmp_path, damage):
    path = tmp_path/'state.sqlite3'
    seed(path, 'scope', ['scene', 'markers'])
    tags, cores = policies()
    kinds = ['scene', 'markers']
    if damage == 'scope':
        kinds = ['markers']
    else:
        (tags if damage == 'tags' else cores)['wd'].append('changed')
    with StateStore(path, 'scope', allow_cleanup=True) as store:
        before = store.connection.execute('SELECT * FROM cleanup_state').fetchall()
        with pytest.raises(RuntimeError, match='does not match'):
            migrate(store, tags, cores, kinds)
        assert store.connection.execute('SELECT * FROM cleanup_state').fetchall() == before


def test_cli_reads_yaml_and_stays_offline_without_models(tmp_path, monkeypatch):
    app = RuntimeContext()
    path = tmp_path/'state.sqlite3'
    scope = scope_for(app.STASH_URL)
    tags = {s: ['blocked'] for s in SOURCES}
    cores = {s: [] for s in SOURCES}
    with StateStore(path, scope) as store:
        with store.connection:
            store.connection.execute('INSERT INTO cleanup_state VALUES (?,?,?,?)',
                (scope, 1, encode({}), historical(tags, cores, ['markers'])))
    tag_file, core_file = tmp_path/'tags.yml', tmp_path/'cores.yml'
    tag_file.write_text('version: 1\nexcluded_scene_tags: [blocked]\n')
    core_file.write_text('version: 1\nexcluded_core_tags: []\n')
    def forbidden(*args, **kwargs):
        pytest.fail('Maintenance must not connect to Stash or load models')
    for service in ('create_stash_client', 'load_tagger', 'load_model_settings'):
        monkeypatch.setattr(app.services, service, forbidden)
    command = ['--state-db', str(path), '--cleanup-markers-only',
               '--marker-core-exclusions', str(core_file), '--scene-tag-exclusions', str(tag_file)]
    raw = path.read_bytes()
    assert main([*command, '--dry-run'], runtime=app) == 0
    assert path.read_bytes() == raw
    assert main(command, runtime=app) == main(command, runtime=app) == 0
    with StateStore(path, scope, allow_cleanup=True) as store:
        assert store.cleanup_pending == fingerprint(tags, cores, ['markers'])
