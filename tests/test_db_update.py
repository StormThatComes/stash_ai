"""Offline schema upgrades preserve evidence and resumable work in every scope."""
from tests.helpers.evidence import inference_record

import sqlite3
import zlib

import pytest

from tests.helpers.runtime import runtime as app
from stash_ai.db.checks import check_database
from stash_ai.db.update import update_database
from stash_ai.storage.schema import STATE_TABLES
from stash_ai.publication import fingerprints as performance
import stash_ai.storage.merge_checks as performance_merge_checks
from stash_ai.commands import arguments
from stash_ai.storage.store import StateStore
from stash_ai.storage.codec import encode
from stash_ai.storage.database import APPLICATION_ID, LATEST_SCHEMA_VERSION


def seed_version(path, version):
    segment = {'mode': 'scene', 'meta': {'origin': 'inference', 'inference_signature': 'saved'},
               'items': [{'tag': 'example', 'score': .9}]}
    segment = inference_record(**segment)
    for scope in ('scope-a', 'scope-b'):
        with StateStore(path, scope) as state:
            state.put('1', 'wd', {'scene': segment})
    with sqlite3.connect(path) as db:
        db.execute(f'PRAGMA user_version={version}')


def rows(path, table):
    with sqlite3.connect(path) as db:
        return db.execute(f'SELECT * FROM {table} ORDER BY 1,2').fetchall()


@pytest.mark.parametrize('version', [18, 19, 20])
@pytest.mark.parametrize('dry_run', [False, True])
def test_runtime_requires_explicit_upgrade_without_changing_old_database(tmp_path, version, dry_run):
    path = tmp_path / 'state.sqlite3'
    seed_version(path, version)
    before = path.read_bytes(), path.stat().st_mtime_ns
    with pytest.raises(ValueError, match='--db-update'):
        StateStore(path, 'scope-a', dry_run=dry_run)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    update_database(path)
    with StateStore(path, 'scope-a', dry_run=dry_run) as state:
        assert state.records('1', 'scene')['wd']


def test_update_backfills_missing_and_corrupt_metadata_in_every_scope(tmp_path):
    path = tmp_path / 'state.sqlite3'
    seed_version(path, LATEST_SCHEMA_VERSION)
    evidence = rows(path, 'results')
    with sqlite3.connect(path) as db:
        db.execute("DELETE FROM result_metadata WHERE scope='scope-a'")
        db.execute("UPDATE result_metadata SET data='corrupt' WHERE scope='scope-b'")
    update_database(path)
    from stash_ai.storage.codec import decode
    assert [(row[:-1], decode(row[-1])) for row in rows(path, 'results')] == [
        (row[:-1], decode(row[-1])[-1] if isinstance(decode(row[-1]), list) else decode(row[-1])) for row in evidence]
    assert len(rows(path, 'result_metadata')) == 2
    for scope in ('scope-a', 'scope-b'):
        with StateStore(path, scope) as state:
            state.unpack = lambda *_: pytest.fail('runtime repeated historical backfill')
            assert state.result_metadata('1', 'scene', 'wd')['populated']


@pytest.mark.parametrize('damage', ['snapshot', 'build_options', 'core_exclusions', 'scene_exclusions'])
def test_invalid_snapshot_upgrade_rolls_back_all_publication_changes(tmp_path, damage):
    path = tmp_path / 'state.sqlite3'
    with StateStore(path, 'scope') as state:
        state.publish('1', {'scene': {'contributions': []}})
        state.publish('2', {'scene': {'contributions': []}})
        malformed = []
        if damage != 'snapshot':
            from stash_ai.evidence.temporal.policy import build_options
            from stash_ai.models import SOURCES
            options = {**build_options(app), 'core_exclusions': {source: [] for source in SOURCES}}
            malformed = {'contributions': [{'build_options': options}]}
            # These values were never saved: migration must not use today's app.
            if damage == 'build_options':
                options.pop('settings')
            elif damage == 'core_exclusions':
                options['core_exclusions'].pop('wd')
            else:
                malformed['contributions'][0]['build_options'] = {'policy': options['policy'], 'max_tags': 0}
        with state.connection:
            state.connection.execute("UPDATE snapshots SET published=? WHERE scene='2'", (encode({'__stash_ai_snapshot__': 4, 'snapshot': malformed, 'proofs': []}),))
            state.connection.execute('PRAGMA user_version=18')
    before = rows(path, 'snapshots')
    with pytest.raises(ValueError, match='historical snapshot|frozen build options|historical frozen exclusions'):
        update_database(path)
    assert rows(path, 'snapshots') == before
    with sqlite3.connect(path) as db:
        assert db.execute('PRAGMA user_version').fetchone() == (18,)


@pytest.mark.parametrize('version', range(18, LATEST_SCHEMA_VERSION + 1))
def test_update_migrates_every_supported_version_and_all_scopes(tmp_path, version):
    path = tmp_path / 'state.sqlite3'
    seed_version(path, version)
    evidence = rows(path, 'results')

    assert update_database(path) == (version, LATEST_SCHEMA_VERSION)

    from stash_ai.storage.codec import decode
    assert [(row[:-1], decode(row[-1])) for row in rows(path, 'results')] == [
        (row[:-1], decode(row[-1])[-1] if isinstance(decode(row[-1]), list) else decode(row[-1])) for row in evidence]
    with sqlite3.connect(path) as db:
        assert db.execute('PRAGMA user_version').fetchone() == (LATEST_SCHEMA_VERSION,)
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert STATE_TABLES <= tables
        assert 'refinement_progress' not in tables
        assert db.execute('SELECT scope,populated,signature FROM result_index ORDER BY scope').fetchall() == [
            ('scope-a', 1, 'saved'), ('scope-b', 1, 'saved')]
    assert check_database(path, 'scope-a')['ok']
    assert update_database(path) == (LATEST_SCHEMA_VERSION, LATEST_SCHEMA_VERSION)
    from stash_ai.storage.codec import decode
    assert [(row[:-1], decode(row[-1])) for row in rows(path, 'results')] == [
        (row[:-1], decode(row[-1])[-1] if isinstance(decode(row[-1]), list) else decode(row[-1])) for row in evidence]


@pytest.mark.parametrize('version', [18, 19, 20, LATEST_SCHEMA_VERSION])
def test_dry_run_only_reports_target_without_modifying_database_or_lock(tmp_path, version):
    path = tmp_path / 'state.sqlite3'
    seed_version(path, version)
    lock = path.with_name(path.name + '.lock')
    lock.unlink(missing_ok=True)
    before = path.read_bytes(), path.stat().st_mtime_ns

    assert update_database(path, dry_run=True) == (version, LATEST_SCHEMA_VERSION)

    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    assert not lock.exists()


def test_update_preserves_pending_and_recovery_work_checkpoints_and_thresholds(tmp_path):
    path = tmp_path / 'state.sqlite3'
    seed_version(path, LATEST_SCHEMA_VERSION)
    with StateStore(path, 'scope-b') as state:
        state.prepare('1', {'scene': {'owned_ids': [], 'contributions': []}})
        with state.connection as db:
            db.execute('INSERT INTO cleanup_state VALUES (?,?,?,?)',
                       ('scope-b', 1, encode({'global': {'example': 1}}), 'pending-cleanup'))
            db.execute('INSERT INTO recovery_cleanups VALUES (?,?,?)', ('scope-b', 0, '["scene"]'))
            db.execute('INSERT INTO recovery_publications VALUES (?,?,?,?,?)',
                       ('scope-a', '1', 'scene', 'old-operation', encode({'__stash_ai_snapshot__': 4, 'snapshot': {'contributions': []}, 'proofs': []})))
            db.execute('INSERT INTO checkpoints VALUES (?,?,?,?,?,?,?,?)',
                       ('scope-a', '2', 'wd', 'partial', 1, 2, 1, zlib.compress(b'\0' * 4)))
            db.execute('INSERT INTO checkpoint_frames VALUES (?,?,?,?,?)',
                       ('scope-a', '2', 'wd', 1, encode([])))
            db.execute('INSERT INTO checkpoint_analysis VALUES (?,?,?,?)',
                       ('scope-a', '2', 'wd', encode({})))
            db.execute('INSERT INTO refresh_requests VALUES (?,?,?,?)', ('scope-a', '2', 'wd', 'scene'))
            db.execute('INSERT INTO result_thresholds VALUES (?,?,?,?,?)', ('scope-a', '1', 'wd', 'scene', .8))

    with sqlite3.connect(path) as connection:
        connection.execute('PRAGMA user_version=18')
    preserved = ('results', 'snapshots', 'cleanup_state', 'recovery_cleanups', 'recovery_publications',
                 'checkpoints', 'checkpoint_frames', 'checkpoint_analysis', 'refresh_requests',
                 'result_thresholds', 'result_metadata')
    before = {table: rows(path, table) for table in preserved}

    update_database(path)

    after = {table: rows(path, table) for table in preserved}
    from stash_ai.db.snapshot_migration import decode_legacy
    for table, columns in (('snapshots', (3, 4)), ('recovery_publications', (4,))):
        def logical(values):
            return [tuple(decode_legacy(value) if i in columns and value is not None else value for i, value in enumerate(row)) for row in values]
        after[table], before[table] = logical(after[table]), logical(before[table])
    assert after == before


def test_update_refuses_an_active_writer_and_releases_failed_lock(tmp_path):
    path = tmp_path / 'state.sqlite3'
    seed_version(path, LATEST_SCHEMA_VERSION)
    with StateStore(path, 'scope-a'):
        before = path.read_bytes()
        with pytest.raises(RuntimeError, match='Another merged-output process'):
            update_database(path)
        assert path.read_bytes() == before
    assert update_database(path) == (LATEST_SCHEMA_VERSION, LATEST_SCHEMA_VERSION)


def test_failed_migration_can_be_retried_and_does_not_claim_latest_version(tmp_path, monkeypatch):
    path = tmp_path / 'state.sqlite3'
    seed_version(path, 18)
    evidence = rows(path, 'results')
    with monkeypatch.context() as patch:
        def fail(_):
            raise RuntimeError('interrupted migration')
        patch.setattr(performance_merge_checks, 'initialize', fail)
        with pytest.raises(RuntimeError, match='interrupted migration'):
            update_database(path)
    with sqlite3.connect(path) as db:
        assert db.execute('PRAGMA user_version').fetchone()[0] < LATEST_SCHEMA_VERSION
    assert rows(path, 'results') == evidence

    update_database(path)

    from stash_ai.storage.codec import decode
    assert [(row[:-1], decode(row[-1])) for row in rows(path, 'results')] == [
        (row[:-1], decode(row[-1])[-1] if isinstance(decode(row[-1]), list) else decode(row[-1])) for row in evidence]
    assert check_database(path, 'scope-a')['ok']


@pytest.mark.parametrize('dry_run', [False, True])
@pytest.mark.parametrize('kind', ['missing', 'foreign', 'future'])
def test_update_rejects_unsupported_databases_without_changing_them(tmp_path, dry_run, kind):
    path = tmp_path / 'state.sqlite3'
    if kind == 'future':
        seed_version(path, LATEST_SCHEMA_VERSION + 1)
    elif kind == 'foreign':
        with sqlite3.connect(path) as db:
            db.execute('CREATE TABLE unrelated (value TEXT)')
    path.with_name(path.name + '.lock').unlink(missing_ok=True)
    before = (path.read_bytes(), path.stat().st_mtime_ns) if path.exists() else None

    with pytest.raises((FileNotFoundError, ValueError)):
        update_database(path, dry_run=dry_run)

    assert ((path.read_bytes(), path.stat().st_mtime_ns) if path.exists() else None) == before
    assert not path.with_name(path.name + '.lock').exists()


def test_db_update_cli_stays_offline_and_reports_dry_run_and_idempotence(tmp_path, monkeypatch, capsys):
    path = tmp_path / 'state.sqlite3'
    seed_version(path, 18)
    for name in ('load_model_settings', 'create_stash_client', 'validate_config', 'load_tagger'):
        monkeypatch.setattr(app.services, name, lambda *a, **kw: pytest.fail('db-update must stay offline'))
    monkeypatch.setattr(app, 'STASH_URL', 'unused-invalid-url')
    command = ['--state-db', str(path), '--db-update']
    before = path.read_bytes()

    assert app.main([*command, '--dry-run']) == 0
    assert path.read_bytes() == before
    assert 'DRY RUN: state DB schema v18 -> v21' in capsys.readouterr().out
    assert app.main(command) == 0
    assert 'State DB updated: schema v18 -> v21' in capsys.readouterr().out
    assert app.main(command) == 0
    assert 'up to date: v21' in capsys.readouterr().out
    assert app.main(['--state-db', str(tmp_path / 'missing.sqlite3'), '--db-update']) == 1
    assert 'does not exist' in capsys.readouterr().out


def test_db_update_cli_reports_unreadable_legacy_evidence_without_claiming_success(tmp_path, capsys):
    path = tmp_path / 'state.sqlite3'
    seed_version(path, 18)
    with sqlite3.connect(path) as db:
        db.execute('UPDATE results SET data=?', (b'not-compressed-json',))
    evidence = rows(path, 'results')

    assert app.main(['--state-db', str(path), '--db-update']) == 1

    assert 'Saved state could not be migrated' in capsys.readouterr().out
    assert rows(path, 'results') == evidence
    with sqlite3.connect(path) as db:
        assert db.execute('PRAGMA user_version').fetchone() == (18,)


@pytest.mark.parametrize('extra', [
    ['--db-check'], ['--db-compact'], ['--merge-only'], ['--models', 'wd'],
    ['--output-mode', 'separate'], ['--scene-id', '1'], ['--abandon-pending'],
])
def test_db_update_cli_rejects_unrelated_modes_and_modifiers(extra):
    with pytest.raises(SystemExit) as error:
        arguments.parse_args(app, ['--db-update', *extra])
    assert error.value.code == 2


@pytest.mark.parametrize('version', range(18))
@pytest.mark.parametrize('dry_run', [False, True])
def test_pre_v18_upgrade_requires_bridge_without_lock_or_mutation(tmp_path, version, dry_run):
    from stash_ai.storage.schema import UPGRADE_BRIDGE_COMMIT
    path = tmp_path / 'unsupported.sqlite3'
    seed_version(path, version)
    lock = path.with_name(path.name + '.lock')
    lock.unlink()
    before = path.read_bytes(), path.stat().st_mtime_ns
    with pytest.raises(ValueError, match='supported minimum v18') as error:
        update_database(path, dry_run=dry_run)
    assert UPGRADE_BRIDGE_COMMIT in str(error.value) and '--db-update' in str(error.value)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    assert not lock.exists()
    with pytest.raises(ValueError, match='supported minimum v18'):
        StateStore(path, 'scope-a', dry_run=dry_run)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_pre_v18_cli_reports_bridge_before_accessing_stash(tmp_path, monkeypatch, capsys):
    path = tmp_path / 'unsupported.sqlite3'
    seed_version(path, 17)
    before = path.read_bytes()
    monkeypatch.setattr(app.services, 'create_stash_client', lambda *_: pytest.fail('accessed Stash'))
    assert app.main(['--state-db', str(path), '--db-update']) == 1
    assert 'supported minimum v18' in capsys.readouterr().out
    assert path.read_bytes() == before


@pytest.mark.parametrize('version', [18, 19, 20])
def test_publication_upgrade_decodes_each_blob_once_and_drops_only_retired_options(tmp_path, monkeypatch, version):
    from copy import deepcopy
    from stash_ai.db import snapshot_migration
    from stash_ai.storage.snapshots import compact, decode as decode_snapshot
    from tests.test_snapshot_proofs import shared_snapshot
    path = tmp_path / 'publications.sqlite3'
    with StateStore(path, 'first'):
        pass
    saved = shared_snapshot()
    for item in saved['contributions']:
        item['requires_rebase'] = True
        item['retired_build_options'] = {'algorithm': 'historical', 'unused': 'old policy'}
    old = compact(saved)
    old['__stash_ai_snapshot__'] = 4 if version < 20 else 5
    packed = encode(old)
    with sqlite3.connect(path) as connection:
        for scope in ('first', 'second'):
            connection.execute('INSERT INTO snapshots VALUES (?,?,?,?,?)', (scope, '1', 'markers', packed, packed))
            connection.execute('INSERT INTO recovery_publications VALUES (?,?,?,?,?)', (scope, '1', 'markers', 'op', packed))
        connection.execute(f'PRAGMA user_version={version}')
    calls = []
    decode = snapshot_migration.decode
    def counted(blob):
        calls.append(blob)
        return decode(blob)
    monkeypatch.setattr(snapshot_migration, 'decode', counted)
    assert update_database(path) == (version, LATEST_SCHEMA_VERSION)
    assert len(calls) == 6
    expected = deepcopy(saved)
    for item in expected['contributions']:
        item.pop('retired_build_options')
        item.pop('build_options')
    with sqlite3.connect(path) as connection:
        blobs = [blob for row in connection.execute('SELECT published,pending FROM snapshots') for blob in row]
        blobs += [row[0] for row in connection.execute('SELECT data FROM recovery_publications')]
    for blob in blobs:
        restored = decode_snapshot(blob)
        assert restored == expected
        assert restored['contributions'][0]['extra_support']['extra'] is restored['contributions'][1]['extra_support']['extra']
