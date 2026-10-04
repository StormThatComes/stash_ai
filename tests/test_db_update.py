"""Offline schema upgrades preserve evidence and resumable work in every scope."""

import sqlite3
import zlib

import pytest

from tests.helpers.runtime import runtime as app
from stash_ai import db as stash_db
from stash_ai.publication import fingerprints as performance
import stash_ai.storage.merge_checks as performance_merge_checks
from stash_ai.commands import arguments
from stash_ai.storage.store import StateStore
from stash_ai.storage.codec import encode
from stash_ai.storage.database import APPLICATION_ID, LATEST_SCHEMA_VERSION


def seed_version(path, version):
    segment = {'mode': 'scene', 'meta': {'origin': 'inference', 'inference_signature': 'saved'},
               'items': [{'tag': 'example', 'score': .9}]}
    if version < 2:
        with sqlite3.connect(path) as db:
            db.executescript(f'''
                PRAGMA application_id={APPLICATION_ID};
                CREATE TABLE results (
                    scope TEXT, scene TEXT, model TEXT, kind TEXT, data BLOB,
                    PRIMARY KEY(scope,scene,model,kind));
                CREATE TABLE snapshots (
                    scope TEXT, scene TEXT, kind TEXT, published BLOB, pending BLOB,
                    PRIMARY KEY(scope,scene,kind));
                CREATE TABLE legacy_imports (
                    scope TEXT, scene TEXT, kind TEXT, data BLOB,
                    PRIMARY KEY(scope,scene,kind));
            ''')
            for scope in ('scope-a', 'scope-b'):
                db.execute('INSERT INTO results VALUES (?,?,?,?,?)',
                           (scope, '1', 'wd', 'scene', encode([segment])))
    else:
        for scope in ('scope-a', 'scope-b'):
            with StateStore(path, scope) as state:
                state.put('1', 'wd', {'scene': segment})
    with sqlite3.connect(path) as db:
        db.execute(f'PRAGMA user_version={version}')
        if version < 4:
            db.execute('CREATE TABLE refinement_progress (data BLOB)')
            db.execute('INSERT INTO refinement_progress VALUES (?)', (encode({'retired': True}),))


def rows(path, table):
    with sqlite3.connect(path) as db:
        return db.execute(f'SELECT * FROM {table} ORDER BY 1,2').fetchall()


@pytest.mark.parametrize('version', [0, 1, 5, 8])
@pytest.mark.parametrize('dry_run', [False, True])
def test_runtime_requires_explicit_upgrade_without_changing_old_database(tmp_path, version, dry_run):
    path = tmp_path / 'state.sqlite3'
    seed_version(path, version)
    before = path.read_bytes(), path.stat().st_mtime_ns
    with pytest.raises(ValueError, match='--db-update'):
        StateStore(path, 'scope-a', dry_run=dry_run)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    stash_db.update_database(path)
    with StateStore(path, 'scope-a', dry_run=dry_run) as state:
        assert state.records('1', 'scene')['wd']


def test_update_backfills_missing_and_corrupt_metadata_in_every_scope(tmp_path):
    path = tmp_path / 'state.sqlite3'
    seed_version(path, LATEST_SCHEMA_VERSION)
    evidence = rows(path, 'results')
    with sqlite3.connect(path) as db:
        db.execute("DELETE FROM result_metadata WHERE scope='scope-a'")
        db.execute("UPDATE result_metadata SET data='corrupt' WHERE scope='scope-b'")
    stash_db.update_database(path)
    assert rows(path, 'results') == evidence
    assert len(rows(path, 'result_metadata')) == 2
    for scope in ('scope-a', 'scope-b'):
        with StateStore(path, scope) as state:
            state.unpack = lambda *_: pytest.fail('runtime repeated historical backfill')
            assert state.result_metadata('1', 'scene', 'wd')[0]['populated']


@pytest.mark.parametrize('table_threshold', [None, .7])
def test_v9_migration_preserves_mixed_history_and_frozen_proof_identity(tmp_path, table_threshold):
    from stash_ai.evidence.identity import marker_segment_signature
    from stash_ai.storage.codec import decode
    from tests.test_result_thresholds import outputs
    path = tmp_path / 'state.sqlite3'
    old = [outputs()['markers'], outputs(.6)['markers']]
    old[0]['marker_threshold_override'] = .8
    old[0]['refinement'] = {'retired': True}
    expected = [{k: v for k, v in segment.items() if k != 'refinement'} for segment in old]
    if table_threshold is not None:
        for segment in expected:
            segment['marker_threshold_override'] = table_threshold
    signatures = [marker_segment_signature(segment) for segment in expected]
    with StateStore(path, 'scope-a') as state:
        state.prepare('1', {'markers': {'contributions': [], 'frozen_signatures': signatures}})
        with state.connection as db:
            db.execute('INSERT INTO results VALUES (?,?,?,?,?)', ('scope-a', '1', 'wd', 'markers', encode(old)))
            db.execute('DROP TABLE result_thresholds')
            db.execute('CREATE TABLE result_thresholds(scope TEXT,scene TEXT,model TEXT,kind TEXT,threshold REAL,PRIMARY KEY(scope,scene,model,kind))')
            if table_threshold is not None:
                db.execute('INSERT INTO result_thresholds VALUES (?,?,?,?,?)', ('scope-a', '1', 'wd', 'markers', table_threshold))
            db.execute('PRAGMA user_version=9')
    pending = rows(path, 'snapshots')
    stash_db.update_database(path)
    with StateStore(path, 'scope-a') as state:
        effective = state.model_records('1', 'markers', 'wd')
        assert effective == expected
        assert [marker_segment_signature(segment) for segment in effective] == signatures
        raw = decode(state.connection.execute('SELECT data FROM results').fetchone()[0])
        assert all('marker_threshold_override' not in segment and 'refinement' not in segment for segment in raw)
        assert state.connection.execute('SELECT thresholds FROM result_thresholds').fetchone() == (('[0.8,null]' if table_threshold is None else '[0.7,0.7]'),)
    assert rows(path, 'snapshots') == pending
    evidence = rows(path, 'results')
    stash_db.update_database(path)
    assert rows(path, 'results') == evidence


@pytest.mark.parametrize('version', range(LATEST_SCHEMA_VERSION + 1))
def test_update_migrates_every_supported_version_and_all_scopes(tmp_path, version):
    path = tmp_path / 'state.sqlite3'
    seed_version(path, version)
    evidence = rows(path, 'results')

    assert stash_db.update_database(path) == (version, LATEST_SCHEMA_VERSION)

    assert rows(path, 'results') == evidence
    with sqlite3.connect(path) as db:
        assert db.execute('PRAGMA user_version').fetchone() == (LATEST_SCHEMA_VERSION,)
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert stash_db.STATE_TABLES <= tables
        assert 'refinement_progress' not in tables
        assert db.execute('SELECT scope,inference,populated,signature FROM result_index ORDER BY scope').fetchall() == [
            ('scope-a', 1, 1, 'saved'), ('scope-b', 1, 1, 'saved')]
    assert stash_db.check_database(path, 'scope-a')['ok']
    assert stash_db.update_database(path) == (LATEST_SCHEMA_VERSION, LATEST_SCHEMA_VERSION)
    assert rows(path, 'results') == evidence


@pytest.mark.parametrize('version', [0, 1, 3, 5, LATEST_SCHEMA_VERSION])
def test_dry_run_only_reports_target_without_modifying_database_or_lock(tmp_path, version):
    path = tmp_path / 'state.sqlite3'
    seed_version(path, version)
    lock = path.with_name(path.name + '.lock')
    lock.unlink(missing_ok=True)
    before = path.read_bytes(), path.stat().st_mtime_ns

    assert stash_db.update_database(path, dry_run=True) == (version, LATEST_SCHEMA_VERSION)

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
                       ('scope-a', '1', 'scene', 'old-operation', encode({'contributions': []})))
            db.execute('INSERT INTO checkpoints VALUES (?,?,?,?,?,?,?,?)',
                       ('scope-a', '2', 'wd', 'partial', 1, 2, 1, zlib.compress(b'\0' * 4)))
            db.execute('INSERT INTO checkpoint_frames VALUES (?,?,?,?,?)',
                       ('scope-a', '2', 'wd', 1, encode([])))
            db.execute('INSERT INTO checkpoint_analysis VALUES (?,?,?,?)',
                       ('scope-a', '2', 'wd', encode({})))
            db.execute('INSERT INTO refresh_requests VALUES (?,?,?,?)', ('scope-a', '2', 'wd', 'scene'))
            db.execute('INSERT INTO result_thresholds VALUES (?,?,?,?,?)', ('scope-a', '1', 'wd', 'scene', '[0.8]'))
            db.execute('INSERT INTO legacy_imports VALUES (?,?,?,?)', ('scope-a', '1', 'scene', encode([])))
    with sqlite3.connect(path) as connection:
        connection.execute('PRAGMA user_version=8')
    preserved = ('results', 'snapshots', 'cleanup_state', 'recovery_cleanups', 'recovery_publications',
                 'checkpoints', 'checkpoint_frames', 'checkpoint_analysis', 'refresh_requests',
                 'result_thresholds', 'result_metadata', 'legacy_imports')
    before = {table: rows(path, table) for table in preserved}

    stash_db.update_database(path)

    assert {table: rows(path, table) for table in preserved} == before


def test_update_refuses_an_active_writer_and_releases_failed_lock(tmp_path):
    path = tmp_path / 'state.sqlite3'
    seed_version(path, LATEST_SCHEMA_VERSION)
    with StateStore(path, 'scope-a'):
        before = path.read_bytes()
        with pytest.raises(RuntimeError, match='Another merged-output process'):
            stash_db.update_database(path)
        assert path.read_bytes() == before
    assert stash_db.update_database(path) == (LATEST_SCHEMA_VERSION, LATEST_SCHEMA_VERSION)


def test_failed_migration_can_be_retried_and_does_not_claim_latest_version(tmp_path, monkeypatch):
    path = tmp_path / 'state.sqlite3'
    seed_version(path, 1)
    evidence = rows(path, 'results')
    with monkeypatch.context() as patch:
        def fail(_):
            raise RuntimeError('interrupted migration')
        patch.setattr(performance_merge_checks, 'initialize', fail)
        with pytest.raises(RuntimeError, match='interrupted migration'):
            stash_db.update_database(path)
    with sqlite3.connect(path) as db:
        assert db.execute('PRAGMA user_version').fetchone()[0] < LATEST_SCHEMA_VERSION
    assert rows(path, 'results') == evidence

    stash_db.update_database(path)

    assert rows(path, 'results') == evidence
    assert stash_db.check_database(path, 'scope-a')['ok']


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
        stash_db.update_database(path, dry_run=dry_run)

    assert ((path.read_bytes(), path.stat().st_mtime_ns) if path.exists() else None) == before
    assert not path.with_name(path.name + '.lock').exists()


def test_db_update_cli_stays_offline_and_reports_dry_run_and_idempotence(tmp_path, monkeypatch, capsys):
    path = tmp_path / 'state.sqlite3'
    seed_version(path, 1)
    for name in ('load_model_settings', 'create_stash_client', 'validate_config', 'load_tagger'):
        monkeypatch.setattr(app.services, name, lambda *a, **kw: pytest.fail('db-update must stay offline'))
    monkeypatch.setattr(app, 'STASH_URL', 'unused-invalid-url')
    command = ['--state-db', str(path), '--db-update']
    before = path.read_bytes()

    assert app.main([*command, '--dry-run']) == 0
    assert path.read_bytes() == before
    assert 'DRY RUN: state DB schema v1 -> v10' in capsys.readouterr().out
    assert app.main(command) == 0
    assert 'State DB updated: schema v1 -> v10' in capsys.readouterr().out
    assert app.main(command) == 0
    assert 'up to date: v10' in capsys.readouterr().out
    assert app.main(['--state-db', str(tmp_path / 'missing.sqlite3'), '--db-update']) == 1
    assert 'does not exist' in capsys.readouterr().out


def test_db_update_cli_reports_unreadable_legacy_evidence_without_claiming_success(tmp_path, capsys):
    path = tmp_path / 'state.sqlite3'
    seed_version(path, 1)
    with sqlite3.connect(path) as db:
        db.execute('UPDATE results SET data=?', (b'not-compressed-json',))
    evidence = rows(path, 'results')

    assert app.main(['--state-db', str(path), '--db-update']) == 1

    assert 'Saved state could not be migrated' in capsys.readouterr().out
    assert rows(path, 'results') == evidence
    with sqlite3.connect(path) as db:
        assert db.execute('PRAGMA user_version').fetchone() == (1,)


@pytest.mark.parametrize('extra', [
    ['--db-check'], ['--db-compact'], ['--merge-only'], ['--models', 'wd'],
    ['--output-mode', 'separate'], ['--scene-id', '1'], ['--abandon-pending'],
])
def test_db_update_cli_rejects_unrelated_modes_and_modifiers(extra):
    with pytest.raises(SystemExit) as error:
        arguments.parse_args(app, ['--db-update', *extra])
    assert error.value.code == 2
