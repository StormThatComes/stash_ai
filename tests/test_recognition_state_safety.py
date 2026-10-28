"""Storage boundaries must fail before changing history or abandoning ownership."""
import os
import sqlite3
import threading
import uuid

import pytest

from stash_ai.recognition import database as module
from stash_ai.recognition.database import FaceDatabase


@pytest.mark.parametrize('extra', [
    'CREATE TRIGGER unrecognized AFTER UPDATE ON runs BEGIN DELETE FROM people; END;',
    'CREATE VIEW unrecognized AS SELECT * FROM people;',
])
def test_unknown_executable_schema_is_refused_before_recovery(tmp_path, extra):
    path = tmp_path / 'state.sqlite3'
    with FaceDatabase(path) as state:
        state.connection.execute('INSERT INTO people VALUES (?,?,?,?,?,?)', ('p', 'scope', 'ns', 1., None, 'token'))
        state.connection.commit()
        state.start_run('scope', ['1'], {})
    with sqlite3.connect(path) as db:
        db.executescript(extra)
    before = path.read_bytes()
    with pytest.raises(ValueError, match='triggers or views'):
        FaceDatabase(path)
    assert path.read_bytes() == before
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT id FROM people').fetchone()[0] == 'p'
        assert db.execute('SELECT status FROM runs').fetchone()[0] == 'running'


def create_schema(path, schema, version=module.SCHEMA_VERSION, installation=None):
    with sqlite3.connect(path) as db:
        db.executescript(schema)
        db.execute(f'PRAGMA application_id={module.APPLICATION_ID}')
        db.execute(f'PRAGMA user_version={version}')
        if installation is not None:
            db.execute('INSERT INTO metadata VALUES (?,?)', ('installation', installation))
        db.execute("INSERT INTO runs(id,scope,selection,config,started,status) VALUES ('run','scope','[]','{}',1,'running')")


@pytest.mark.parametrize('old,new', [
    ("DEFAULT 'partial'", "DEFAULT 'complete'"),
    ('CHECK(source<>target)', 'CHECK(1)'),
    ("CHECK(status IN ('pending','confirmed','observed','local'))", 'CHECK(1)'),
    ('CHECK(source<>target)', '/* CHECK(source<>target) */ CHECK((1))'),
])
def test_defaults_and_each_check_constraint_are_required(tmp_path, old, new):
    path = tmp_path / 'state.sqlite3'
    create_schema(path, module.SCHEMA.replace(old, new), installation=str(uuid.uuid4()))
    before = path.read_bytes()
    with pytest.raises(ValueError, match='schema shape|CHECK constraints'):
        FaceDatabase(path)
    assert path.read_bytes() == before


def test_check_validation_accepts_sql_formatting_and_quoted_identifiers(tmp_path):
    path = tmp_path / 'state.sqlite3'
    schema = module.SCHEMA.replace('CHECK(source<>target)', 'check ( "source" /* comment */ <> [target] )')
    create_schema(path, schema, installation=str(uuid.uuid4()))
    with FaceDatabase(path) as state:
        assert state.connection.execute('SELECT status FROM runs').fetchone()[0] == 'interrupted'


@pytest.mark.parametrize('version,schema', [
    (1, module.SCHEMA_V1), (2, module.SCHEMA_V2),
    (3, module.SCHEMA_V3), (module.SCHEMA_VERSION, module.SCHEMA),
], ids=['v1','v2','v3','current'])
@pytest.mark.parametrize('installation', [None, 'invalid-identity'])
def test_invalid_installation_is_refused_before_recovery_or_migration(tmp_path, version, schema, installation):
    path = tmp_path / 'state.sqlite3'
    create_schema(path, schema, version, installation)
    before = path.read_bytes()
    with pytest.raises(ValueError, match='installation identity'):
        FaceDatabase(path)
    assert path.read_bytes() == before
    with sqlite3.connect(path) as db:
        assert db.execute('PRAGMA user_version').fetchone()[0] == version
        assert db.execute('SELECT status FROM runs').fetchone()[0] == 'running'


def test_wrong_thread_close_retains_live_connection_and_writer(tmp_path):
    state = FaceDatabase(tmp_path / 'state.sqlite3')
    try:
        run_id = state.start_run('scope', ['1'], {})
        raw, lock = state.connection, state.lock
        failures = []
        def close():
            try:
                state.close()
            except Exception as error:
                failures.append(error)
        worker = threading.Thread(target=close)
        worker.start()
        worker.join(timeout=2)
        assert not worker.is_alive()
        assert len(failures) == 1 and isinstance(failures[0], sqlite3.ProgrammingError)
        assert state.connection is raw and state.lock is lock
        with pytest.raises(RuntimeError, match='recognition writer'):
            FaceDatabase(state.path)
        assert raw.execute('SELECT status FROM runs WHERE id=?', (run_id,)).fetchone()[0] == 'running'
    finally:
        state.close()
    with FaceDatabase(state.path) as replacement:
        assert replacement.connection.execute('SELECT status FROM runs').fetchone()[0] == 'interrupted'


def test_failed_close_keeps_writer_until_sqlite_is_closed(tmp_path):
    state = FaceDatabase(tmp_path / 'state.sqlite3')
    raw, lock = state.connection, state.lock
    class FailedClose:
        def close(self):
            raise OSError('connection still live')
    state.connection = FailedClose()
    try:
        with pytest.raises(OSError, match='still live'):
            state.close()
        assert state.lock is lock
        assert raw.execute('SELECT 1').fetchone()[0] == 1
        with pytest.raises(RuntimeError, match='recognition writer'):
            FaceDatabase(state.path)
    finally:
        state.connection = raw
        state.close()


def test_backup_rejects_active_transaction_without_changing_it(tmp_path):
    destination = tmp_path / 'backup.sqlite3'
    with FaceDatabase(tmp_path / 'state.sqlite3') as state:
        state.connection.execute('INSERT INTO people VALUES (?,?,?,?,?,?)', ('p', 'scope', 'ns', 1., None, 'token'))
        assert state.connection.in_transaction
        with pytest.raises(ValueError, match='active recognition transaction'):
            state.backup(destination)
        assert state.connection.in_transaction
        assert state.connection.execute('SELECT id FROM people').fetchone()[0] == 'p'
        assert not destination.exists() and not list(tmp_path.glob('.recognition-backup-*.tmp'))
        state.connection.rollback()
        state.backup(destination)
    with FaceDatabase(destination) as restored:
        assert restored.connection.execute('SELECT COUNT(*) FROM people').fetchone()[0] == 0


def test_backup_descriptor_close_failure_removes_owned_stage(tmp_path, monkeypatch):
    with FaceDatabase(tmp_path / 'state.sqlite3') as state:
        close = module.os.close
        def fail(descriptor):
            close(descriptor)
            raise OSError('initial staging descriptor close failed')
        with monkeypatch.context() as patch:
            patch.setattr(module.os, 'close', fail)
            with pytest.raises(OSError, match='initial staging'):
                state.backup(tmp_path / 'copy.sqlite3')
        assert not list(tmp_path.glob('.recognition-backup-*.tmp'))
        assert not (tmp_path / 'copy.sqlite3').exists()
        state.backup(tmp_path / 'copy.sqlite3')


@pytest.mark.skipif(os.name != 'posix', reason='POSIX NAME_MAX boundary')
def test_backup_accepts_a_legal_long_destination_basename(tmp_path):
    maximum = os.pathconf(tmp_path, 'PC_NAME_MAX')
    if maximum <= 0:
        pytest.skip('Filesystem does not report a finite NAME_MAX')
    destination = tmp_path / ('b' * (maximum - 1))
    destination.write_bytes(b'probe')
    destination.unlink()
    with FaceDatabase(tmp_path / 'state.sqlite3') as state:
        installation = state.installation
        state.backup(destination)
    with sqlite3.connect(destination) as restored:
        assert restored.execute("SELECT value FROM metadata WHERE key='installation'").fetchone()[0] == installation
        assert restored.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    assert not list(tmp_path.glob('.recognition-backup-*.tmp'))
