import json
import os
import sqlite3

import pytest

from stash_ai.recognition.database import FaceDatabase, APPLICATION_ID, SCHEMA_V1, SCHEMA_VERSION


def test_one_database_owns_history_progress_and_backup(tmp_path):
    path = tmp_path / 'faces.sqlite3'
    with FaceDatabase(path) as state:
        identifier = state.start_run('scope', ['6306'], {'models': 'test'})
        state.progress(identifier, 'scan', {'completed': 3})
        state.finish_run(identifier, 'success', {'faces': 2})
        state.backup(tmp_path / 'backup.sqlite3')
        with pytest.raises(FileExistsError):
            state.backup(tmp_path / 'backup.sqlite3')
        owner = state.installation
    with FaceDatabase(tmp_path / 'backup.sqlite3') as state:
        assert state.installation == owner
        assert json.loads(state.connection.execute('SELECT selection FROM runs').fetchone()[0]) == ['6306']
        assert state.connection.execute('SELECT phase FROM progress').fetchone()[0] == 'scan'
    assert not list(tmp_path.glob('*.runs.sqlite3'))


def test_refuses_tag_database_without_modification_or_lock(tmp_path):
    path = tmp_path / 'tag.sqlite3'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE results(x)')
        db.execute('INSERT INTO results VALUES (123)')
    before = path.read_bytes()
    with pytest.raises(ValueError, match='Not a facial'):
        FaceDatabase(path)
    assert path.read_bytes() == before
    assert not (tmp_path / 'tag.sqlite3.lock').exists()


def test_refuses_damaged_shape_and_foreign_empty_marker(tmp_path):
    path = tmp_path / 'faces.sqlite3'
    with FaceDatabase(path):
        pass
    with sqlite3.connect(path) as db:
        db.execute('DROP TABLE portraits')
    with pytest.raises(ValueError, match='incomplete'):
        FaceDatabase(path)
    other = tmp_path / 'foreign.sqlite3'
    with sqlite3.connect(other) as db:
        db.execute('PRAGMA application_id=123')
    with pytest.raises(ValueError):
        FaceDatabase(other)


def test_exclusive_writer_and_hardlink_rejection(tmp_path):
    path = tmp_path / 'faces.sqlite3'
    with FaceDatabase(path):
        with pytest.raises(RuntimeError, match='recognition writer'):
            FaceDatabase(path)
    alias = tmp_path / 'alias.sqlite3'
    try:
        os.link(path, alias)
    except OSError:
        pytest.skip('hard links unavailable')
    with pytest.raises(ValueError, match='hard-linked'):
        FaceDatabase(alias)


def test_schema_failure_rolls_back_versions_and_releases_writer(tmp_path, monkeypatch):
    import stash_ai.recognition.database as module
    schema = module.SCHEMA
    monkeypatch.setattr(module, 'SCHEMA', schema + '\nNOT VALID SQL;')
    path = tmp_path / 'faces.sqlite3'
    with pytest.raises(sqlite3.Error):
        FaceDatabase(path)
    with sqlite3.connect(path) as db:
        assert db.execute('PRAGMA application_id').fetchone()[0] == 0
        assert not db.execute('SELECT name FROM sqlite_master').fetchall()
    monkeypatch.setattr(module, 'SCHEMA', schema)
    with FaceDatabase(path) as state:
        assert state.connection.execute('PRAGMA application_id').fetchone()[0] == APPLICATION_ID


def test_start_run_does_not_rewrite_unrelated_run_history(tmp_path):
    with FaceDatabase(tmp_path / 'faces.sqlite3') as state:
        other = state.start_run('scope', ['other'], {})
        state.start_run('scope', ['selected'], {})
        assert state.connection.execute('SELECT status FROM runs WHERE id=?', (other,)).fetchone()[0] == 'running'


def legacy(path):
    with sqlite3.connect(path) as db:
        db.executescript(SCHEMA_V1)
        db.execute(f'PRAGMA application_id={APPLICATION_ID}')
        db.execute('PRAGMA user_version=1')
        db.execute('INSERT INTO metadata VALUES (?,?)', ('installation', '01234567-89ab-cdef-0123-456789abcdef'))
        db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)', ('oldest', 'scope', 'namespace', 123., '77', 'old-token'))


def test_v1_migration_preserves_identity_bindings_and_is_repeatable(tmp_path):
    path = tmp_path / 'faces.sqlite3'
    legacy(path)
    for _ in range(2):
        with FaceDatabase(path) as state:
            assert state.connection.execute('PRAGMA user_version').fetchone()[0] == SCHEMA_VERSION
            assert tuple(state.connection.execute('SELECT * FROM people').fetchone()) == (
                'oldest', 'scope', 'namespace', 123., '77', 'old-token')
            assert not state.connection.execute('SELECT * FROM identity_merges').fetchall()
            assert state.installation == '01234567-89ab-cdef-0123-456789abcdef'


def test_corrupt_v1_is_not_partially_repaired_by_migration(tmp_path):
    path = tmp_path / 'faces.sqlite3'
    legacy(path)
    with sqlite3.connect(path) as db:
        db.execute('DROP TABLE portraits')
    with pytest.raises(ValueError, match='incomplete'):
        FaceDatabase(path)
    with sqlite3.connect(path) as db:
        assert db.execute('PRAGMA user_version').fetchone()[0] == 1
        assert not db.execute("SELECT name FROM sqlite_master WHERE name='identity_merges'").fetchone()


def test_failed_v1_migration_rolls_back_ddl_version_and_releases_writer(tmp_path, monkeypatch):
    import stash_ai.recognition.database as module
    path = tmp_path / 'faces.sqlite3'
    legacy(path)
    schema = module.MERGE_SCHEMA
    monkeypatch.setattr(module, 'MERGE_SCHEMA', schema + '\nNOT VALID SQL;')
    with pytest.raises(sqlite3.Error):
        FaceDatabase(path)
    with sqlite3.connect(path) as db:
        assert db.execute('PRAGMA user_version').fetchone()[0] == 1
        assert not db.execute("SELECT name FROM sqlite_master WHERE name='identity_merges'").fetchone()
    monkeypatch.setattr(module, 'MERGE_SCHEMA', schema)
    with FaceDatabase(path) as state:
        assert state.connection.execute('PRAGMA user_version').fetchone()[0] == SCHEMA_VERSION
