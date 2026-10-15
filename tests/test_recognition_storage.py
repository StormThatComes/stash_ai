import json
import os
import sqlite3

import pytest

from stash_ai.recognition.database import FaceDatabase, APPLICATION_ID, SCHEMA_V1, SCHEMA_V2, SCHEMA_VERSION


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


@pytest.mark.parametrize('old_scope,new_scope,old_selection,new_selection',[
    ('scope','scope',['1'],['2']),('scope','scope',['1','2'],['2','1']),
    ('scope','scope',['1'],['1']),('old-instance','new-instance',['1'],['2']),
])
def test_new_writer_recovers_only_prior_owner_runs_independent_of_selection(tmp_path,old_scope,new_scope,old_selection,new_selection):
    path=tmp_path/'faces.sqlite3'
    with FaceDatabase(path) as db:
        orphan=db.start_run(old_scope,old_selection,{})
        db.progress(orphan,'scan',{'completed':2})
        db.connection.execute('UPDATE runs SET summary=? WHERE id=?', ('{"counts":{"retained":2}}',orphan))
        completed=db.start_run(old_scope,['complete'],{})
        db.finish_run(completed,'completed',{'done':True})
        db.connection.execute('INSERT INTO publications VALUES (?,?,?,?,?,?,?,?,?,?)',
            ('unknown','scope','1','person','image','42','{}','{}','unknown','lost response'))
        db.connection.commit()
        summary=db.connection.execute('SELECT summary FROM runs WHERE id=?',(orphan,)).fetchone()[0]
        progress=[tuple(row) for row in db.connection.execute('SELECT * FROM progress')]
        pending=[tuple(row) for row in db.connection.execute('SELECT * FROM publications')]
    with FaceDatabase(path) as db:
        current=db.start_run(new_scope,new_selection,{})
        previous=db.connection.execute('SELECT * FROM runs WHERE id=?',(orphan,)).fetchone()
        assert previous['status']=='interrupted' and previous['finished'] is not None
        assert previous['summary']==summary
        assert [tuple(row) for row in db.connection.execute('SELECT * FROM progress')]==progress
        assert [tuple(row) for row in db.connection.execute('SELECT * FROM publications')]==pending
        assert db.connection.execute('SELECT status FROM runs WHERE id=?',(completed,)).fetchone()[0]=='completed'
        assert db.connection.execute('SELECT status FROM runs WHERE id=?',(current,)).fetchone()[0]=='running'
        with pytest.raises(RuntimeError,match='recognition writer'): FaceDatabase(path)
        assert db.connection.execute('SELECT status FROM runs WHERE id=?',(current,)).fetchone()[0]=='running'


def test_v2_migration_retains_people_and_adds_persistent_merge_support(tmp_path):
    path = tmp_path / 'faces.sqlite3'
    with sqlite3.connect(path) as db:
        db.executescript(SCHEMA_V2)
        db.execute(f'PRAGMA application_id={APPLICATION_ID}')
        db.execute('PRAGMA user_version=2')
        db.execute('INSERT INTO metadata VALUES (?,?)', ('installation', '01234567-89ab-cdef-0123-456789abcdef'))
        db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)', ('oldest', 'scope', 'ns', 123., '77', 'token'))
    for _ in range(2):
        with FaceDatabase(path) as db:
            assert db.connection.execute('PRAGMA user_version').fetchone()[0] == SCHEMA_VERSION
            assert db.connection.execute('SELECT performer FROM people WHERE id=?', ('oldest',)).fetchone()[0] == '77'
            assert db.connection.execute('SELECT * FROM merge_support').fetchall() == []


def test_failed_v2_merge_support_migration_is_atomic(tmp_path, monkeypatch):
    import stash_ai.recognition.database as module
    path = tmp_path / 'faces.sqlite3'
    with sqlite3.connect(path) as db:
        db.executescript(SCHEMA_V2)
        db.execute(f'PRAGMA application_id={APPLICATION_ID}')
        db.execute('PRAGMA user_version=2')
    original = module.SUPPORT_SCHEMA
    monkeypatch.setattr(module, 'SUPPORT_SCHEMA', original + '\nINVALID SQL;')
    with pytest.raises(sqlite3.Error):
        FaceDatabase(path)
    with sqlite3.connect(path) as db:
        assert db.execute('PRAGMA user_version').fetchone()[0] == 2
        assert db.execute("SELECT name FROM sqlite_master WHERE name='merge_support'").fetchone() is None


import json
import sqlite3
import uuid
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest


def test_face_schema_rejects_missing_foreign_key_and_unique_token(tmp_path):
    from stash_ai.recognition.database import FaceDatabase, SCHEMA, APPLICATION_ID, SCHEMA_VERSION
    path = tmp_path / 'faces.sqlite3'
    schema = SCHEMA.replace(' REFERENCES people(id)','').replace('token TEXT NOT NULL UNIQUE','token TEXT NOT NULL')
    with sqlite3.connect(path) as db:
        db.executescript(schema)
        db.execute('INSERT INTO metadata VALUES (?,?)',('installation',str(uuid.uuid4())))
        db.execute(f'PRAGMA application_id={APPLICATION_ID}')
        db.execute(f'PRAGMA user_version={SCHEMA_VERSION}')
    before = path.read_bytes()
    with pytest.raises(ValueError, match='constraints'):
        FaceDatabase(path)
    assert path.read_bytes() == before


import json
import sqlite3
import uuid
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest


def test_crashed_face_backup_leaves_final_destination_available(tmp_path):
    import os
    import subprocess
    import sys
    from stash_ai.recognition.database import FaceDatabase
    source,destination = tmp_path/'faces.sqlite3',tmp_path/'backup.sqlite3'
    with FaceDatabase(source):pass
    script = '''import os,sys
from stash_ai.recognition.database import FaceDatabase
with FaceDatabase(sys.argv[1]) as db:
    class Crash:
        def backup(self,target): os._exit(17)
    db.connection.close()
    db.connection=Crash()
    db.backup(sys.argv[2])
'''
    process = subprocess.run([sys.executable,'-c',script,str(source),str(destination)],capture_output=True,timeout=10)
    assert process.returncode == 17
    assert not destination.exists()
    with FaceDatabase(source) as db:
        db.backup(destination)
    with FaceDatabase(destination) as restored:
        assert restored.connection.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'



import json
import os
from types import SimpleNamespace

import numpy as np
import pytest


def test_face_database_close_preserves_primary_interruption(tmp_path):
    from stash_ai.recognition.database import FaceDatabase
    database = FaceDatabase(tmp_path / 'faces.sqlite3')
    connection = database.connection
    class FailedClose:
        def close(self):
            connection.close()
            raise OSError('synthetic connection close failure')
    database.connection = FailedClose()
    with pytest.raises(KeyboardInterrupt, match='primary recognition interruption') as caught:
        with database:
            raise KeyboardInterrupt('primary recognition interruption')
    assert database.lock is None
    assert any('synthetic connection close failure' in note for note in caught.value.__notes__)
    database.close()



@pytest.mark.parametrize('change', ['foreign_key', 'token', 'scan_unique', 'partial_unique'])
def test_required_schema_constraints_are_independently_validated(tmp_path, change):
    import uuid
    from stash_ai.recognition.database import SCHEMA
    schema = SCHEMA
    if change == 'foreign_key':
        schema = schema.replace(' REFERENCES observations(id)', '')
    elif change in ('token', 'partial_unique'):
        schema = schema.replace('token TEXT NOT NULL UNIQUE', 'token TEXT NOT NULL')
        if change == 'partial_unique':
            schema += "CREATE UNIQUE INDEX partial_token ON people(token) WHERE token<>'';"
    else:
        schema = schema.replace('UNIQUE(scope,scene,namespace,input_sha,policy_sha)', 'CHECK(total>=0)')
    path = tmp_path / 'invalid.sqlite3'
    with sqlite3.connect(path) as db:
        db.executescript(schema)
        db.execute('INSERT INTO metadata VALUES (?,?)', ('installation', str(uuid.uuid4())))
        db.execute(f'PRAGMA application_id={APPLICATION_ID}')
        db.execute(f'PRAGMA user_version={SCHEMA_VERSION}')
    before = path.read_bytes()
    with pytest.raises(ValueError, match='constraints'):
        FaceDatabase(path)
    assert path.read_bytes() == before


def test_supported_constraints_reject_duplicate_tokens_and_orphans(tmp_path):
    with FaceDatabase(tmp_path / 'state.sqlite3') as state:
        state.connection.execute('INSERT INTO people VALUES (?,?,?,?,?,?)', ('one', 'scope', 'ns', 1., None, 'token'))
        with pytest.raises(sqlite3.IntegrityError):
            state.connection.execute('INSERT INTO people VALUES (?,?,?,?,?,?)', ('two', 'scope', 'ns', 1., None, 'token'))
        with pytest.raises(sqlite3.IntegrityError):
            state.connection.execute('INSERT INTO identity_merges VALUES (?,?,?,?,?)', ('one', 'missing', 1., '{}', 'pending'))


def test_backup_preserves_racing_destination_and_cleans_stage(tmp_path, monkeypatch):
    import stash_ai.recognition.database as module
    path, destination = tmp_path / 'state.sqlite3', tmp_path / 'backup.sqlite3'
    link = module.os.link
    def race(stage, target):
        target.write_bytes(b'other writer')
        link(stage, target)
    with FaceDatabase(path) as state:
        before = path.read_bytes()
        monkeypatch.setattr(module.os, 'link', race)
        with pytest.raises(FileExistsError):
            state.backup(destination)
        assert destination.read_bytes() == b'other writer'
        assert path.read_bytes() == before
        assert not list(tmp_path.glob('.backup.sqlite3-*.tmp'))


@pytest.mark.parametrize('primary', [False, True])
def test_database_connection_and_lock_close_are_exhaustive(tmp_path, primary):
    state = FaceDatabase(tmp_path / 'state.sqlite3')
    connection, lock = state.connection, state.lock
    calls = []
    class Connection:
        def close(self):
            calls.append('connection')
            connection.close()
            raise OSError('connection failed')
    class Lock:
        def close(self):
            calls.append('lock')
            lock.close()
            raise RuntimeError('lock failed')
    state.connection, state.lock = Connection(), Lock()
    if primary:
        with pytest.raises(KeyboardInterrupt, match='primary') as caught:
            with state:
                raise KeyboardInterrupt('primary')
    else:
        with pytest.raises(OSError, match='connection failed') as caught:
            state.close()
    assert calls == ['connection', 'lock']
    assert any('lock failed' in note for note in caught.value.__notes__)
    state.close()



def test_backup_sync_handle_supports_windows_commit_requirement(tmp_path, monkeypatch):
    import stash_ai.recognition.database as module
    original = module.os.fsync
    synced = []
    def sync(descriptor):
        # Windows _commit rejects a read-only file descriptor. Exercise the
        # same required capability with a real syscall on every test platform.
        os.write(descriptor, b'')
        synced.append(descriptor)
        return original(descriptor)
    monkeypatch.setattr(module.os, 'fsync', sync)
    destination = tmp_path / 'backup.sqlite3'
    with FaceDatabase(tmp_path / 'state.sqlite3') as state:
        installation = state.installation
        state.backup(destination)
    assert len(synced) == 1
    with FaceDatabase(destination) as state:
        assert state.installation == installation
        assert state.connection.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
