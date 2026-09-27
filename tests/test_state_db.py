"""Resource ownership across state-store initialization failures and previews."""
import sqlite3

import pytest

from stash_ai import progress as stash_progress
from stash_ai import state_db as stash_state_db
from stash_ai.state import StateStore


@pytest.mark.parametrize('stage', ['base_schema', 'feature_schema'])
def test_failed_initialization_closes_connection_and_releases_writer(tmp_path, monkeypatch, stage):
    path = tmp_path / 'state.sqlite3'
    connections = []
    connect = sqlite3.connect

    def tracked_connect(*args, **kwargs):
        connection = connect(*args, **kwargs)
        connections.append(connection)
        return connection

    def fail(*args):
        raise RuntimeError('interrupted initialization')

    with monkeypatch.context() as patch:
        patch.setattr(stash_state_db.sqlite3, 'connect', tracked_connect)
        if stage == 'base_schema':
            patch.setattr(stash_state_db.StateDatabase, '_initialize_schema', fail)
        else:
            patch.setattr(stash_progress, 'initialize_progress', fail)
        with pytest.raises(RuntimeError, match='interrupted initialization'):
            StateStore(path, 'scope')

    assert len(connections) == 1
    with pytest.raises(sqlite3.ProgrammingError, match='closed'):
        connections[0].execute('SELECT 1')
    # Reopening verifies that failure on either side of the ownership boundary
    # released the advisory lock, not only the SQLite connection.
    with StateStore(path, 'scope') as state:
        assert state.records('1', 'scene') == {}


def test_preview_does_not_release_active_writer_or_change_source(tmp_path):
    path = tmp_path / 'state.sqlite3'
    original = {'mode': 'scene', 'meta': {'origin': 'inference'}, 'items': []}
    changed = {**original, 'items': [{'tag': 'example', 'score': .8}]}
    with StateStore(path, 'scope') as writer:
        writer.put('1', 'wd', {'scene': original})
        before = path.read_bytes(), path.stat().st_mtime_ns
        with StateStore(path, 'scope', dry_run=True) as preview:
            preview.put('1', 'wd', {'scene': changed})
            assert preview.records('1', 'scene')['wd'] == [changed]
            assert writer.records('1', 'scene')['wd'] == [original]
        assert (path.read_bytes(), path.stat().st_mtime_ns) == before
        with pytest.raises(RuntimeError, match='Another merged-output process'):
            StateStore(path, 'scope')
    with StateStore(path, 'scope') as reopened:
        assert reopened.records('1', 'scene')['wd'] == [original]


@pytest.mark.parametrize('dry_run', [False, True])
def test_unsupported_database_is_rejected_without_changing_source(tmp_path, dry_run):
    path = tmp_path / 'foreign.sqlite3'
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE unrelated (value TEXT)')
        connection.execute("INSERT INTO unrelated VALUES ('keep')")
    connection.close()
    before = path.read_bytes(), path.stat().st_mtime_ns
    with pytest.raises(ValueError, match='Not a supported stash-taggers state DB'):
        StateStore(path, 'scope', dry_run=dry_run)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_close_is_idempotent_and_releases_writer(tmp_path):
    path = tmp_path / 'state.sqlite3'
    state = StateStore(path, 'scope', cache_bytes=1024)
    connection = state.connection
    state.close()
    state.close()
    assert state.connection is None
    with pytest.raises(sqlite3.ProgrammingError, match='closed'):
        connection.execute('SELECT 1')
    with StateStore(path, 'scope') as reopened:
        assert reopened.connection.execute('SELECT 1').fetchone() == (1,)
