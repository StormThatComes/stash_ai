"""Diagnostics close SQLite handles and tolerate missing initialization tables."""
import sqlite3

import pytest

from stash_ai.db.inspection import inspect_database
from stash_ai.db.compaction import compact_database
from stash_ai.reports.developer.report import build_report
from stash_ai.storage.store import StateStore
from stash_ai.storage.writer_lock import writer_lock


def test_report_describes_missing_init_table_instead_of_joining_it(tmp_path):
    path = tmp_path / 'state.db'
    with StateStore(path, 'scope') as state:
        state.connection.execute('DROP TABLE db_init_scenes')
    report = inspect_database(path, 'scope')
    assert report['quick_check'] == 'ok'
    assert 'db_init_scenes' in report['missing_tables']
    assert report['schema_issues']['missing_state_tables'] == 1
    assert report['db_init_completed_scenes'] == 0
    with pytest.raises(RuntimeError, match='Incomplete state DB schema'):
        compact_database(path, 'scope')


@pytest.mark.parametrize('operation', ['inspection', 'developer', 'compaction'])
def test_diagnostics_and_vacuum_explicitly_close_connections(tmp_path, monkeypatch, operation):
    path = tmp_path / 'state.db'
    with StateStore(path, 'scope'): pass
    original, opened = sqlite3.connect, []
    class Connection(sqlite3.Connection):
        pass
    def connect(*args, **kwargs):
        value = original(*args, **kwargs, factory=Connection)
        opened.append(value)
        return value
    monkeypatch.setattr(sqlite3, 'connect', connect)
    {'inspection': inspect_database, 'developer': lambda p, _: build_report(p),
     'compaction': compact_database}[operation](path, 'scope')
    assert opened
    for connection in opened:
        with pytest.raises(sqlite3.ProgrammingError, match='closed database'):
            connection.execute('SELECT 1')


def test_shared_lock_excludes_state_writer_and_releases_after_error(tmp_path):
    path = tmp_path / 'state.db'
    with StateStore(path, 'scope'): pass
    with pytest.raises(ValueError, match='primary'):
        with writer_lock(path):
            with pytest.raises(RuntimeError, match='Another merged-output process'):
                StateStore(path, 'scope')
            raise ValueError('primary')
    with StateStore(path, 'scope'): pass
