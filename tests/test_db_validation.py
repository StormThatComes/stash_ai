"""Extracted DB validators share one read snapshot and complete progress accounting."""
from io import StringIO
import sqlite3

import pytest
from tqdm import tqdm

from stash_ai import db as stash_db
from stash_ai.db.validation import results
from tests.test_db import seed


def test_checks_share_snapshot_when_another_writer_commits(tmp_path, monkeypatch):
    path = tmp_path / 'state.sqlite3'
    seed(path, 'scope')
    with sqlite3.connect(path) as writer:
        assert writer.execute('PRAGMA journal_mode=WAL').fetchone() == ('wal',)
    original = results.check_results_and_index

    def concurrent_update(connection, tables, errors, progress=None):
        assert connection.in_transaction
        with pytest.raises(sqlite3.OperationalError, match='readonly'):
            connection.execute("UPDATE result_index SET signature='forbidden'")
        with sqlite3.connect(path) as writer:
            writer.execute("UPDATE result_index SET signature='changed' WHERE scope='scope'")
        return original(connection, tables, errors, progress)

    monkeypatch.setattr(results, 'check_results_and_index', concurrent_update)
    report = stash_db.check_database(path, 'scope')
    assert report['ok'] and report['results_checked'] == 2
    monkeypatch.setattr(results, 'check_results_and_index', original)
    later = stash_db.check_database(path, 'scope')
    assert {issue['code']: issue['count'] for issue in later['errors']} == {'result_index_mismatch': 2}


def test_db_progress_counts_all_work_and_closes(tmp_path, monkeypatch):
    path = tmp_path / 'state.sqlite3'
    seed(path, 'scope')
    bars = []

    class Progress(tqdm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, file=StringIO())
            bars.append(self)

    monkeypatch.setattr('tqdm.tqdm', Progress)
    report = stash_db.check_database(path, 'scope', show_progress=True)
    assert report['ok']
    assert len(bars) == 1
    assert bars[0].n == bars[0].total == 8
    assert bars[0].disable
