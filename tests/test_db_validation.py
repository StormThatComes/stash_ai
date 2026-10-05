"""Extracted DB validators share one read snapshot and complete progress accounting."""
from io import StringIO
import sqlite3

import pytest
from tqdm import tqdm

from stash_ai.db.checks import check_database, format_check_report
from stash_ai.db.validation import results
from tests.test_db import seed


def test_historical_schema_reports_upgrade_without_decoding(tmp_path, monkeypatch):
    path = tmp_path / 'state.sqlite3'
    seed(path, 'scope')
    with sqlite3.connect(path) as db:
        db.execute('PRAGMA user_version=18')
        db.execute("UPDATE results SET data=x'01'")
    before = path.read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError('Historical payloads require migration first.')
    monkeypatch.setattr(results, 'check_results_and_index', forbidden)
    report = check_database(path, 'scope')
    assert not report['ok'] and report['integrity_check_ok']
    assert report['results_checked'] == 0
    assert [issue['code'] for issue in report['errors']] == ['schema_upgrade_required']
    assert '--db-update' in format_check_report(report)
    assert path.read_bytes() == before


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
    report = check_database(path, 'scope')
    assert report['ok'] and report['results_checked'] == 2
    monkeypatch.setattr(results, 'check_results_and_index', original)
    later = check_database(path, 'scope')
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
    report = check_database(path, 'scope', show_progress=True)
    assert report['ok']
    assert len(bars) == 1
    assert bars[0].n == bars[0].total == 8
    assert bars[0].disable



def test_existing_store_open_reads_state_without_schema_ddl(tmp_path, monkeypatch):
    from stash_ai.storage.store import StateStore
    from stash_ai.storage import database
    path = tmp_path / 'state.sqlite3'
    seed(path, 'scope')
    before = path.read_bytes()
    statements = []
    original = sqlite3.connect
    def traced(*args, **kwargs):
        db = original(*args, **kwargs)
        db.set_trace_callback(statements.append)
        return db
    monkeypatch.setattr(database.sqlite3, 'connect', traced)
    with StateStore(path, 'scope') as store:
        assert store.records('1', 'scene')
    assert not any(sql.lstrip().upper().startswith(('CREATE ', 'ALTER ', 'DROP ')) for sql in statements)
    assert path.read_bytes() == before


@pytest.mark.parametrize('damage,code', [
    ('DROP TRIGGER results_metadata_update', 'missing_state_triggers'),
    ('DROP INDEX snapshots_pending_scope', 'missing_state_indexes'),
    ('ALTER TABLE result_index ADD COLUMN surprise TEXT', 'invalid_table_columns'),
])
def test_current_schema_shape_preflight_is_read_only(tmp_path, damage, code):
    from stash_ai.storage.store import StateStore
    path = tmp_path / 'state.sqlite3'
    seed(path, 'scope')
    with sqlite3.connect(path) as db:
        db.execute(damage)
    before = path.read_bytes()
    report = check_database(path, 'scope')
    assert not report['ok'] and report['results_checked'] == 0
    assert [issue['code'] for issue in report['errors']] == [code]
    with pytest.raises(ValueError, match='Incomplete state DB'):
        StateStore(path, 'scope')
    assert path.read_bytes() == before
