"""Real SQLite maintenance: sizing, vacuum, ownership, WAL and offline CLI."""
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from stash_ai.recognition import db_maintenance as maintenance
from stash_ai.recognition.cli import main, parser
from stash_ai.recognition.database import FaceDatabase, TABLES, SCHEMA_VERSION
from tests.test_recognition_storage import legacy


def seed(path):
    with FaceDatabase(path) as state:
        state.start_run('scope-a', ['1'], {'secret': 'not-in-report'})
        db = state.connection
        with db:
            db.execute("INSERT INTO scans(id,scope,scene,namespace,input_sha,policy_sha,manifest,coverage,total,status) "
                       "VALUES ('scan','scope-a','1','model','input','policy','{}','{}',1,'complete')")
            db.execute("INSERT INTO observations VALUES ('face','scan',0,0.,'vtt','[]','[]','{}',NULL,4,?,?)",
                       (b'face-vector', b'face-pixels' * 10000))
            db.execute("INSERT INTO people VALUES ('person','scope-a','model',1.,'77','token')")
            db.execute("INSERT INTO members VALUES ('face','person')")
            db.execute("INSERT INTO body_observations VALUES ('body','scan',0,0.,'contract','face','{}','{}',NULL,?,?)",
                       (b'\x00' * 16384, b'body-pixels' * 1000))
            db.execute("INSERT INTO publications VALUES ('publish','scope-b','9','other','create',NULL,'{}','{}','pending',NULL)")
            db.execute("INSERT INTO metadata VALUES ('temporary',?)", (b'\x00' * 2_000_000,))
        with db:
            db.execute("DELETE FROM metadata WHERE key='temporary'")


def logical_rows(path):
    with closing(sqlite3.connect(path)) as db:
        tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        # Row order matters to recognition's latest-scan/decision semantics.
        return {t: db.execute(f'SELECT * FROM "{t}" ORDER BY rowid').fetchall() for t in tables}


def test_status_accounts_for_tables_indexes_overflow_and_free_pages_without_writes(tmp_path):
    path = tmp_path / 'faces.db'
    seed(path)
    before = path.read_bytes()
    calls = []
    report = maintenance.inspect_database(path, notify=lambda *args: calls.append(args))
    assert path.read_bytes() == before
    assert report['quick_check'] == 'ok' and report['schema_version'] == SCHEMA_VERSION
    assert report['file_bytes'] == path.stat().st_size == report['total_disk_bytes']
    assert report['allocated_bytes'] == report['page_count'] * report['page_size']
    assert report['reclaimable_estimate_bytes'] == report['freelist_count'] * report['page_size'] > 1_000_000
    assert report['states']['runs'] == {'running': 1}  # no orphan recovery
    assert report['states']['publications'] == {'pending': 1}
    tables = {row['name']: row for row in report['tables']}
    assert set(tables) == set(TABLES)
    assert tables['observations']['rows'] == tables['body_observations']['rows'] == 1
    if report['dbstat_available']:
        assert tables['observations']['table_bytes'] > 100_000  # overflow BLOB pages
        assert tables['observations']['index_bytes'] > 0
        for row in tables.values():
            assert row['total_bytes'] == row['table_bytes'] + row['index_bytes']
        assert sum(row['total_bytes'] for row in tables.values()) == report['tables_total_bytes']
        assert (report['tables_total_bytes'] + report['non_table_bytes'] + report['free_page_bytes']
                == report['allocated_bytes'])
    text = maintenance.format_report(report)
    assert 'not-in-report' not in text and 'face-pixels' not in text
    assert 'Estimated VACUUM savings' in text and 'Total on disk' in text
    assert calls[-1] == ('Count table rows', len(TABLES), len(TABLES))


def test_vacuum_reclaims_space_preserves_every_row_and_does_not_recover_runs(tmp_path):
    path = tmp_path / 'faces.db'
    seed(path)
    expected = logical_rows(path)
    before, after = maintenance.compact_database(path)
    assert after['file_bytes'] < before['file_bytes'] - 1_000_000
    assert after['total_disk_bytes'] < before['total_disk_bytes']
    assert after['free_page_bytes'] < before['free_page_bytes']
    assert after['quick_check'] == 'ok' and after['states'] == before['states']
    assert logical_rows(path) == expected


def test_dry_run_is_read_only_without_writer_lock(tmp_path, monkeypatch):
    path = tmp_path / 'faces.db'
    seed(path)
    before = path.read_bytes()
    monkeypatch.setattr(maintenance, 'writer_lock', lambda _: pytest.fail('dry-run must not acquire writer'))
    first, last = maintenance.compact_database(path, dry_run=True)
    assert first is last and first['free_page_bytes'] > 0
    assert path.read_bytes() == before


def test_active_recognition_writer_blocks_vacuum_but_allows_status(tmp_path):
    path = tmp_path / 'faces.db'
    with FaceDatabase(path) as state:
        run = state.start_run('scope', ['1'], {})
        assert maintenance.inspect_database(path)['states']['runs'] == {'running': 1}
        with pytest.raises(RuntimeError, match='recognition writer'):
            maintenance.compact_database(path)
        assert state.connection.execute('SELECT status FROM runs WHERE id=?', (run,)).fetchone()[0] == 'running'


def test_insufficient_disk_space_refuses_vacuum_and_releases_lock(tmp_path, monkeypatch):
    path = tmp_path / 'faces.db'
    seed(path)
    expected = path.read_bytes()
    monkeypatch.setattr(maintenance.shutil, 'disk_usage', lambda _: SimpleNamespace(free=0))
    with pytest.raises(RuntimeError, match='Insufficient free disk space'):
        maintenance.compact_database(path)
    assert path.read_bytes() == expected
    with maintenance.writer_lock(path):
        pass


@pytest.mark.parametrize('command', ['db-status', 'db-report', 'db-vacuum', 'db-compact'])
def test_offline_cli_and_aliases_emit_json_without_models_or_stash(tmp_path, monkeypatch, capsys, command):
    path = tmp_path / 'faces.db'
    seed(path)
    config = tmp_path / 'faces.yml'
    config.write_text('version: 1\nstorage: {database: "missing.db"}\n')
    monkeypatch.setenv('STASH_URL', 'invalid-unused-url')
    import stash_ai.recognition.run as run
    monkeypatch.setattr(run, 'Services', lambda: pytest.fail('No recognition runtime'))
    args = [command, '--faces-db', str(path), '--face-config', str(config), '--json']
    if command in ('db-vacuum', 'db-compact'):
        args += ['--dry-run']
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    if command in ('db-vacuum', 'db-compact'):
        assert result['dry_run'] and result['saved_disk_bytes'] == 0
        result = result['before']
    assert result['path'] == str(path)
    assert result['states']['runs'] == {'running': 1}
    with pytest.raises(SystemExit):
        parser().parse_args(['db-status', '--dry-run'])


@pytest.mark.parametrize('operation', [maintenance.inspect_database, maintenance.compact_database])
def test_missing_and_foreign_databases_are_not_created_or_modified(tmp_path, operation):
    path = tmp_path / 'missing.db'
    with pytest.raises(FileNotFoundError):
        operation(path)
    assert list(tmp_path.iterdir()) == []
    with closing(sqlite3.connect(path)) as db:
        db.execute('CREATE TABLE unrelated(value)')
    before = path.read_bytes()
    with pytest.raises(ValueError, match='facial recognition'):
        operation(path)
    assert path.read_bytes() == before
    assert not Path(str(path) + '.lock').exists()


def test_legacy_schema_is_reported_and_compacted_without_migration(tmp_path):
    path = tmp_path / 'legacy.db'
    legacy(path)
    expected = logical_rows(path)
    assert maintenance.inspect_database(path)['schema_version'] == 1
    before, after = maintenance.compact_database(path)
    assert before['schema_version'] == after['schema_version'] == 1
    assert logical_rows(path) == expected


@pytest.mark.parametrize('extra', [
    'CREATE VIEW unexpected AS SELECT * FROM metadata',
    'CREATE TRIGGER unexpected AFTER DELETE ON metadata BEGIN DELETE FROM people; END',
    'DROP TABLE portraits',
])
def test_invalid_schema_is_not_vacuumed(tmp_path, extra):
    path = tmp_path / 'faces.db'
    seed(path)
    with closing(sqlite3.connect(path)) as db, db:
        db.execute(extra)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        maintenance.compact_database(path)
    assert path.read_bytes() == before


def test_hard_link_is_refused(tmp_path):
    path, alias = tmp_path / 'faces.db', tmp_path / 'alias.db'
    seed(path)
    try:
        os.link(path, alias)
    except OSError:
        pytest.skip('hard links unavailable')
    with pytest.raises(ValueError, match='hard-linked'):
        maintenance.compact_database(alias)


def test_missing_dbstat_reports_logical_values_without_fabricating_physical_sizes(tmp_path, monkeypatch):
    path = tmp_path / 'faces.db'
    seed(path)
    monkeypatch.setattr(maintenance, '_page_usage', lambda _: None)
    report = maintenance.inspect_database(path)
    assert not report['dbstat_available'] and report['tables_total_bytes'] is None
    face = next(row for row in report['tables'] if row['name'] == 'observations')
    assert face['table_bytes'] is face['index_bytes'] is face['total_bytes'] is None
    assert face['logical_value_bytes'] > 100_000 and face['rows'] == 1
    assert 'physical size unavailable' in maintenance.format_report(report)
    before, after = maintenance.compact_database(path)
    assert after['file_bytes'] < before['file_bytes']


def test_wal_totals_include_uncheckpointed_pages_and_vacuum_truncates_wal(tmp_path):
    path = tmp_path / 'faces.db'
    seed(path)
    with closing(sqlite3.connect(path)) as writer:
        writer.execute('PRAGMA journal_mode=WAL')
        writer.execute('PRAGMA wal_autocheckpoint=0')
        with writer:
            writer.execute("INSERT INTO metadata VALUES ('wal-value',?)", (b'\x00' * 3_000_000,))
        before = maintenance.inspect_database(path)
        assert before['wal_bytes'] > 600_000
        assert before['total_disk_bytes'] == sum(before[key] for key in ('file_bytes', 'wal_bytes', 'shm_bytes', 'journal_bytes'))
        assert before['allocated_bytes'] > before['file_bytes']
        expected = logical_rows(path)
        first, after = maintenance.compact_database(path)
        assert first['wal_bytes'] > 0 and after['wal_bytes'] == 0
        assert after['journal_mode'] == 'wal' and logical_rows(path) == expected


def test_busy_wal_reader_prevents_vacuum(tmp_path):
    path = tmp_path / 'faces.db'
    seed(path)
    with closing(sqlite3.connect(path)) as writer, closing(sqlite3.connect(path)) as reader:
        writer.execute('PRAGMA journal_mode=WAL')
        reader.execute('BEGIN')
        reader.execute('SELECT * FROM metadata').fetchall()
        with writer:
            writer.execute("INSERT INTO metadata VALUES ('new-wal-value','preserve')")
        expected = logical_rows(path)
        with pytest.raises(RuntimeError, match='WAL checkpoint is busy'):
            maintenance.compact_database(path)
        assert logical_rows(path) == expected


def test_failed_vacuum_releases_writer_and_closes_connection(tmp_path, monkeypatch):
    path = tmp_path / 'faces.db'
    seed(path)
    expected = logical_rows(path)

    def interrupt(phase, completed, total):
        if phase == 'VACUUM' and not completed:
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        maintenance.compact_database(path, notify=interrupt)
    assert logical_rows(path) == expected
    with maintenance.writer_lock(path):
        pass


def test_cli_vacuum_prints_actual_savings_and_preserves_history(tmp_path, capsys):
    path = tmp_path / 'faces.db'
    seed(path)
    config = tmp_path / 'faces.yml'
    config.write_text('version: 1\nstorage: {database: "faces.db"}\n')
    before = path.stat().st_size
    assert main(['db-vacuum', '--face-config', str(config), '--progress-style', 'off']) == 0
    assert path.stat().st_size < before
    text = capsys.readouterr().out
    assert 'VACUUM complete:' in text and 'saved' in text and 'Health: ok' in text
    assert maintenance.inspect_database(path)['states']['runs'] == {'running': 1}


def test_unclean_health_blocks_vacuum_without_rewriting_state(tmp_path, monkeypatch):
    path = tmp_path / 'faces.db'
    seed(path)
    before = path.read_bytes()
    inspect = maintenance._inspect

    def unhealthy(*args):
        result = inspect(*args)
        result['quick_check'] = 'synthetic corruption'
        return result

    monkeypatch.setattr(maintenance, '_inspect', unhealthy)
    with pytest.raises(RuntimeError, match='quick_check is not clean'):
        maintenance.compact_database(path)
    assert path.read_bytes() == before
