"""Constant-work update preflight, SQL invalidation, and authentic v18 migrations."""
from pathlib import Path
import sqlite3
import zlib

import pytest

from stash_ai.db.update import update_database
from stash_ai.db.progress import maintenance_progress
from stash_ai.runtime.report import RunReport
from stash_ai.storage import codec
from stash_ai.storage.metadata_ledger import definitions, trusted
from stash_ai.storage.schema import LATEST_SCHEMA_VERSION
from stash_ai.storage.store import StateStore
from tests.helpers.evidence import inference_record
from tests.helpers.runtime import runtime as app

pytestmark = pytest.mark.usefixtures('reset_merge')


def seed(path, count):
    record = inference_record(mode='scene', items=[])
    with StateStore(path, 'scope') as store:
        for i in range(count):
            store.put(str(i), 'wd', {'scene': record})


def test_clean_preflight_never_reads_results_or_metadata(tmp_path, monkeypatch):
    from stash_ai.db import update
    path = tmp_path/'state.db'
    seed(path, 1000)
    connect = update.connect_readonly
    def readonly(*args):
        db = connect(*args)
        def authorize(action, table, *args):
            if action == sqlite3.SQLITE_READ and table in ('results', 'result_metadata'):
                pytest.fail('clean preflight scanned evidence/metadata')
            return sqlite3.SQLITE_OK
        db.set_authorizer(authorize)
        return db
    monkeypatch.setattr(update, 'connect_readonly', readonly)
    assert update_database(path) == (LATEST_SCHEMA_VERSION, LATEST_SCHEMA_VERSION)


def test_current_writer_upsert_handles_existing_dirty_key(tmp_path):
    path = tmp_path/'state.db'
    seed(path, 1)
    record = inference_record(mode='scene', items=[])
    with StateStore(path, 'scope') as store:
        store.put('0', 'wd', {'scene': record})
        store.connection.execute("DELETE FROM result_metadata WHERE scene='0'")
        store.connection.commit()
        store.put('0', 'wd', {'scene': record})
        assert store.connection.execute('SELECT * FROM metadata_dirty').fetchall() == []


@pytest.mark.parametrize('sql', [
    "DELETE FROM result_metadata WHERE scene='0'",
    "UPDATE result_metadata SET data='invalid' WHERE scene='0'",
    "UPDATE result_metadata SET format=0 WHERE scene='0'",
    "UPDATE results SET data=data WHERE scene='0'",
    "INSERT OR REPLACE INTO results SELECT * FROM results WHERE scene='0'",
])
def test_external_sql_marks_only_affected_result_for_repair(tmp_path, sql):
    path = tmp_path/'state.db'
    seed(path, 5)
    with sqlite3.connect(path) as db:
        db.execute(sql)
        assert db.execute('SELECT scene FROM metadata_dirty').fetchall() == [('0',)]
    details = {}
    update_database(path, details=details)
    assert details['metadata_repaired'] == 1
    with sqlite3.connect(path) as db:
        assert trusted(db)
        assert db.execute('SELECT * FROM metadata_dirty').fetchall() == []
        db.execute("DELETE FROM result_metadata WHERE scene='0'")
        db.execute("DELETE FROM results WHERE scene='0'")
        assert db.execute('SELECT * FROM metadata_dirty').fetchall() == []


@pytest.mark.parametrize('damage', ['absent', 'tampered'])
def test_untrusted_ledger_is_reseeded_before_fast_preflight(tmp_path, damage):
    path = tmp_path/'state.db'
    seed(path, 3)
    with sqlite3.connect(path) as db:
        name = next(iter(definitions()))
        db.execute(f'DROP TRIGGER {name}')
        if damage == 'tampered':
            db.execute(f'CREATE TRIGGER {name} AFTER INSERT ON results BEGIN SELECT 1; END')
        db.execute("DELETE FROM result_metadata WHERE scene='0'")
        db.execute('DELETE FROM metadata_dirty')
    details = {}
    update_database(path, details=details)
    assert not details['noop'] and details['metadata_repaired'] == 1
    with sqlite3.connect(path) as db:
        assert trusted(db)
    update_database(path, details=details)
    assert details['noop']


@pytest.mark.parametrize('error,status', [(None, 'completed'), (RuntimeError, 'failed'), (KeyboardInterrupt, 'interrupted')])
def test_maintenance_records_phase_outcomes_without_changing_error(tmp_path, error, status):
    report = RunReport(write=lambda _: None)
    def operation():
        with maintenance_progress('test', report=report) as notify:
            notify('reading', 0, 4)
            notify('reading', 3, 4)
            if error:
                raise error('primary')
            notify('reading', 4, 4)
            notify('committed', 1, 1)
    if error:
        with pytest.raises(error, match='primary'):
            operation()
    else:
        operation()
    outcome = report.workload['maintenance'][0]
    assert outcome['status'] == status
    assert outcome['phases'][-1]['status'] == status
    assert all(row['seconds'] >= 0 and '_start' not in row for row in outcome['phases'])
    assert outcome['phases'][0]['completed'] == (3 if error else 4)


def test_db_update_history_records_noop_reason(tmp_path):
    from tests.test_run_history import history_rows
    path = tmp_path/'state.db'
    seed(path, 1)
    assert app.main(['--db-update', '--state-db', str(path)]) == 0
    maintenance = history_rows(path)[0]['workload']['maintenance'][0]
    assert maintenance['status'] == 'completed'
    assert maintenance['phases'] == []
    assert maintenance['result']['skipped_reason'] == 'already current'


def historical(path):
    with sqlite3.connect(path) as db:
        db.executescript((Path(__file__).parent/'fixtures/schema_v18.sql').read_text())
        assert db.execute('PRAGMA user_version').fetchone() == (18,)
        assert 'inference' in [r[1] for r in db.execute('PRAGMA table_info(result_index)')]
        assert not db.execute("SELECT 1 FROM sqlite_master WHERE name='inference_identities'").fetchone()
        snapshot = codec.encode({'__stash_ai_snapshot__': 4, 'snapshot': {'owned_ids': ['saved'], 'contributions': []}, 'proofs': []})
        for scope in ('first', 'second'):
            # Contemporary v18 records use analysis v4; only those obsolete observations are retired.
            old = dict(mode='frames', meta=dict(origin='inference', analysis_format=4), items=[])
            db.execute('INSERT INTO results VALUES (?,?,?,?,?)', (scope,'1','wd','markers',codec.encode(old)))
            db.execute('INSERT INTO snapshots VALUES (?,?,?,?,?)', (scope,'1','markers',snapshot,snapshot))
            db.execute('INSERT INTO recovery_publications VALUES (?,?,?,?,?)', (scope,'1','markers','saved',snapshot))
            db.execute('INSERT INTO checkpoints VALUES (?,?,?,?,?,?,?,?)', (scope,'2','wd','saved',1,2,1,zlib.compress(b'\x00\x00\x00\x3f')))
            db.execute('INSERT INTO checkpoint_frames VALUES (?,?,?,?,?)', (scope,'2','wd',1,codec.encode([dict(start=0.,end=1.,tags=[])])))


def test_authentic_v18_schema_preserves_pending_recovery_and_checkpoints(tmp_path):
    path = tmp_path/'historical.db'
    historical(path)
    with sqlite3.connect(path) as db:
        checkpoints = db.execute('SELECT * FROM checkpoints ORDER BY scope').fetchall()
        chunks = db.execute('SELECT * FROM checkpoint_frames ORDER BY scope').fetchall()
    assert update_database(path) == (18, LATEST_SCHEMA_VERSION)
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT * FROM checkpoints ORDER BY scope').fetchall() == checkpoints
        assert db.execute('SELECT * FROM checkpoint_frames ORDER BY scope').fetchall() == chunks
        assert db.execute('SELECT scope FROM refresh_requests ORDER BY scope').fetchall() == [('first',), ('second',)]
        assert db.execute('SELECT COUNT(*) FROM recovery_publications').fetchone() == (2,)
        assert db.execute('SELECT COUNT(*) FROM snapshots WHERE pending IS NOT NULL').fetchone() == (2,)
    with StateStore(path, 'first', allow_cleanup=True) as store:
        assert store.snapshot('1','markers')[0]['owned_ids'] == ['saved']


def test_authentic_v18_failure_rolls_back_payloads_and_version(tmp_path):
    path = tmp_path/'historical.db'
    historical(path)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE snapshots SET pending=? WHERE scope='second'", (b'invalid',))
        before = {t: db.execute(f'SELECT * FROM {t} ORDER BY 1,2').fetchall() for t in ('results','snapshots','checkpoints','recovery_publications')}
    with pytest.raises(ValueError, match='could not be migrated'):
        update_database(path)
    with sqlite3.connect(path) as db:
        assert db.execute('PRAGMA user_version').fetchone() == (18,)
        if db.execute("SELECT 1 FROM sqlite_master WHERE name='data_revisions'").fetchone():
            assert not db.execute('SELECT * FROM data_revisions').fetchall()
        assert before == {t: db.execute(f'SELECT * FROM {t} ORDER BY 1,2').fetchall() for t in before}
