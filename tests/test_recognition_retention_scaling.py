"""Bounded JSON reference traversal preserves pins and exposes planning work."""
from contextlib import closing
import json
import sqlite3

import pytest

from stash_ai.recognition import db_retention as retention, retention_plan as plan
from stash_ai.recognition.database import FaceDatabase, encoded
from stash_ai.recognition.progress import observe_updates
from tests.test_recognition_retention import logical, scans, settings


def reference_db():
    db = sqlite3.connect(':memory:')
    db.executescript('''
        CREATE TABLE candidates(id TEXT PRIMARY KEY, scan TEXT, payload TEXT);
        CREATE TEMP TABLE retention_targets(id TEXT PRIMARY KEY, scan TEXT);
        CREATE TEMP TABLE retention_pins(id TEXT PRIMARY KEY, image INTEGER);
        CREATE TEMP TABLE retention_scans(id TEXT PRIMARY KEY, action TEXT);
    ''')
    return db


def test_reference_join_does_not_multiply_targets_by_json_documents():
    # A VM-instruction ceiling detects the former quadratic join without
    # depending on wall-clock speed or SQLite's EXPLAIN wording.
    count = 1500
    with closing(reference_db()) as db:
        db.executemany('INSERT INTO retention_targets VALUES (?,?)',
                       [(f'face-{i}', 'scan') for i in range(count)])
        db.executemany('INSERT INTO candidates VALUES (?,?,?)',
                       [(str(i), 'scan', json.dumps({'observation': f'face-{i}', 'scores': list(range(10))}))
                        for i in range(count)])
        db.commit()
        instructions = [0]
        def budget():
            instructions[0] += 1000
            return instructions[0] > 2_000_000
        db.set_progress_handler(budget, 1000)
        try:
            plan._pin_json(db, 'candidates', 'payload', '1', 0)
        finally:
            db.set_progress_handler(None, 0)
        assert db.execute('SELECT COUNT(*) FROM retention_pins').fetchone()[0] == count
        assert instructions[0] <= 2_000_000


def test_batches_count_source_rows_with_sparse_negative_rowids_and_nested_pins(monkeypatch):
    monkeypatch.setattr(plan, 'REFERENCE_BATCH_ROWS', 2)
    events = []
    with closing(reference_db()) as db:
        db.executemany('INSERT INTO retention_targets VALUES (?,?)',
                       [('a', 'old'), ('b', 'current'), ('42', 'current'), ('key-only', 'current')])
        db.execute("INSERT INTO retention_pins VALUES ('a', 2)")
        db.execute("INSERT INTO retention_scans VALUES ('old', 'obsolete')")
        db.executemany('INSERT INTO candidates(rowid,id,scan,payload) VALUES (?,?,?,?)', [
            (-10, 'one', 'old', encoded({'nested': ['a', {'cross_scan': 'b'}]})),
            (0, 'two', 'current', encoded({'refs': [['a'], ['a']], 'numeric': 42})),
            (10, 'three', 'current', encoded({'key-only': None, 'missing': 'not-a-target'})),
            (1000000, 'four', 'current', encoded(['a', 'b'])),
        ])
        condition = "(t.scan NOT IN (SELECT id FROM retention_scans WHERE action='obsolete') OR e.scan<>t.scan)"
        with observe_updates(lambda *event: events.append(event)):
            plan._pin_json(db, 'candidates', 'payload', condition, 1)
        assert dict(db.execute('SELECT id,image FROM retention_pins')) == {'a': 2, 'b': 1}
        updates = [data for kind, data in events if kind == 'work']
        assert [data['completed'] for data in updates] == [0, 2, 4, 4]
        assert all(data['total'] == 4 and data['unit'] == 'record' for data in updates)


def test_same_scan_obsolete_candidate_does_not_pin_its_own_evidence():
    with closing(reference_db()) as db:
        db.executemany('INSERT INTO retention_targets VALUES (?,?)', [('own', 'old'), ('other', 'elsewhere')])
        db.execute("INSERT INTO retention_scans VALUES ('old', 'obsolete')")
        db.execute('INSERT INTO candidates VALUES (?,?,?)', ('decision', 'old', encoded(['own', 'other'])))
        condition = "(t.scan NOT IN (SELECT id FROM retention_scans WHERE action='obsolete') OR e.scan<>t.scan)"
        plan._pin_json(db, 'candidates', 'payload', condition, 0)
        assert dict(db.execute('SELECT id,image FROM retention_pins')) == {'other': 0}


def test_later_malformed_batch_aborts_retention_and_does_not_finish_row(tmp_path, monkeypatch):
    monkeypatch.setattr(plan, 'REFERENCE_BATCH_ROWS', 2)
    config = settings(tmp_path)
    with FaceDatabase(config.database) as state:
        scans(state, config)
        with state.connection as db:
            db.executemany("INSERT INTO candidates VALUES (?,'s3','policy',NULL,'accepted',?)", [
                ('first', encoded({'witness': 'face-s0'})),
                ('second', encoded({'witness': 'face-s1'})),
                ('bad', 'not-json'),
            ])
    before, events = logical(config.database), []
    with observe_updates(lambda *event: events.append(event)):
        with pytest.raises(ValueError, match='malformed candidates.payload'):
            retention.prune_database(config, apply=True)
    assert logical(config.database) == before
    updates = [data for kind, data in events if kind == 'work'
               and data['label'] == 'Inspect references: candidates.payload']
    assert updates[-1]['completed'] == 2 and updates[-1]['total'] == 3


def test_selected_scene_keeps_evidence_referenced_from_another_scope(tmp_path):
    config = settings(tmp_path)
    with FaceDatabase(config.database) as state:
        scans(state, config)
        scans(state, config, scope='foreign', scene='2', prefix='foreign', count=1)
        with state.connection as db:
            db.execute("INSERT INTO candidates VALUES ('foreign-decision','foreign0','policy',NULL,'accepted',?)",
                       (encoded({'nested': {'history': ['face-s0']}}),))
    result = retention.prune_database(config, apply=True, scope='scope', scenes=['1'])
    assert result['scans'] == {'archive': 1, 'delete': 1, 'keep': 2}
    with FaceDatabase(config.database) as state:
        assert state.connection.execute("SELECT embedding FROM observations WHERE id='face-s0'").fetchone()[0]
        assert state.connection.execute("SELECT 1 FROM candidates WHERE id='foreign-decision'").fetchone()


def test_unobserved_reference_checks_do_not_run_optional_total_counts():
    with closing(reference_db()) as db:
        statements = []
        db.set_trace_callback(statements.append)
        plan._pin_json(db, 'candidates', 'payload', '1', 0)
        assert 'SELECT COUNT(*) FROM candidates' not in statements


def test_warm_retention_refreshes_only_changed_documents_and_repairs_missing_pins(tmp_path):
    from stash_ai.recognition.retention_index import refresh
    config=settings(tmp_path)
    with FaceDatabase(config.database) as state:
        scans(state,config)
        db=state.connection
        with db:
            db.execute("INSERT INTO candidates VALUES ('other','s3','p',NULL,'accepted',?)",(encoded(['face-s0']),))
        # An external decision preserves s0; after applying, its indexed document
        # and verified runs survive reopen and unchanged previews.
        db.execute('BEGIN IMMEDIATE')
        assert retention.on_connection(db,config,apply=True)['scans']['archive']==1
        db.commit()
        def refreshed():
            operations=[]
            db.set_trace_callback(operations.append)
            try:
                with db:
                    assert refresh(db,True)
            finally:db.set_trace_callback(None)
            return [s for s in operations if 'CROSS JOIN json_tree' in s or "CROSS JOIN json_each(p.data" in s]
        assert refreshed()==[]
        with db:
            db.execute("DELETE FROM retention_links WHERE doc='other'")
        assert len(refreshed())>=1
        assert db.execute("SELECT value FROM retention_links WHERE doc='other'").fetchone()[0]==b'face-s0'
        db.execute("UPDATE candidates SET payload='bad' WHERE id='other'")
        db.rollback()
        assert refreshed()==[]
        with db:
            db.execute("UPDATE candidates SET payload='[]' WHERE id='other'")
        assert refreshed()
        assert not db.execute("SELECT 1 FROM retention_links WHERE doc='other'").fetchone()


def test_version_ten_upgrades_and_indexes_legacy_cross_scene_pins(tmp_path):
    import uuid
    from stash_ai.recognition import database
    config=settings(tmp_path)
    with closing(sqlite3.connect(config.database)) as db:
        db.executescript(database.SCHEMA_V10)
        db.execute(f'PRAGMA application_id={database.APPLICATION_ID}')
        db.execute('PRAGMA user_version=10')
        db.execute("INSERT INTO metadata VALUES ('installation',?)",(str(uuid.uuid4()),))
        db.row_factory=sqlite3.Row
        legacy=FaceDatabase.__new__(FaceDatabase)
        legacy.connection=db
        scans(legacy,config)
        db.execute("INSERT INTO candidates VALUES ('legacy','s3','p',NULL,'accepted',?)",(encoded(['face-s0']),))
        db.commit()
    with FaceDatabase(config.database) as state:
        assert state.connection.execute('PRAGMA user_version').fetchone()[0]==database.SCHEMA_VERSION
    result=retention.prune_database(config,apply=True)
    assert result['scans']=={'archive':1,'delete':1,'keep':2}
