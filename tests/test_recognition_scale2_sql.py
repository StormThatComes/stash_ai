import sqlite3

from stash_ai.recognition.database import SCHEMA
from stash_ai.recognition.rebuild import _apply_members


def test_members_apply_in_batches_and_only_mark_changed_scenes():
    db = sqlite3.connect(':memory:')
    db.executescript(SCHEMA)
    replay = sqlite3.connect(':memory:')
    replay.row_factory = sqlite3.Row
    replay.executescript('CREATE TABLE members(observation,person); CREATE TABLE sample_keys(observation,sample_key);')
    db.execute("INSERT INTO scans VALUES ('s','scope','7','ns','i','p','{}','{}',600,600,'complete')")
    db.executemany("INSERT INTO observations VALUES (?,'s',0,0,'','[]','{}','{}',NULL,2,NULL,NULL)",
                   ((str(i),) for i in range(600)))
    replay.executemany("INSERT INTO members VALUES (?,'temporary')", ((str(i),) for i in range(600)))
    replay.executemany('INSERT INTO sample_keys VALUES (?,?)', ((str(i),str(i)) for i in range(600)))
    changed = set()
    assert len(list(_apply_members(db,replay,{'temporary':'stable'},changed))) == 600
    assert changed == {'7'}
    before = list(db.execute('SELECT rowid,* FROM members'))
    changed.clear()
    list(_apply_members(db,replay,{'temporary':'stable'},changed))
    assert not changed and list(db.execute('SELECT rowid,* FROM members')) == before
    db.close(); replay.close()


def test_pruning_thumbnail_lookup_uses_expression_index(tmp_path):
    from stash_ai.recognition import db_retention, retention_plan
    from stash_ai.recognition.database import FaceDatabase
    from tests.test_recognition_retention import settings, scans
    config = settings(tmp_path)
    with FaceDatabase(config.database) as state:
        db = state.connection
        scans(state, config)
        db.execute('BEGIN')
        retention_plan.prepare(db,config,writable=True)
        plans=[]
        class Observer:
            def execute(self,sql,args=()):
                if sql.startswith('DELETE FROM evidence_thumbnails'):
                    plans.extend(db.execute('EXPLAIN QUERY PLAN '+sql,args))
                return db.execute(sql,args)
        try:
            db_retention._apply(Observer())
            assert any('observations_thumbnail' in row[3] and 'SEARCH' in row[3] for row in plans)
        finally:
            db.rollback()
