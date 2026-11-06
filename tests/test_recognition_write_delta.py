import json

from stash_ai.recognition.database import FaceDatabase


def test_progress_coalesces_and_flushes_final_value_without_committing_caller(tmp_path,monkeypatch):
    from tests.test_recognition_retention import settings
    from stash_ai.recognition import database
    config=settings(tmp_path)
    with FaceDatabase(config.database) as state:
        db=state.connection
        db.execute("INSERT INTO runs(id,scope,selection,config,started,status) VALUES ('run','scope','[]','{}',0,'running')")
        db.commit()
        monkeypatch.setattr(database.time,'monotonic',lambda:10.)
        writes=[]
        db.set_trace_callback(lambda sql:writes.append(sql) if sql.startswith('INSERT OR REPLACE INTO progress') else None)
        for i in range(100):state.progress('run','scan',dict(completed=i))
        assert len(set(writes))==1
        # Scan and run counters alternate in the real pipeline; they must not
        # force each other to commit on every scene.
        for i in range(100):
            state.progress('run','scan',dict(completed=i))
            state.progress('run','run',dict(completed=i))
        assert len(set(writes))<=3
        state.finish_run('run','success',{})
        assert len(set(writes))<=4
        assert json.loads(db.execute("SELECT data FROM progress WHERE run='run' AND phase='scan'").fetchone()[0])['completed']==99
        db.execute("INSERT INTO metadata VALUES ('tentative','true')")
        state.progress('run','scan',dict(completed=200))
        assert db.in_transaction
        db.rollback()
        assert db.execute("SELECT value FROM metadata WHERE key='tentative'").fetchone() is None
        assert json.loads(db.execute("SELECT data FROM progress WHERE run='run' AND phase='scan'").fetchone()[0])['completed']==99


def test_forced_rebuild_preserves_member_and_candidate_rowids(tmp_path,monkeypatch):
    from tests.test_recognition_rebuild import library
    from tests.test_recognition_run import run
    from stash_ai.recognition.rebuild import rebuild_selected
    config,_,services,_=library(tmp_path,monkeypatch,{'1':[[1.,0,0,0]]*3})
    assert run(config,services,'1')==0
    with FaceDatabase(config.database) as state:
        db=state.connection
        selected=[dict(scan=r['id'],scene=r['scene']) for r in db.execute('SELECT * FROM scans')]
        before=[tuple(r) for r in db.execute('SELECT rowid,* FROM members ORDER BY observation')]
        candidates=[tuple(r) for r in db.execute('SELECT rowid,* FROM candidates ORDER BY id')]
        db.execute("DELETE FROM metadata WHERE key LIKE 'identity-rebuild%'");db.commit()
        rebuild_selected(state,selected,config)
        assert [tuple(r) for r in db.execute('SELECT rowid,* FROM members ORDER BY observation')]==before
        assert [tuple(r) for r in db.execute('SELECT rowid,* FROM candidates ORDER BY id')]==candidates


def test_empty_prune_never_reads_evidence_tables(tmp_path):
    from tests.test_recognition_retention import settings,scans
    from stash_ai.recognition import retention_plan,db_retention
    config=settings(tmp_path)
    with FaceDatabase(config.database) as state:
        scans(state,config);db=state.connection
        db.execute('BEGIN')
        retention_plan.prepare(db,config,scope='scope',scenes=['999'],writable=True)
        queries=[];db.set_trace_callback(queries.append)
        try:db_retention._apply(db)
        finally:db.set_trace_callback(None);retention_plan.discard(db);db.rollback()
        assert not any('observations' in q or 'foreign_key_check' in q for q in queries)


def test_failed_standalone_body_apply_does_not_leave_temporary_delta_table(tmp_path,monkeypatch):
    from tests.test_recognition_body_global import configured,scene
    from stash_ai.recognition.body_identity import rebuild
    import pytest
    config=configured(tmp_path)
    with FaceDatabase(config.database) as state:
        db=state.connection;item,_=scene(db,config,'1');db.commit()
        def fail(*a,**kw):raise RuntimeError('publication failure')
        monkeypatch.setattr('stash_ai.recognition.publication_queue.enqueue',fail)
        with pytest.raises(RuntimeError,match='publication failure'):
            rebuild(state,[item],config,{item['scan']:[]})
        assert db.execute('SELECT count(*) FROM body_members').fetchone()[0]==0
        assert db.execute("SELECT 1 FROM sqlite_temp_master WHERE name='desired_body_members'").fetchone() is None
