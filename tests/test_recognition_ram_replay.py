from dataclasses import replace
import json
import sqlite3
import threading

import numpy as np
import pytest

from stash_ai.recognition.config import plain
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.rebuild import rebuild_selected
from stash_ai.recognition.replay_memory import memory_replay, Lookahead, prepared_vector, MISSING
from tests.test_recognition_clustering import setup,scan


def enabled(config,**execution):
    data=plain(config.data)
    data['execution'].update(execution)
    return replace(config,data=data)


def test_ram_replay_has_no_file_and_enforces_page_capacity(tmp_path):
    config=enabled(setup(tmp_path),rebuild_memory_bytes=1048576)
    with pytest.raises(ValueError,match='RAM replay exceeds'):
        with memory_replay(config) as (db,uri):
            assert db.execute('PRAGMA database_list').fetchone()[2]==''
            assert db.execute('PRAGMA temp_store').fetchone()[0]==2
            db.execute('CREATE TABLE too_big(value BLOB)')
            db.execute('INSERT INTO too_big VALUES (?)',(b'x'*1048576,))


def test_lookahead_matches_serial_decisions_and_keeps_writes_on_owner(tmp_path):
    config=setup(tmp_path)
    with FaceDatabase(config.database) as state:
        selected=[]
        for i in range(8):
            v=np.array([1.,0,0,0]) if i%2 else np.array([0.,1,0,0])
            identifier,_=scan(state,config,str(i+1),[v]*3)
            selected.append(dict(scene=str(i+1),scan=identifier))
        expected=rebuild_selected(state,selected,config)
        members=[tuple(r) for r in state.connection.execute('SELECT * FROM members ORDER BY observation')]
        candidates=[tuple(r) for r in state.connection.execute('SELECT * FROM candidates ORDER BY id')]
        state.connection.execute("DELETE FROM metadata WHERE key LIKE 'identity-rebuild:%'")
        state.connection.commit()
        owner=threading.get_ident()
        writes=[]
        state.connection.set_trace_callback(lambda sql:writes.append(threading.get_ident()))
        assert rebuild_selected(state,selected,enabled(config,rebuild_prefetch_scenes=True,profile_rebuild=True))==expected
        assert set(writes)=={owner}
        assert members==[tuple(r) for r in state.connection.execute('SELECT * FROM members ORDER BY observation')]
        assert candidates==[tuple(r) for r in state.connection.execute('SELECT * FROM candidates ORDER BY id')]
        assert state.last_rebuild_profile['counts']['prefetched_replay_vectors']>0
        assert state.last_rebuild_profile['counts']['replay_image_bytes']>0
        state.connection.set_trace_callback(None)


def test_prefetch_defers_bad_vectors_and_accepts_large_scene(tmp_path):
    config=enabled(setup(tmp_path),rebuild_prefetch_scenes=True)
    with memory_replay(config) as (db,uri):
        db.execute('CREATE TABLE observations(id TEXT,scan TEXT,dimensions INTEGER,embedding BLOB,rejected TEXT,cue INTEGER,geometry TEXT)')
        db.execute("INSERT INTO observations VALUES ('bad','s',4,?,NULL,0,'{}')",(b'bad',))
        db.commit()
        with Lookahead(uri,config) as ahead:
            ahead.submit('s'); ahead.take('s')
            with pytest.raises(ValueError,match='embedding'):
                prepared_vector(dict(id='bad',dimensions=4,embedding=b'bad'))
            with pytest.raises(RuntimeError,match='prefetched evidence changed'):
                prepared_vector(dict(id='bad',dimensions=4,embedding=b'changed'))
        embedding=np.array([1.,0.,0.,0.],dtype='<f4').tobytes()
        db.executemany("INSERT INTO observations VALUES (?,'large',4,?,NULL,0,'{}')",
                       [(str(i),embedding) for i in range(257)])
        db.commit()
        with Lookahead(uri,config) as ahead:
            ahead.submit('large');ahead.take('large')
            assert len(ahead.prepared)==257
            assert prepared_vector(dict(id='256',dimensions=4,embedding=embedding))[0]==1.


def test_prefetch_handles_more_than_eight_mib_of_embeddings(tmp_path):
    config=enabled(setup(tmp_path),rebuild_prefetch_scenes=True)
    embedding=np.zeros(1024,dtype='<f4')
    embedding[0]=1.
    blob=embedding.tobytes()
    total=2050
    assert total*len(blob)>8*1024*1024
    with memory_replay(config) as (db,uri):
        db.execute('CREATE TABLE observations(id TEXT,scan TEXT,dimensions INTEGER,embedding BLOB,rejected TEXT,cue INTEGER,geometry TEXT)')
        db.executemany("INSERT INTO observations VALUES (?,'large',1024,?,NULL,0,'{}')",
                       ((str(i),blob) for i in range(total)))
        db.commit()
        with Lookahead(uri,config) as ahead:
            ahead.submit('large');ahead.take('large')
            assert len(ahead.prepared)==total
            assert prepared_vector(dict(id=str(total-1),dimensions=1024,embedding=blob))[0]==1.


def test_prefetch_memory_budget_exhaustion_fails_when_enabled(tmp_path):
    # No fixed 256/8 MiB admission cap, but an explicit RAM budget failure
    # cannot silently degrade a requested lookahead into serial matching.
    config=enabled(setup(tmp_path),rebuild_prefetch_scenes=True,rebuild_memory_bytes=4*1024*1024)
    blob=np.array([1.]+[0.]*1023,dtype='<f4').tobytes()
    with memory_replay(config) as (db,uri):
        db.execute('CREATE TABLE observations(id TEXT,scan TEXT,dimensions INTEGER,embedding BLOB,rejected TEXT,cue INTEGER,geometry TEXT)')
        db.executemany("INSERT INTO observations VALUES (?,'large',1024,?,NULL,0,'{}')",
                       ((str(i),blob) for i in range(120)))
        db.commit()
        with Lookahead(uri,config) as ahead:
            ahead.submit('large')
            with pytest.raises(RuntimeError,match='prefetch failed for scan large') as failure:
                ahead.take('large')
            assert 'prefetch memory budget exceeded' in str(failure.value)
            assert ahead.prepared=={}
            assert ahead.pending is None and ahead.scan is None



def test_enabled_prefetch_fails_on_sqlite_table_lock(tmp_path):
    config=enabled(setup(tmp_path),rebuild_prefetch_scenes=True)
    blob=np.array([1.,0.,0.,0.],dtype='<f4').tobytes()
    with memory_replay(config) as (db,uri):
        db.execute('CREATE TABLE observations(id TEXT,scan TEXT,dimensions INTEGER,embedding BLOB,rejected TEXT,cue INTEGER,geometry TEXT)')
        db.execute("INSERT INTO observations VALUES ('one','s',4,?,NULL,0,'{}')",(blob,))
        db.commit()
        # Shared-cache memory databases reject reads while the writer holds
        # this table. With enabled lookahead the error must be visible.
        db.execute("UPDATE observations SET dimensions=dimensions WHERE id='one'")
        with Lookahead(uri,config) as ahead:
            ahead.submit('s')
            with pytest.raises(RuntimeError,match='prefetch failed for scan s') as failure:
                ahead.take('s')
            assert isinstance(failure.value.__cause__,sqlite3.OperationalError)
            assert ahead.prepared=={}
            assert ahead.pending is None and ahead.scan is None
        db.rollback()
        with Lookahead(uri,config) as ahead:
            ahead.submit('s');ahead.take('s')
            assert len(ahead.prepared)==1


def test_prefetch_uses_scan_index_without_temp_sort(tmp_path):
    config=enabled(setup(tmp_path),rebuild_prefetch_scenes=True)
    with memory_replay(config) as (db,uri):
        db.execute('CREATE TABLE observations(id TEXT,scan TEXT,dimensions INTEGER,embedding BLOB,rejected TEXT,cue INTEGER,geometry TEXT)')
        db.execute('CREATE INDEX observations_scan ON observations(scan,cue,id)')
        plan=[row[3] for row in db.execute(
            "EXPLAIN QUERY PLAN SELECT id,dimensions,embedding FROM observations WHERE scan=? "
            "AND rejected IS NULL ORDER BY cue,id",('s',))]
        assert any('USING INDEX observations_scan' in part for part in plan)
        assert not any('TEMP B-TREE' in part for part in plan)



def test_enabled_prefetch_requires_submitted_scene(tmp_path):
    config=enabled(setup(tmp_path),rebuild_prefetch_scenes=True)
    with memory_replay(config) as (_,uri):
        with Lookahead(uri,config) as ahead:
            with pytest.raises(RuntimeError,match='was not scheduled'):
                ahead.take('missing')


def test_prefetch_worker_failure_keeps_durable_assignments_unchanged(tmp_path,monkeypatch):
    from stash_ai.recognition import replay_memory
    config=enabled(setup(tmp_path),rebuild_prefetch_scenes=True)
    with FaceDatabase(config.database) as state:
        selected=[]
        for index in range(3):
            identifier,_=scan(state,config,str(index+1),
                              [np.array([1.,0.,0.,0.],dtype='<f4')]*3)
            selected.append(dict(scene=str(index+1),scan=identifier))
        expected_members=[tuple(row) for row in state.connection.execute(
            'SELECT * FROM members ORDER BY observation')]
        expected_candidates=[tuple(row) for row in state.connection.execute(
            'SELECT * FROM candidates ORDER BY id')]
        calls=[]
        def broken(self,scan_id):
            calls.append(scan_id)
            raise sqlite3.OperationalError('injected prefetch read failure')
        monkeypatch.setattr(replay_memory.Lookahead,'_prepare',broken)
        with pytest.raises(RuntimeError,match='prefetch failed for scan') as failure:
            rebuild_selected(state,selected,config)
        assert isinstance(failure.value.__cause__,sqlite3.OperationalError)
        assert calls==[selected[0]['scan']]
        assert expected_members==[tuple(row) for row in state.connection.execute(
            'SELECT * FROM members ORDER BY observation')]
        assert expected_candidates==[tuple(row) for row in state.connection.execute(
            'SELECT * FROM candidates ORDER BY id')]
