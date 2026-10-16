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


def test_prefetch_defers_bad_vectors_and_refuses_oversized_scene(tmp_path):
    config=enabled(setup(tmp_path),rebuild_prefetch_scenes=True)
    with memory_replay(config) as (db,uri):
        db.execute('CREATE TABLE observations(id TEXT,scan TEXT,dimensions INTEGER,embedding BLOB,rejected TEXT,cue INTEGER,geometry TEXT)')
        db.execute("INSERT INTO observations VALUES ('bad','s',4,?,NULL,0,'{}')",(b'bad',))
        db.commit()
        with Lookahead(uri,config) as ahead:
            ahead.submit('s'); ahead.take('s')
            with pytest.raises(ValueError,match='embedding'):
                prepared_vector(dict(id='bad',dimensions=4,embedding=b'bad'))
            assert prepared_vector(dict(id='bad',dimensions=4,embedding=b'changed')) is MISSING
        db.executemany("INSERT INTO observations VALUES (?,'large',4,?,NULL,0,'{}')",[(str(i),b'bad') for i in range(257)])
        db.commit()
        with Lookahead(uri,config) as ahead:
            ahead.submit('large');ahead.take('large')
            assert ahead.prepared=={}
