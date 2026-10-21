import json
import sqlite3

import pytest

from stash_ai.recognition.clustering import cluster_scene, matching_policy, complete_candidates
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.rebuild import _copy_selection, rebuild_selected, sample_key
from stash_ai.recognition.replay_addresses import addresses, address
from stash_ai.recognition.sample_keys import physical_keys
from tests.test_recognition_clustering import setup, scan


def test_sample_keys_reuse_all_dependencies_across_rebuilds(tmp_path,monkeypatch):
    from stash_ai.recognition import sample_keys
    config=setup(tmp_path)
    with FaceDatabase(config.database) as state:
        selected,_=scan(state,config,'1')
        cluster_scene(state,selected,config)
        original,calls=sample_keys.digest,[]
        def hashed(value):
            calls.append(1)
            return original(value)
        monkeypatch.setattr(sample_keys,'digest',hashed)
        for _ in range(2):
            with sqlite3.connect(':memory:') as replay, physical_keys(state,config):
                replay.row_factory=sqlite3.Row
                _copy_selection(state,replay,[dict(scene='1',scan=selected)],'scope','ns')
        assert len(calls)==3
        row=dict(state.connection.execute('SELECT o.*,s.* FROM observations o JOIN scans s ON s.id=o.scan LIMIT 1').fetchone())
        with physical_keys(state,config):
            first=sample_key(row)
            for field,value in [('geometry','changed'),('rect','[0,0,1,1]'),('input_sha','new'),('source','new'),('seconds',99)]:
                updated=dict(row,**{field:value})
                assert sample_key(updated)!=first
            assert sample_key(row)==first


def test_historical_reads_and_final_reference_context_are_batched(tmp_path):
    config=setup(tmp_path)
    with FaceDatabase(config.database) as state:
        selected=[]
        for i in range(260):
            identifier,_=scan(state,config,str(i+1))
            selected.append(dict(scene=str(i+1),scan=identifier))
        statements=[]
        state.connection.set_trace_callback(statements.append)
        with sqlite3.connect(':memory:') as replay:
            replay.row_factory=sqlite3.Row
            _copy_selection(state,replay,selected,'scope','ns')
        historical=[sql for sql in statements if 'm.person FROM members m' in sql]
        assert len(historical)==2
        statements.clear()
        rebuild_selected(state,selected[:2],config)
        # One set-based finalization after current roles have been inserted;
        # no per-observation or per-scene context rewrites.
        assert sum(sql.startswith('UPDATE candidates') for sql in statements) == 1
        assert all(json.loads(r[0])['reference_context'] for r in state.connection.execute('SELECT payload FROM candidates'))
        state.connection.set_trace_callback(None)


def test_private_addresses_follow_authoritative_payload_and_rollback():
    db=sqlite3.connect(':memory:')
    db.execute('CREATE TABLE candidates(id TEXT PRIMARY KEY,payload TEXT)')
    with addresses(db):
        db.execute('INSERT INTO candidates VALUES (?,?)',('a',json.dumps(dict(observation='one'))))
        sql='SELECT '+address(db)+' FROM candidates'
        assert db.execute(sql).fetchone()[0]=='one'
        db.execute('UPDATE candidates SET payload=?',(json.dumps(dict(observation='two')),))
        assert db.execute(sql).fetchone()[0]=='two'
        db.rollback()
        assert db.execute('SELECT * FROM candidate_addresses').fetchall()==[]
    assert not db.execute("SELECT 1 FROM sqlite_temp_master WHERE name LIKE 'candidate_address%'").fetchall()
    db.close()
