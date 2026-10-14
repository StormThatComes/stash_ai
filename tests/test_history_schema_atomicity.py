"""Schema/version transitions publish together and retain the primary failure."""
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import subprocess
import sys

import pytest
from stash_ai.history.store import HistoryStore, APPLICATION_ID, FORMAT

BOUNDARIES = [
    'CREATE TABLE IF NOT EXISTS runs',
    'ALTER TABLE runs ADD COLUMN last_progress_at',
    'ALTER TABLE runs ADD COLUMN progress_seq',
    'ALTER TABLE runs ADD COLUMN workload',
    'CREATE INDEX IF NOT EXISTS runs_scope_started',
    'CREATE TABLE IF NOT EXISTS reuse_decisions',
    'CREATE INDEX IF NOT EXISTS reuse_scope_scene',
    'CREATE TABLE IF NOT EXISTS reuse_proofs',
    'PRAGMA application_id=', 'PRAGMA user_version=',
]


def seed(path, version):
    if not version: return
    with HistoryStore(path) as store:
        store.start('retained','scope','inference','now',{'fingerprint':'old'})
        store.put_proof('scope','1','wd','scene','signature',{'retained':True})
        store.record_decision('retained','scope','1','wd','reuse','finished',[],{})
    with sqlite3.connect(path) as db:
        for column in ('last_progress_at','progress_seq','workload'):
            db.execute('ALTER TABLE runs DROP COLUMN '+column)
        if version==1:
            db.execute('DROP TABLE reuse_decisions')
            db.execute('DROP TABLE reuse_proofs')
        db.execute(f'PRAGMA user_version={version}')


def snapshot(path):
    with sqlite3.connect(path) as db:
        return (db.execute('PRAGMA application_id').fetchone()[0],
                db.execute('PRAGMA user_version').fetchone()[0],tuple(db.iterdump()))


@pytest.mark.parametrize('version',[0,1,2])
@pytest.mark.parametrize('boundary',BOUNDARIES)
@pytest.mark.parametrize('after',[False,True])
def test_every_schema_boundary_rolls_back_and_can_be_retried(tmp_path,monkeypatch,version,boundary,after):
    path=tmp_path/'history.sqlite'
    seed(path,version)
    before=snapshot(path)
    original=sqlite3.connect
    primary=sqlite3.OperationalError('schema boundary')
    class Fault(sqlite3.Connection):
        def execute(self,sql,*args,**kwargs):
            matches=sql.startswith(boundary)
            if matches and not after: raise primary
            value=super().execute(sql,*args,**kwargs)
            if matches and after: raise primary
            return value
    with monkeypatch.context() as patch:
        patch.setattr(sqlite3,'connect',lambda *args,**kwargs: original(*args,**kwargs,factory=Fault))
        with pytest.raises(sqlite3.OperationalError) as raised: HistoryStore(path)
    assert raised.value is primary and snapshot(path)==before
    with HistoryStore(path) as store:
        assert store.connection.execute('PRAGMA user_version').fetchone()[0]==FORMAT
        if version:
            assert list(store.rows('scope'))[0]['id']=='retained'
        if version==2:
            assert store.get_proof('scope','1','wd','scene','signature')=={'retained':True}
            assert store.decisions('scope')[0]['run_id']=='retained'


@pytest.mark.parametrize('after',[False,True])
def test_commit_failure_leaves_a_coherent_old_or_complete_new_schema(tmp_path,monkeypatch,after):
    path=tmp_path/'history.sqlite'
    seed(path,2)
    before=snapshot(path)
    original=sqlite3.connect
    class Fault(sqlite3.Connection):
        def commit(self):
            if after: super().commit()
            raise sqlite3.OperationalError('commit boundary')
    with monkeypatch.context() as patch:
        patch.setattr(sqlite3,'connect',lambda *args,**kwargs: original(*args,**kwargs,factory=Fault))
        with pytest.raises(sqlite3.OperationalError,match='commit boundary'): HistoryStore(path)
    if not after: assert snapshot(path)==before
    with HistoryStore(path,readonly=True) as store:
        assert list(store.rows('scope'))[0]['id']=='retained'
        assert store.connection.execute('PRAGMA user_version').fetchone()[0]==(FORMAT if after else 2)


def test_rollback_and_close_failures_never_replace_schema_failure(tmp_path,monkeypatch):
    path=tmp_path/'history.sqlite'
    original=sqlite3.connect
    calls=[]
    primary=sqlite3.OperationalError('primary schema failure')
    class Fault(sqlite3.Connection):
        def execute(self,sql,*args,**kwargs):
            if sql.startswith('CREATE TABLE IF NOT EXISTS reuse_decisions'): raise primary
            return super().execute(sql,*args,**kwargs)
        def rollback(self):
            calls.append('rollback');super().rollback();raise OSError('rollback failed')
        def close(self):
            calls.append('close');super().close();raise OSError('close failed')
    with monkeypatch.context() as patch:
        patch.setattr(sqlite3,'connect',lambda *args,**kwargs: original(*args,**kwargs,factory=Fault))
        with pytest.raises(sqlite3.OperationalError) as raised: HistoryStore(path)
    assert raised.value is primary and calls==['rollback','close']
    assert len(primary.__notes__)==2
    assert snapshot(path)[:2]==(0,0)


def test_readonly_current_incomplete_shape_is_refused_and_known_missing_features_can_be_repaired(tmp_path):
    path=tmp_path/'history.sqlite'
    seed(path,2)
    with HistoryStore(path): pass
    with sqlite3.connect(path) as db: db.execute('DROP TABLE reuse_decisions')
    before=path.read_bytes()
    with pytest.raises(ValueError,match='schema'): HistoryStore(path,readonly=True)
    assert path.read_bytes()==before
    with HistoryStore(path) as store:
        assert list(store.rows('scope'))[0]['id']=='retained'
        assert store.get_proof('scope','1','wd','scene','signature')=={'retained':True}
    with sqlite3.connect(path) as db: db.execute('ALTER TABLE runs DROP COLUMN wall_seconds')
    before=path.read_bytes()
    for readonly in (False,True):
        with pytest.raises(ValueError,match='column shape'): HistoryStore(path,readonly=readonly)
        assert path.read_bytes()==before


def test_concurrent_initializers_serialize_before_inspection_and_preserve_rows(tmp_path):
    path=tmp_path/'history.sqlite'
    def initialize(index):
        with HistoryStore(path) as store: store.start(str(index),'scope','inference','now',{})
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(initialize,range(8)))
    with HistoryStore(path,readonly=True) as store:
        assert len(list(store.rows('scope')))==8
        assert store.connection.execute('PRAGMA application_id').fetchone()[0]==APPLICATION_ID


def test_process_exit_before_schema_commit_never_publishes_current_version(tmp_path):
    path=tmp_path/'history.sqlite'
    script='''import os,sqlite3,sys
from stash_ai.history.store import HistoryStore
original=sqlite3.connect
class Crash(sqlite3.Connection):
    def execute(self,sql,*args,**kwargs):
        result=super().execute(sql,*args,**kwargs)
        if sql.startswith('PRAGMA user_version='): os._exit(7)
        return result
sqlite3.connect=lambda *args,**kwargs: original(*args,**kwargs,factory=Crash)
HistoryStore(sys.argv[1])
'''
    result=subprocess.run([sys.executable,'-c',script,str(path)],capture_output=True,timeout=10)
    assert result.returncode==7 and snapshot(path)[:2]==(0,0)
    with HistoryStore(path): pass


def test_actual_optional_schema_failure_does_not_request_new_inference(pipeline,monkeypatch):
    from tests.helpers.runtime import runtime as app
    original=sqlite3.connect
    class Fault(sqlite3.Connection):
        def execute(self,sql,*args,**kwargs):
            if sql.startswith('CREATE TABLE IF NOT EXISTS reuse_decisions'):
                raise sqlite3.OperationalError('optional schema failure')
            return super().execute(sql,*args,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(sqlite3,'connect',lambda *args,**kwargs: original(*args,**kwargs,factory=Fault))
        assert app.main(pipeline.args)==0
    loaded,writes=len(pipeline.loaded),list(pipeline.stash.writes)
    assert app.main(pipeline.args)==0
    assert len(pipeline.loaded)==loaded and pipeline.stash.writes==writes
