from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import numpy as np
import pytest
from stash_ai.recognition.inference_cache import Store,many,using


def test_batches_freeze_shared_geometry_and_use_one_writer_operation(tmp_path):
    original={'points':np.arange(34,dtype='float32').reshape(17,2),'records':[{'x':1}]}
    with closing(Store(tmp_path/'cache',1024**2,queue_bytes=65536)) as store,using(store):
        calls=[];write=store._put_blobs
        def capture(entries,stage):calls.append(len(entries));return write(entries,stage)
        store._put_blobs=capture
        outputs=many('pose','v1',list(range(12)),lambda values:[original]*len(values))
        outputs[0]['points'][:]=0;outputs[0]['records'][0]['x']=5
        original['points'][:]=-1;original['records'][0]['x']=8
        assert outputs[1]['points'][0,1]==1 and outputs[1]['records'][0]['x']==1
        again=many('pose','v1',[1],lambda values:pytest.fail('cache miss'))
        assert again[0]['points'][0,1]==1 and again[0]['records'][0]['x']==1
        assert calls==[12] and store.queued_bytes<=store.queue_limit


def test_parallel_queue_admission_keeps_budget_and_replacement_accounting(tmp_path):
    with closing(Store(tmp_path/'cache',4*1024**2,queue_bytes=8192)) as store:
        def produce(i):
            store.enqueue_many([(str(i),np.full(64,i,dtype='float32'))], 'pose')
        with ThreadPoolExecutor(4) as pool:list(pool.map(produce,range(50)))
        assert store.queued_bytes<=store.queue_limit
        assert len(store.get_many(list(map(str,range(50)))))==50
        store.put('1',np.ones(10),'pose')
        assert store.used==store.db.execute('SELECT SUM(size) FROM entries').fetchone()[0]
        assert store.by_stage==dict(store.db.execute('SELECT stage,SUM(size) FROM entries GROUP BY stage'))


def test_sqlite_full_recounts_rolled_back_batch_and_remains_usable(tmp_path):
    with closing(Store(tmp_path/'cache',1024**2,queue_bytes=65536)) as store:
        store.put('committed',{'a':1},'pose');store.db.commit()
        real=store.db
        class FullOnce:
            fail=True
            def __getattr__(self,name):return getattr(real,name)
            def executemany(self,sql,rows):
                if self.fail and sql.startswith('INSERT'):
                    self.fail=False
                    error=sqlite3.OperationalError('database or disk is full')
                    error.sqlite_errorcode=sqlite3.SQLITE_FULL
                    raise error
                return real.executemany(sql,rows)
        store.db=FullOnce()
        store.enqueue_many([(str(i),{'x':i}) for i in range(8)],'pose')
        assert store.get('0') is None
        assert store.get('committed')=={'a':1}
        assert store.used==real.execute('SELECT SUM(size) FROM entries').fetchone()[0]
        store.put('next',{'ok':True},'pose')
        assert store.get('next')=={'ok':True}


def test_writer_error_drains_all_reservations_on_close(tmp_path):
    store=Store(tmp_path/'cache',1024**2,queue_bytes=65536)
    def fail(*args):raise RuntimeError('write failed')
    store._put_blobs=fail
    store.enqueue_many([(str(i),{'a':i}) for i in range(5)],'pose')
    with pytest.raises(RuntimeError,match='write failed'):store.close()
    assert store.queued_bytes==0 and not store.queued
    with pytest.raises(sqlite3.ProgrammingError):store.db.execute('SELECT 1')


def test_sql_stream_fetches_blocks_and_closes_on_early_exit():
    from stash_ai.recognition.sql_stream import rows
    class Cursor:
        def __init__(self):self.next=0;self.calls=0;self.closed=False
        def fetchmany(self,size):
            self.calls+=1
            batch=list(range(self.next,min(1000,self.next+size)));self.next+=len(batch)
            return batch
        def close(self):self.closed=True
    class Database:
        def execute(self,*args):self.cursor=Cursor();return self.cursor
    db=Database()
    assert list(rows(db,'select'))==list(range(1000))
    assert db.cursor.calls==17 and db.cursor.closed
    stream=rows(db,'select');assert next(stream)==0;stream.close()
    assert db.cursor.calls==1 and db.cursor.closed
