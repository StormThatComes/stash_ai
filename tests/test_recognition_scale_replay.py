from pathlib import Path
import numpy as np
import pytest
from stash_ai.recognition.config import defaults
from types import SimpleNamespace


def test_disk_replay_capacity_is_independent_from_ram_and_removed():
    from stash_ai.recognition.replay_memory import memory_replay
    c=SimpleNamespace(data=defaults());c.data['execution']['rebuild_memory_bytes']=1048576
    with memory_replay(c) as (db,uri):
        path=Path(db.execute('PRAGMA database_list').fetchone()[2])
        assert path.exists()
        db.execute('CREATE TABLE data(value BLOB)')
        db.execute('INSERT INTO data VALUES (zeroblob(4194304))');db.commit()
        assert db.execute('SELECT length(value) FROM data').fetchone()[0]==4194304
        assert db.execute('PRAGMA temp_store').fetchone()[0]==1
    assert not path.exists()


def test_ten_thousand_scene_plans_fit_one_scene_memory_budget():
    from stash_ai.recognition.body_plan_spool import Plans
    with Plans(2048,16*1024**2) as plans:
        for i in range(10000):plans.append((dict(id=str(i)),[dict(id='sample')],{'sample':{'person':None,'status':'withheld'}}))
        assert sum(1 for _ in plans)==10000
        assert sum(1 for _ in plans)==10000


def test_paged_references_keep_every_vector_and_witness():
    from stash_ai.recognition.paged_references import scope,pack,rows,cues,Samples,Rows
    config=SimpleNamespace(data=defaults())
    vectors=[np.eye(2,dtype='<f4')[i%2] for i in range(1000)]
    with scope(config):
        samples=pack(vectors);assert isinstance(samples,Samples)
        assert all(np.array_equal(a,b) for a,b in zip(samples,vectors))
        records=rows(dict(id=str(i),scan='s',cue=i) for i in range(10000))
        assert isinstance(records,Rows) and len(records)==10000
        assert records[-1]['id']=='9999'
        a,b=cues(),cues();a.add(('s',1));b.add(('s',2))
        assert a.isdisjoint(b)
        b.add(('s',1));assert not a.isdisjoint(b)


@pytest.mark.parametrize('journal', ['wal','delete'])
def test_live_replay_keeps_committed_vectors_readable_during_writer_spill(tmp_path,journal):
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.rebuild import _copy_selection
    from stash_ai.recognition.replay_memory import memory_replay
    from tests.test_recognition_clustering import setup,scan
    c=setup(tmp_path)
    with FaceDatabase(c.database) as state:
        sid,_=scan(state,c,'1')
        state.connection.commit()
        assert state.connection.execute('PRAGMA journal_mode='+journal).fetchone()[0]==journal
        state.connection.execute('PRAGMA cache_size=4')
        state.connection.execute('BEGIN IMMEDIATE')
        selected=[dict(scan=sid,scene='1')]
        with memory_replay(c) as (target,_):
            _copy_selection(state,target,selected,'scope','ns')
            private_bytes = target.execute('SELECT SUM(length(embedding)) FROM main.observations').fetchone()[0]
            assert (private_bytes is None) == (journal=='wal')
            state.connection.executemany('INSERT INTO metadata(key,value) VALUES (?,?)',
                                         ((f'pressure-{n}','x'*4096) for n in range(128)))
            assert target.execute('SELECT SUM(length(embedding)) FROM observations').fetchone()[0]>0
        state.connection.execute("UPDATE observations SET quality=json_set(quality,'$.edited',1)")
        with memory_replay(c) as (target,_):
            _copy_selection(state,target,selected,'scope','ns')
            assert target.execute('SELECT SUM(length(embedding)) FROM main.observations').fetchone()[0]>0


def test_disk_cleanup_preserves_primary_failure_and_finalizes_cursors(monkeypatch):
    import pytest
    from stash_ai.recognition import disk_replay
    from stash_ai.recognition.replay_memory import memory_replay
    connections=[]
    class Connection(disk_replay.Connection):
        def close(self):
            super().close();connections.append(self)
            raise OSError('disk close failed')
    monkeypatch.setattr(disk_replay,'Connection',Connection)
    primary=RuntimeError('primary rebuild failure')
    with pytest.raises(RuntimeError) as caught:
        with memory_replay(SimpleNamespace(data=defaults())) as (db,_):
            cursor=db.execute('SELECT 1 UNION ALL SELECT 2');next(cursor)
            raise primary
    assert caught.value is primary and any('disk close failed' in n for n in primary.__notes__)
    assert len(connections)==1
    with pytest.raises(Exception,match='closed'):next(cursor)


def test_warm_paged_score_caches_do_not_reread_vectors(tmp_path):
    from stash_ai.recognition.paged_references import scope,pack
    from stash_ai.recognition.run_cache import scope as cache_scope
    from stash_ai.recognition.scoring import _prepared
    from stash_ai.recognition.evidence_cache import Evidence
    from stash_ai.recognition.clustering import Group
    from tests.test_recognition_clustering import setup
    config=setup(tmp_path);query=np.array([1.,0.],dtype='<f4')
    with cache_scope(config),scope(config):
        samples=pack([query]);group=Group('person',samples,True)
        evidence=Evidence(None,config)
        first=_prepared([group],query);scores=evidence.scores(group,query,'observation')
        reads=[];samples.pool.db.set_trace_callback(reads.append)
        assert all(a is b for a,b in zip(_prepared([group],query),first))
        assert evidence.scores(group,query,'observation')==scores
        assert not reads
