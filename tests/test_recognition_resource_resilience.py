"""Resource failures cannot abort an otherwise valid solver search or waste inference."""
from types import SimpleNamespace
import numpy as np
import pytest

from stash_ai.recognition.resource_errors import ResourceLimit
from stash_ai.recognition.solver_shared import Arrays, scope, numerical, key
from stash_ai.recognition.merge_solver_feasibility import resource_limit


def test_typed_limit_survives_message_changes_and_pickling():
    import pickle
    error = ResourceLimit('Cross-scene provisional candidates exceed max_people_per_lookup.')
    assert resource_limit(pickle.loads(pickle.dumps(error)))
    assert not resource_limit(ValueError('Reference plan exceeds sapiensid.execution.plan_bytes.'))


@pytest.mark.parametrize('failure', ['save', 'replace', 'corrupt', 'ledger'])
def test_optional_cache_failure_preserves_exact_computation(tmp_path, monkeypatch, failure):
    assert Arrays.prepare(tmp_path)
    data = np.arange(16, dtype='float64')
    signature = key('test', (data,))
    def broken(*a, **kw):
        raise OSError('disk unavailable')
    if failure == 'save': monkeypatch.setattr(np, 'save', broken)
    if failure == 'replace': monkeypatch.setattr('os.replace', broken)
    if failure == 'corrupt': (tmp_path/(signature+'.npy')).write_bytes(b'partial file')
    if failure == 'ledger':
        import sqlite3
        with sqlite3.connect(tmp_path/'ledger.sqlite') as db: db.execute('DROP TABLE entries')
    with scope(tmp_path, 10000) as shared:
        assert np.array_equal(numerical('test', (data,), lambda: data*3), data*3)
        assert shared.disabled
        if failure != 'ledger':
            assert shared.db.execute('SELECT COUNT(*) FROM entries').fetchone()[0] == 0
    assert not list(tmp_path.glob('*.tmp'))


def test_cache_interrupt_propagates_and_releases_reservation(tmp_path, monkeypatch):
    Arrays.prepare(tmp_path)
    def stop(*a, **kw): raise KeyboardInterrupt()
    monkeypatch.setattr(np, 'save', stop)
    with scope(tmp_path, 10000) as shared:
        with pytest.raises(KeyboardInterrupt): numerical('test', (np.ones(8),), lambda: np.ones(8))
        assert shared.db.execute('SELECT COUNT(*) FROM entries').fetchone()[0] == 0
    assert not list(tmp_path.glob('*.tmp'))


def test_cache_unavailable_directory_is_optional(tmp_path):
    path = tmp_path/'file'; path.write_text('occupied')
    assert not Arrays.prepare(path)
    with scope(path, 10000):
        assert numerical('test', (np.ones(2),), lambda: 7) == 7


def test_prefetch_reserves_current_and_pending_before_worker_allocation(tmp_path):
    from stash_ai.recognition.replay_memory import Lookahead, memory_replay, release_vectors
    from stash_ai.recognition.run_cache import scope as cache_scope
    from tests.test_recognition_ram_replay import enabled, setup
    config = enabled(setup(tmp_path), rebuild_prefetch_scenes=True, rebuild_memory_bytes=4*1024*1024)
    blob=np.zeros(1024, dtype='<f4').tobytes()
    with memory_replay(config) as (db, uri), cache_scope(config) as cache:
        db.execute('CREATE TABLE observations(id TEXT,scan TEXT,dimensions INTEGER,embedding BLOB,rejected TEXT,cue INTEGER)')
        db.executemany('INSERT INTO observations VALUES (?,?,1024,?,NULL,0)',
            ((s+str(i),s,blob) for s in ('a','b') for i in range(60)))
        db.commit()
        with Lookahead(uri,config) as ahead:
            ahead.submit('a'); ahead.take('a')
            assert cache.reserved_size == 60*(2*len(blob)+1024)
            with pytest.raises(ResourceLimit): ahead.submit('b')
            assert ahead.pending is None
            release_vectors()
            assert cache.reserved_size == 0
            ahead.submit('b'); ahead.take('b')
        assert cache.reserved_size == 0


@pytest.mark.parametrize('reason', ['person_sharpness', None])
def test_rejected_body_with_accepted_face_skips_encoder_but_keeps_metadata(tmp_path,monkeypatch,reason):
    from stash_ai.recognition.body_collection import collect
    from tests.test_recognition_body_collection import settings
    config=settings(tmp_path); person=dict(bounds=[0,0,100,100],keypoints=[[10,10]]*17,score=1)
    monkeypatch.setattr('stash_ai.recognition.sapiensid_backend.association_choices',lambda *a:[[0]])
    monkeypatch.setattr('stash_ai.recognition.body_collection.quality',lambda *a:({'sharpness':0},reason))
    calls=[];items=[]
    model=SimpleNamespace(frame_people=[person],face_matches=[0],person_geometry=lambda *a:calls.append(a) or 'input')
    pipeline=SimpleNamespace(collect_items=lambda db,image,rows:items.extend(rows))
    collect(pipeline,None,model,config,'00000000-0000-0000-0000-000000000001',0,0,None,[dict(id='face',rejected=None)],[object()])
    assert len(calls)==int(reason is None)
    record,value=items[0]
    assert record[1]['identity_rejection']==reason and record[1]['face_status']=='accepted'
    assert value==('input' if reason is None else None)
