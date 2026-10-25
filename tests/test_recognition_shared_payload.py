import pickle
from multiprocessing.shared_memory import SharedMemory
import pytest
from stash_ai.recognition.shared_payload import pack,unpack,release
from stash_ai.recognition.resource_plan import plan
from tests.test_recognition_clustering import setup


def test_process_descriptor_excludes_embedding_bytes_and_roundtrips():
    blob=b'\x01\x02\x03\x04'*100000
    rows=(('a','s',0,0.,'[]','{}',100000,blob),)
    packed,segment=pack(rows)
    name=segment.name
    try:
        assert len(pickle.dumps(packed)) < 1024
        with unpack(packed) as values:
            assert tuple(values)==rows
    finally:
        release(segment)
    with pytest.raises(FileNotFoundError):
        SharedMemory(name=name)


def test_cache_and_queue_share_one_admission_budget(tmp_path):
    result=plan(setup(tmp_path))
    assert result.cache+result.queue <= result.working


def test_borrowed_embeddings_are_readonly_shared_views():
    from stash_ai.recognition.shared_payload import borrowed
    packed,segment=pack((('x','s',0,0.,'[]','{}',1,b'1234'),))
    try:
        with borrowed(packed) as values:
            assert isinstance(values[0][-1],memoryview)
            assert values[0][-1].readonly and bytes(values[0][-1])==b'1234'
    finally:
        release(segment)


def test_idle_cache_memory_is_reclaimed_before_work_admission():
    from stash_ai.recognition.run_cache import Cache
    cache=Cache(2000,2000)
    cache.configure_working(4000,4000)
    cache.put(('vector','a'),b'x'*1500,1500)
    assert cache.size>2000
    cache.reserve('queue',3000)
    assert cache.size<=1000 and cache.limit==1000
    cache.reserve('queue',0)
    assert cache.limit==4000
