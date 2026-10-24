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
