import base64
from contextlib import closing
import sqlite3
from types import SimpleNamespace

import numpy as np

from stash_ai.recognition.derived_cache import Pools
from stash_ai.recognition.config import defaults


def store():
    db=sqlite3.connect(':memory:');db.execute('CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT)')
    return db,Pools(db,256*1024**2)


def test_thousand_binary_centroids_survive_ordered_warm_pass():
    db,cache=store()
    with closing(db):
        mean=base64.b64encode(np.zeros(4096,dtype='<f8').tobytes()).decode()
        for i in range(1000):cache.put('body-owner-means',str(i),dict(modes=[[str(i)]],means=[mean]))
        assert sum(cache.get('body-owner-means',str(i)) is not None for i in range(1000))==1000
        cache.put('strict-scene','scene',{'kept':True})
        assert cache.store('strict-scene') is not cache.store('body-appearance-chunk')
        assert cache.get('strict-scene','scene')=={'kept':True}


def test_128_witness_bank_is_chunked_and_corrupt_chunk_falls_back(tmp_path,monkeypatch):
    from stash_ai.recognition import body_mode_cache
    from stash_ai.recognition.disk_body_modes import Matrix,Records
    db,cache=store();monkeypatch.setattr(body_mode_cache,'current',lambda:cache)
    config=SimpleNamespace(data=defaults())
    with closing(db):
        raw=np.random.default_rng(7).normal(size=(128,4096)).astype('<f4').tobytes()
        values=tmp_path/'values';values.write_bytes(raw)
        weights=tmp_path/'weights';weights.write_bytes(bytes([1])*128)
        original=Records();loaded=Records()
        try:
            body_mode_cache.save(original,'bank',[Matrix(values,128,weights_path=weights)],128,[],config)
            found=body_mode_cache.load(loaded,'bank',config)
            assert found is not None and len(found[0][0])==128
            assert np.array_equal(found[0][0][:],np.frombuffer(raw,dtype='<f4').reshape(128,4096))
            manifest=cache.get('body-appearances','bank')
            key=manifest['modes'][0][1][0][0]
            owner=cache.store('body-appearance-chunk')
            db.execute('DELETE FROM metadata WHERE key=?',(owner.prefix+'body-appearance-chunk:'+key,))
            assert body_mode_cache.load(loaded,'bank',config) is None
            assert not (loaded.path/'0.bin').exists()
        finally:original.close();loaded.close()
