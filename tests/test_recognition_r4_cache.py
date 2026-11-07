import sqlite3
from contextlib import closing
from types import SimpleNamespace
import numpy as np
from stash_ai.recognition.inference_cache import pack,unpack,Store
from stash_ai.recognition.media_cache import MediaCache


def test_binary_embeddings_and_stage_isolation(tmp_path):
    vector=np.arange(4096,dtype=np.float32)
    payload=pack((vector,{'x':vector[::2]}))
    assert len(payload)<vector.nbytes*1.5+512
    np.testing.assert_array_equal(unpack(payload)[1]['x'],vector[::2])
    with closing(Store(tmp_path/'cache',1024**2)) as store:
        store.put('small',{'pose':True},'pose')
        for i in range(50):store.put(str(i),vector,'body_embedding')
        assert store.get('small')=={'pose':True}
        assert store.by_stage['body_embedding']<=int(store.limit*.16)


def test_media_accounting_survives_replacement_and_discard(tmp_path):
    cache=MediaCache(tmp_path/'faces',10,SimpleNamespace(url='local'))
    for key,data in [('a',b'1234'),('a',b'12'),('b',b'123456789')]:
        cache.put(key,'"tag"',data)
    cache.discard('a')
    with closing(sqlite3.connect(cache.path)) as db:
        assert db.execute('SELECT * FROM usage').fetchone()==(9,1)
        assert db.execute('SELECT SUM(length(body)) FROM media').fetchone()==(9,)
