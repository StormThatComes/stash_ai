from types import SimpleNamespace
import sqlite3
import hashlib
import numpy as np
import pytest


def test_validator_survives_body_eviction_but_actual_pixels_are_reverified(tmp_path,monkeypatch):
    from stash_ai.recognition.media_cache import MediaCache,Validator
    from stash_ai.recognition.media import bounded_media
    from tests.helpers.recognition import Response
    calls=[]
    def get(url,**kwargs):
        calls.append(kwargs.get('headers',{}))
        response=Response(b'original')
        response.headers={'ETag':'"version1"'}
        if kwargs.get('headers'):response.status_code=304
        return response
    client=SimpleNamespace(url='http://stash/',media=SimpleNamespace(get=get))
    cache=MediaCache(tmp_path/'faces',1024,client);url=client.url+'sheet'
    cache.put(url,'"version1"',b'original')
    with sqlite3.connect(cache.path) as db:db.execute('DELETE FROM media')
    proof=bounded_media(client,url,1024,5,cache=cache,verify_only=True)
    assert proof==Validator(hashlib.sha256(b'original').hexdigest(),8)
    assert len(calls)==1
    assert bounded_media(client,url,1024,5,cache=cache)==b'original'
    assert calls[-2:]==[{'If-None-Match':'"version1"'},{}]


def test_dynamic_scrfd_batches_and_static_graph_preserves_single_frame(tmp_path,monkeypatch):
    from stash_ai.recognition import scrfd
    from stash_ai.recognition.config import load
    from tests.test_recognition_scrfd import predictions
    calls=[]
    class Session:
        dynamic=True
        def get_inputs(self):return [SimpleNamespace(type='tensor(float)',shape=['n' if self.dynamic else 1,3,640,640],name='input')]
        def get_outputs(self):return [SimpleNamespace(type='tensor(float)',shape=['n' if self.dynamic else 1,1,1],name=str(i)) for i in range(9)]
        def get_providers(self):return ['CPUExecutionProvider']
        def run(self,names,feeds):
            count=len(feeds['input']);calls.append(count)
            return [np.repeat(row,count,axis=0) for row in predictions()]
    session=Session()
    monkeypatch.setattr(scrfd,'verified_bytes',lambda *args:b'verified')
    monkeypatch.setattr(scrfd,'session',lambda *args,**kwargs:(session,'cpu'))
    config=load(root=tmp_path,device='cpu');info=(SimpleNamespace(__version__='test'),[])
    for dynamic,expected in [(True,[3]),(False,[1,1,1])]:
        session.dynamic=dynamic;calls.clear()
        detector=scrfd.ScrfdDetector(config,{'scrfd_10g':tmp_path/'test'},runtime_info=info)
        try:assert detector.detect_many([np.zeros((80,160,3),dtype='uint8')]*3)==[[],[],[]]
        finally:detector.close()
        assert calls==expected
