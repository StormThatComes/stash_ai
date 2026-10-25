from dataclasses import replace
from contextlib import closing
import sqlite3
from concurrent.futures import wait
import numpy as np
import pytest
from PIL import Image
from tests.helpers.recognition import MediaClient,Backend,Response
from tests.test_recognition_scanning import config
from tests.test_recognition_person_collection import Person
from stash_ai.recognition.config import plain
from stash_ai.recognition.media import SpriteInput,bounded_media
from stash_ai.recognition.media_cache import MediaCache
from stash_ai.recognition.scan_pipeline import CpuStage
from stash_ai.recognition.person_collection import PersonCollector
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.scanning import scan_scene


def test_finished_cpu_output_stays_reserved_until_consumed():
    stage=CpuStage(2,10**7)
    with Image.new('RGB',(8,8)) as image:
        try:
            future=stage.submit(lambda image,value:bytes(65536),image,None,extra=65536)
            wait([future],timeout=5)
            assert future.done() and stage.bytes>=65536 and future in stage.pending
            assert len(future.result())==65536
            assert stage.bytes==0 and not stage.pending
        finally:stage.close()


def test_grouped_face_and_body_checkpoint_is_atomic(tmp_path):
    settings=config(tmp_path);data=plain(settings.data)
    data['sapiensid']['enabled']=True;data['execution']['checkpoint_cues']=3
    settings=replace(settings,data=data);client=MediaClient();model=Person();model.fail=True
    collector=PersonCollector(settings,{},factory=lambda *a:model)
    try:
        with FaceDatabase(settings.database) as state,SpriteInput(client,client.scene,settings) as source:
            with pytest.raises(RuntimeError,match='broken'):
                scan_scene(state,'scope',client.scene,settings,'ns',source,Backend,on_cue=collector.collect_frame)
            db=state.connection
            assert db.execute('SELECT cursor FROM scans').fetchone()[0]==0
            assert db.execute('SELECT COUNT(*) FROM observations').fetchone()[0]==0
            assert db.execute('SELECT COUNT(*) FROM person_evidence').fetchone()[0]==0
            model.fail=False
            scan_scene(state,'scope',client.scene,settings,'ns',source,Backend,on_cue=collector.collect_frame)
            assert db.execute('SELECT COUNT(*) FROM observations').fetchone()[0]==3
            assert db.execute('SELECT COUNT(*) FROM person_evidence').fetchone()[0]==3
    finally:collector.close()


def test_durable_http_cache_revalidates_and_rejects_corrupt_body(tmp_path):
    client=MediaClient();calls=[];cache=MediaCache(tmp_path/'faces.sqlite',100,client)
    url=client.url+'sprite.png'
    def get(url,**kwargs):
        headers=kwargs.get('headers',{});calls.append(headers)
        response=Response(b'pixels')
        response.headers={'ETag':'"revision-1"'}
        response.status_code=304 if headers.get('If-None-Match')=='"revision-1"' else 200
        return response
    client.media.get=get
    assert bounded_media(client,url,None,5,cache=cache)==b'pixels'
    cache=MediaCache(tmp_path/'faces.sqlite',100,client)
    assert bounded_media(client,url,None,5,cache=cache)==b'pixels'
    assert calls==[{}, {'If-None-Match':'"revision-1"'}]
    with closing(sqlite3.connect(cache.path)) as db,db:db.execute("UPDATE media SET body=x'00'")
    assert bounded_media(client,url,None,5,cache=cache)==b'pixels' and calls[-1]=={}
    cache.put(url+'2','"2"',bytes(99))
    assert cache.get(url) is None
    cache.put(url+'3','W/"weak"',b'a')
    assert cache.get(url+'3') is None


def test_next_person_batch_prepares_while_gpu_owner_infers(tmp_path,monkeypatch):
    from threading import Event,get_ident
    started=Event();next_ready=Event();owner=get_ident()
    def prepare(image,points):
        if points[0]>=2:
            assert started.wait(5)
            next_ready.set()
        return (np.zeros(1),np.zeros(1),np.ones(1))
    monkeypatch.setattr('stash_ai.recognition.sapiensid_v1.prepare_person_input',prepare)
    class Prepared(Person):
        def __init__(self):super().__init__();self.ordinal=0;self.batches=[]
        def person_input(self,image,face):
            index=self.ordinal;self.ordinal+=1
            return [0,0,image.width,image.height],[index]
        def embed_inputs(self,inputs):
            assert get_ident()==owner
            started.set();assert next_ready.wait(5)
            self.batches.append(len(inputs));return [np.ones(4096) for _ in inputs]
    settings=config(tmp_path);data=plain(settings.data)
    data['sapiensid']['enabled']=True;data['sapiensid']['execution']['batch_size']=2
    data['execution']['checkpoint_cues']=4;settings=replace(settings,data=data)
    client=MediaClient(count=4);model=Prepared();collector=PersonCollector(settings,{},factory=lambda *a:model)
    try:
        with FaceDatabase(settings.database) as state,SpriteInput(client,client.scene,settings) as source:
            scan_scene(state,'scope',client.scene,settings,'ns',source,Backend,on_cue=collector.collect_frame)
            assert model.batches==[2,2]
    finally:started.set();collector.close()
