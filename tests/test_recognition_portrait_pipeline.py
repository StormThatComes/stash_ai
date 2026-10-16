from dataclasses import replace
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from stash_ai.recognition.config import plain
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.models import checked_detection
from stash_ai.recognition.names import plan_names
from stash_ai.recognition.portrait_pipeline import Pipeline,ModelProxy,current,Capacity
from stash_ai.recognition.visibility import CONTRACT_ID
from tests.helpers.recognition import Backend
from tests.test_recognition_vtt_portraits import setup,FACE
from tests.test_recognition_run import run


SECOND=checked_detection([30,100,100,100,55,135,105,135,80,150,61,170,99,170,.99])


def multi(tmp_path,monkeypatch,enabled=True):
    config,client,services,_=setup(tmp_path,monkeypatch)
    data=plain(config.data);data['execution'].update(prefetch_portraits=enabled,prefetch_scenes=False,prefetch_frames=0)
    config=replace(config,data=data)
    models=[];readers=[];model_calls=[]
    def backend(*args):
        model=Backend();model.detect=lambda image:[FACE,SECOND]
        model.embed=lambda image,face:np.array([1.,0,0,0]) if face.box[0]>150 else np.array([0.,1,0,0])
        return model
    class Visibility:
        def __init__(self,*args):models.append(self)
        def measure(self,image,face):
            model_calls.append((threading.get_ident(),threading.current_thread().name,face.box[0]))
            return dict(visible_face_fraction=.8,visibility=dict(contract=CONTRACT_ID))
        def close(self):pass
    coordinator=threading.get_ident()
    def reader():
        owner=threading.get_ident();entry=dict(owner=owner,closed=False,calls=[]);readers.append(entry)
        def checked():assert threading.get_ident()==owner
        def performers(ids):
            checked();entry['calls'].append(('performers',list(ids)))
            return {value:dict(client.profiles[value]) for value in ids}
        def scenes(ids):
            checked();entry['calls'].append(('scenes',list(ids)))
            return {value:client.scene(value) for value in ids}
        def get(*a,**k):
            # The existing deadline transport delegates HTTP to its bounded
            # helper; it remains an isolated reader with no coordinator session.
            assert threading.get_ident()!=coordinator
            return client.media.get(*a,**k)
        def close():checked();entry['closed']=True
        return SimpleNamespace(url=client.url,api_key=client.api_key,performers=performers,scenes=scenes,
                               media=SimpleNamespace(get=get),scene=lambda value:scenes([value])[value],close=close)
    client.new_media_reader=reader
    services.backend=backend;services.visibility=Visibility
    return config,client,services,readers,model_calls


def result(config):
    with FaceDatabase(config.database) as state:
        return {json.loads(row['metadata'])['geometry'][0]:(row['digest'],json.loads(row['metadata'])['score'])
                for row in state.connection.execute("SELECT * FROM portraits WHERE status='published'")}


def test_pipeline_reuses_one_scene_across_people_and_matches_serial(tmp_path,monkeypatch):
    serial=tmp_path/'serial';serial.mkdir()
    config,client,services,readers,calls=multi(serial,monkeypatch,False)
    assert run(config,services,'1')==0
    expected=result(config)
    parallel=tmp_path/'pipeline';parallel.mkdir()
    config,client,services,readers,calls=multi(parallel,monkeypatch,True)
    assert run(config,services,'1')==0
    assert result(config)==expected
    assert len(calls)==6 and len({tid for tid,_,_ in calls})==1
    assert all(name.startswith('portrait-models') for _,name,_ in calls)
    assert calls[:3][0][2]==calls[:3][-1][2] and calls[3][2]!=calls[0][2]
    assert all(reader['closed'] for reader in readers)
    assert any(kind=='performers' and len(ids)==2 for reader in readers for kind,ids in reader['calls'])
    assert client.media_calls.count('http://stash/sprite/1.jpg')==2
    assert current() is None


def test_manual_upcoming_image_does_not_request_quality_inference(tmp_path,monkeypatch):
    config,client,services,readers,calls=multi(tmp_path,monkeypatch)
    preserved=[]
    def plan(database,stash,report,config,names,scope,targets):
        performer=targets[-1]['performer'];preserved.append(performer)
        client.images[performer]=b'manual'
        client.profiles[performer]['image_path']=f'http://stash/performer/{performer}/image'
        return {}
    services.metadata_plan=plan
    assert run(config,services,'1')==0
    assert client.images[preserved[0]]==b'manual' and len(calls)==3
    assert all(reader['closed'] for reader in readers)


def test_prep_capacity_keeps_original_serial_source_path(tmp_path,monkeypatch):
    from stash_ai.recognition import portrait_pipeline
    config,client,services,readers,calls=multi(tmp_path,monkeypatch)
    monkeypatch.setattr(portrait_pipeline.SourceOwner,'thumbnail',lambda *a,**k:(_ for _ in ()).throw(Capacity('tiny')))
    assert run(config,services,'1')==0
    assert len(result(config))==2 and len(calls)==6
    assert all(reader['closed'] for reader in readers)


def test_next_quality_work_overlaps_publication_with_sqlite_on_coordinator(tmp_path,monkeypatch):
    config,client,services,readers,calls=multi(tmp_path,monkeypatch)
    prepared=threading.Event();sql_threads=[];coordinator=threading.get_ident()
    original_visibility=services.visibility
    class Observed(original_visibility):
        def measure(self,image,face):
            result=super().measure(image,face)
            if len(calls)>3:prepared.set()
            return result
    services.visibility=Observed
    def plan(database,*args):
        database.connection.set_trace_callback(lambda sql:sql_threads.append(threading.get_ident()))
        return {}
    services.metadata_plan=plan
    original=client.update_person;writes=[]
    def update(performer,**fields):
        if 'image' in fields and not writes:
            assert prepared.wait(3),'next target must prepare while current publication waits'
        if 'image' in fields:writes.append(performer)
        return original(performer,**fields)
    client.update_person=update
    assert run(config,services,'1')==0
    assert len(writes)==2 and set(sql_threads)=={coordinator}
    assert all(reader['closed'] for reader in readers)


def test_partial_pipeline_startup_restores_context_and_closes_created_pool(tmp_path,monkeypatch):
    from stash_ai.recognition import portrait_pipeline as module
    from tests.test_recognition_clustering import setup as config_at
    config=config_at(tmp_path);created=[]
    class Pool:
        def __init__(self,*args,**kwargs):
            if created:raise RuntimeError('preparer startup failed')
            created.append(self);self.closed=False
        def shutdown(self,**kwargs):self.closed=True
    monkeypatch.setattr(module,'ThreadPoolExecutor',Pool)
    pipeline=Pipeline(None,SimpleNamespace(new_media_reader=lambda:None),config,'scope',[],None,None,None)
    with pytest.raises(RuntimeError,match='preparer startup failed'):
        with pipeline:pass
    assert created[0].closed and current() is None


def test_model_image_copy_outlives_interrupted_consumer():
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    started,release=threading.Event(),threading.Event();seen=[]
    class Model:
        def measure(self,image,face):
            started.set();assert release.wait(3)
            seen.append(image.getpixel((0,0)));return {}
    with ThreadPoolExecutor(max_workers=1) as executor:
        class Future:
            def __init__(self,real):self.real=real
            def add_done_callback(self,callback):return self.real.add_done_callback(callback)
            def cancel(self):return self.real.cancel()
            def result(self):
                assert started.wait(3)
                raise KeyboardInterrupt('consumer stopped')
        class Pool:
            def submit(self,*args):return Future(executor.submit(*args))
        model=ModelProxy(Model(),Pool())
        try:
            with Image.new('RGB',(4,4),(1,2,3)) as image:
                with pytest.raises(KeyboardInterrupt,match='consumer stopped'):model.measure(image,None)
        finally:release.set()
    assert seen==[(1,2,3)]


def test_corrupt_future_baseline_does_not_fail_current_publication(tmp_path,monkeypatch):
    from stash_ai.recognition.publication import operation_id
    config,client,services,readers,calls=multi(tmp_path,monkeypatch)
    future=[]
    def plan(database,stash,report,config,names,scope,targets):
        target=targets[-1];future.append(target['performer'])
        client.images[target['performer']]=b'owned earlier image'
        client.profiles[target['performer']]['image_path']=f"http://stash/performer/{target['performer']}/image"
        with database.connection:
            database.connection.execute('INSERT INTO publications VALUES (?,?,?,?,?,?,?,?,?,?)',
                (operation_id(scope,'',target['person'],'image'),scope,'',target['person'],'image',target['performer'],
                 'corrupt-json','{}','confirmed',None))
        return {}
    services.metadata_plan=plan
    assert run(config,services,'1')==1
    assert len(result(config))==1
    assert client.images[future[0]]==b'owned earlier image'
    assert all(reader['closed'] for reader in readers)
