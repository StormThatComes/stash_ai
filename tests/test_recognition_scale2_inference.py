from dataclasses import replace
from types import SimpleNamespace
from threading import Event,get_ident
import numpy as np
from PIL import Image
from stash_ai.recognition.config import plain


def test_visibility_batches_across_cues_without_reassessing(tmp_path,monkeypatch):
    from tests.test_recognition_scanning import config
    from tests.helpers.recognition import MediaClient,Backend,Visibility
    from stash_ai.recognition import scanning
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.media import SpriteInput
    cfg=config(tmp_path);data=plain(cfg.data)
    data['execution']['recognizer_batch_size']=4
    data['detection']['min_visible_face_fraction']=.5
    cfg=replace(cfg,data=data)
    class Batched(Visibility):
        batches=[]
        def measure_many(self,items):
            self.batches.append(len(items))
            return [self.measure(image,face) for image,face in items]
    model=Batched();assess=scanning.assess;calls=[]
    def measured(*args):calls.append(1);return assess(*args)
    monkeypatch.setattr(scanning,'assess',measured)
    client=MediaClient()
    with FaceDatabase(cfg.database) as state,SpriteInput(client,client.scene,cfg) as source:
        scanning.scan_scene(state,'scope',client.scene,cfg,'ns',source,Backend,visibility=model)
        assert model.batches==[3] and len(calls)==3
        assert state.connection.execute('SELECT COUNT(*) FROM observations WHERE rejected IS NULL').fetchone()[0]==3


def test_vitpose_reuses_workers_and_prepares_next_batch_during_inference(monkeypatch):
    import torch
    from stash_ai.recognition.vitpose_pose import ViTPoseLarge
    started=Event();prepared=Event();owner=get_ident();threads=[];calls=[]
    zeros=torch.zeros
    monkeypatch.setattr(torch,'zeros',lambda *args,**kwargs:zeros(*args,**dict(kwargs,device='cpu')))
    class Inputs(dict):
        def to(self,device):return self
    class Processor:
        def __call__(self,**kwargs):
            threads.append(get_ident())
            if len(threads)==2:
                assert started.wait(5);prepared.set()
            return Inputs(pixel_values=zeros(1,3,8,8))
    def model(**kwargs):
        calls.append(1)
        if len(calls)==1:
            started.set();assert prepared.wait(5)
        return SimpleNamespace(heatmaps=zeros(1,17,2,2))
    adapter=ViTPoseLarge(model,Processor(),device='cuda',batch_size=1)
    adapter._decode=lambda *args:(np.zeros((1,17,2)),np.ones((1,17)),np.arange(17)[None,:])
    try:
        with Image.new('RGB',(32,32)) as image:
            adapter.predict(image,[[0,0,10,10],[10,10,20,20]])
            pools=adapter._preparer,adapter._decoder
            adapter.predict(image,[[0,0,10,10]])
            assert pools==(adapter._preparer,adapter._decoder)
        assert threads[0]==owner and threads[1]!=owner
    finally:adapter.close()
    assert adapter._preparer is None and adapter._decoder is None and adapter.model is None
