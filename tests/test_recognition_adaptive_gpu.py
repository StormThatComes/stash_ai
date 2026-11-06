from contextlib import nullcontext
from types import SimpleNamespace
import pytest
import numpy as np

torch=pytest.importorskip('torch')
from stash_ai.recognition.adaptive_gpu import Batches
from stash_ai.recognition.gpu_transfers import Transfers,to_device


def test_oom_splits_once_keeps_order_and_remembers_capacity(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:False)
    calls=[]
    def calculate(values):
        calls.append(list(values))
        if len(values)>2:raise torch.cuda.OutOfMemoryError('synthetic')
        return [v*3 for v in values]
    batch=Batches(8,'test')
    assert batch.run(list(range(7)),calculate,'cuda:0')==list(range(0,21,3))
    assert calls[0]==list(range(7)) and batch.maximum==2 and batch.retries==2
    calls.clear();assert batch.run([10,11],calculate,'cuda:0')==[30,33]
    assert calls==[[10,11]]


@pytest.mark.parametrize('device,error',[('cpu',torch.cuda.OutOfMemoryError('host')),('cuda',RuntimeError('bad kernel'))])
def test_nonrecoverable_errors_are_not_retried(device,error):
    calls=[]
    def calculate(values):calls.append(values);raise error
    with pytest.raises(type(error),match=str(error)):
        Batches(4,'test').run([1,2],calculate,device)
    assert len(calls)==1


def test_single_item_oom_fails_instead_of_skipping(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:False)
    def calculate(values):raise torch.cuda.OutOfMemoryError('single')
    with pytest.raises(torch.cuda.OutOfMemoryError,match='single') as caught:
        Batches(4,'test').run([1],calculate,'cuda')
    assert 'batch_size=1' in caught.value.__notes__[0]


def test_pinned_transfer_and_event_wait_precede_cpu_consumption(monkeypatch):
    calls=[]
    class Cpu:
        device=SimpleNamespace(type='cpu')
        def is_pinned(self):return False
        def pin_memory(self):calls.append('pin');return self
        def to(self,device,**kw):calls.append(('to',device,kw));return self
        def copy_(self,value,**kw):calls.append(('copy',kw));return self
    class Source:
        device=SimpleNamespace(type='cuda')
        def detach(self):return self
        def record_stream(self,stream):calls.append('record source')
    class Event:
        def record(self,stream):calls.append('event')
        def synchronize(self):calls.append('wait')
    stream=SimpleNamespace(wait_stream=lambda other:calls.append('wait compute'))
    monkeypatch.setattr(torch.cuda,'current_stream',lambda device:'compute')
    monkeypatch.setattr(torch.cuda,'Stream',lambda **kwargs:stream)
    monkeypatch.setattr(torch.cuda,'stream',lambda stream:nullcontext())
    monkeypatch.setattr(torch.cuda,'Event',Event)
    monkeypatch.setattr(torch,'empty_like',lambda value,**kw:Cpu())
    host=Cpu();assert to_device(host,'cuda:0') is host
    assert calls==['pin',('to','cuda:0',{'non_blocking':True})]
    transfers=Transfers('cuda:0');source=Source();readback=transfers.read(source)
    assert readback.source is source and 'wait' not in calls
    assert readback.wait() is readback.cpu and readback.source is None
    assert calls[-1]=='wait'
    transfers.read(Source());transfers.close()
    assert calls.count('wait')==2 and transfers.pending==[]


def test_rf_detr_retries_only_failed_batch_and_keeps_frame_order(monkeypatch):
    from stash_ai.recognition.rfdetr_detector import RFDETRDetector
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:False)
    adapter=RFDETRDetector(None,device='cuda',batch_size=4)
    calls=[]
    def forward(images):
        calls.append([i.index for i in images])
        if len(images)>2:raise torch.cuda.OutOfMemoryError('synthetic')
        return [i.index for i in images]
    adapter._forward=forward
    adapter._decode=lambda outputs,*args,**kwargs:outputs
    images=[SimpleNamespace(index=i,size=(20,20)) for i in range(5)]
    assert adapter.predict(images,conf=.5,iou=.5,max_det=2)==list(range(5))
    assert calls==[[0,1,2,3],[0,1],[2,3],[4]]
    assert adapter.batches.maximum==2


def test_vitpose_retry_keeps_person_and_frame_coordinates(monkeypatch):
    from stash_ai.recognition.vitpose_pose import ViTPoseLarge
    from stash_ai.recognition import gpu_transfers
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:False)
    monkeypatch.setattr(gpu_transfers,'to_device',lambda value,device:value)
    zeros=torch.zeros
    monkeypatch.setattr(torch,'zeros',lambda *a,**kw:zeros(*a,**dict(kw,device='cpu')))
    calls=[]
    class Processor:
        def __call__(self,images,boxes,return_tensors):
            return {'pixel_values':torch.zeros(sum(map(len,boxes)),3,4,4)}
        def post_process_pose_estimation(self,outputs,boxes,threshold):
            return [[dict(keypoints=torch.tensor([[x,y]]*17),scores=torch.ones(17),labels=torch.arange(17))
                    for x,y,_,_ in group] for group in boxes]
    class Model:
        def __call__(self,pixel_values,dataset_index):
            calls.append(len(pixel_values))
            if len(pixel_values)>2:raise torch.cuda.OutOfMemoryError('synthetic')
            return SimpleNamespace(heatmaps=torch.zeros(len(pixel_values),17,4,4))
    adapter=ViTPoseLarge(Model(),Processor(),device='cuda',batch_size=4)
    try:
        output=adapter.predict_many([(None,[[1,2,11,12],[3,4,13,14]]),(None,[[5,6,15,16],[7,8,17,18]])])
        assert calls==[4,2,2]
        np.testing.assert_array_equal(output[0][0][:,0],[[1,2],[3,4]])
        np.testing.assert_array_equal(output[1][0][:,0],[[5,6],[7,8]])
    finally:adapter.close()


def test_sapiens_prepared_retry_keeps_input_order(monkeypatch):
    from threading import get_ident
    from stash_ai.recognition.sapiensid_inference import SapiensIDV1Encoder
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:False)
    encoder=SapiensIDV1Encoder.__new__(SapiensIDV1Encoder)
    encoder.owner=get_ident();encoder.model=object();encoder.device='cuda'
    encoder.max_batch_size=4;encoder.cache_contract=None;encoder.batches=Batches(4,'SapiensID')
    calls=[]
    def calculate(values):
        calls.append(list(values))
        if len(values)>2:raise torch.cuda.OutOfMemoryError('synthetic')
        return [v+10 for v in values]
    encoder._embed_prepared=calculate
    assert encoder.embed_prepared([1,2,3,4])==[11,12,13,14]
    assert calls==[[1,2,3,4],[1,2],[3,4]]
