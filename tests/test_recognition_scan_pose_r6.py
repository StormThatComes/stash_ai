from dataclasses import replace
from types import SimpleNamespace
from threading import Event, get_ident
import copy
import numpy as np
import pytest
from PIL import Image


@pytest.mark.parametrize('refine',[False,True])
def test_detector_batches_equal_shapes_and_preserves_serial_geometry(monkeypatch,refine):
    torch=pytest.importorskip('torch')
    from stash_ai.recognition import sapiensid_backend as backend
    from stash_ai.recognition.config import defaults
    config=SimpleNamespace(data=defaults())
    config.data['models']['device']='cpu'
    config.data['sapiensid']['collection']['enabled']=True
    calls=[];owner=get_ident()
    class Pose:
        def predict(self,*,source,**kwargs):
            assert get_ident()==owner and kwargs['nms'] is True
            images=source if isinstance(source,list) else [source]
            calls.append([im.size for im in images])
            return [SimpleNamespace(boxes=SimpleNamespace(xyxy=torch.tensor([[2.,3.,30.,40.]]),
                conf=torch.tensor([.99 if im.getpixel((0,0))[0] else .01])),
                keypoints=SimpleNamespace(xy=torch.full((1,17,2),10.),conf=torch.ones((1,17)))) for im in images]
    monkeypatch.setattr(backend,'load_pose',lambda _:Pose())
    model=backend.SapiensIDFaces(config,{},load_identity=False,load_detector=False)
    if refine:
        def predict_many(frames):
            return [(np.full((len(bounds),17,2),10.,dtype='float32'),np.ones((len(bounds),17),dtype='float32')) for _,bounds in frames]
        model.pose_refiner=SimpleNamespace(predict_many=predict_many,predict=lambda image,bounds:predict_many([(image,bounds)])[0])
    images=[Image.new('RGB',size,color) for size,color in [((80,90),'white'),((100,120),'white'),((80,90),'black')]]
    try:
        expected=[]
        for image in images:
            model.detect(image,faces=[]);expected.append(copy.deepcopy(model.frame_people))
        calls.clear()
        model.detect_many([(image,[]) for image in images])
        assert calls==[[(80,90),(80,90)],[(100,120)]]
        actual=[]
        for image in images:
            model.detect(image,faces=[]);actual.append(model.frame_people)
        assert actual==expected and not model._prepared_frames
    finally:
        model.close()
        for image in images:image.close()


def test_pose_decode_overlaps_next_inference_and_preserves_order(monkeypatch):
    torch=pytest.importorskip('torch')
    from stash_ai.recognition.vitpose_pose import ViTPoseLarge
    owner=get_ident();started=Event();next_inference=Event();threads=[];calls=[]
    zeros=torch.zeros
    monkeypatch.setattr(torch,'zeros',lambda *args,**kw:zeros(*args,**dict(kw,device='cpu')))
    class Inputs(dict):
        def to(self,device):return self
    class Processor:
        def __call__(self,*,images,boxes,return_tensors):
            assert get_ident()==owner
            return Inputs(pixel_values=zeros(sum(map(len,boxes)),3,4,4))
        def post_process_pose_estimation(self,outputs,*,boxes,threshold):
            threads.append(get_ident());started.set()
            assert next_inference.wait(5), 'decode blocked the next inference'
            assert outputs.heatmaps.device.type=='cpu'
            return [[dict(keypoints=torch.tensor([[x+1,y+2]]*17),scores=torch.ones(17),labels=torch.arange(17))
                     for x,y,_,_ in group] for group in boxes]
    def model(**inputs):
        assert get_ident()==owner
        calls.append(1)
        if len(calls)==2:
            assert started.wait(5)
            next_inference.set()
        return SimpleNamespace(heatmaps=zeros(len(inputs['pixel_values']),17,4,4))
    # Device allocation is simulated; this verifies scheduling/ownership, not
    # accelerator throughput. All model and transfer calls stay on the owner.
    refiner=ViTPoseLarge(model,Processor(),device='cuda',batch_size=2)
    with Image.new('RGB',(100,100)) as image:
        result=refiner.predict_many([(image,[[0,0,10,10],[10,20,30,40]]),(image,[]),
                                    (image,[[30,40,50,60]])])
    assert len(calls)==2 and all(t!=owner for t in threads)
    np.testing.assert_array_equal(result[0][0][:,0],[[1,2],[11,22]])
    np.testing.assert_array_equal(result[2][0][:,0],[[31,42]])
    assert result[1][0].shape==(0,17,2)


def test_transferred_frames_are_not_copied_and_leases_outlive_microbatch(monkeypatch):
    from stash_ai.recognition.scan_pipeline import cue_batches
    image=Image.new('RGB',(10,10),'red')
    class Frames:
        transfers_images=True
        def __iter__(self):yield 0,None,image
    monkeypatch.setattr(image,'copy',lambda:pytest.fail('redundant frame copy'))
    batches=cue_batches(Frames(),1,100000)
    batch=next(batches);lease=batch.retain(image)
    assert batch[0][2] is image
    batches.close()
    assert image.getpixel((0,0))==(255,0,0)
    lease.close();lease.close()
    with pytest.raises(ValueError):image.getpixel((0,0))


def test_stopping_at_batch_boundary_releases_incoming_transferred_frame():
    from stash_ai.recognition.scan_pipeline import cue_batches
    images=[Image.new('RGB',(10,10)) for _ in range(2)]
    class Frames:
        transfers_images=True
        def __iter__(self):yield from ((i,None,image) for i,image in enumerate(images))
    batches=cue_batches(Frames(),1,100000)
    next(batches);batches.close()
    for image in images:
        with pytest.raises(ValueError):image.getpixel((0,0))


def test_deferred_byte_accounting_does_not_rescan_and_resets(tmp_path):
    from tests.test_recognition_scanning import config
    from stash_ai.recognition.person_collection import PersonCollector
    calls=[]
    class Metadata(dict):
        def __str__(self):calls.append(1);return super().__str__()
    collector=PersonCollector(config(tmp_path),{})
    collector.deferred=[]
    collector._write(None,[('id',Metadata(a=1),np.ones(8,dtype='float32'),b'crop')]*20)
    expected=collector.pending_bytes()
    for _ in range(100):assert collector.pending_bytes()==expected
    assert len(calls)==20 and expected>20*256
    assert len(collector.take_records())==20 and collector.pending_bytes()==0
    collector.deferred=[];collector._write(None,[('id',{},None,None)])
    collector.abort()
    assert collector.pending_bytes()==0 and collector.deferred is None


def test_collector_releases_shared_frame_after_failed_batched_pose(tmp_path):
    from tests.test_recognition_scanning import config
    from stash_ai.recognition.config import plain
    from stash_ai.recognition.person_collection import PersonCollector
    from stash_ai.recognition.frame_ownership import CueBatch
    settings=config(tmp_path);data=plain(settings.data);data['sapiensid']['enabled']=True
    data['sapiensid']['pose']['batch_size']=4
    model=SimpleNamespace(detect_many=lambda _:(_ for _ in ()).throw(RuntimeError('bad pose')),close=lambda:None)
    collector=PersonCollector(replace(settings,data=data),{},factory=lambda *args:model)
    image=Image.new('RGB',(10,10));batch=CueBatch();batch.add(0,None,image)
    collector.collect_records(None,'scan',0,image,[],frame_owner=batch)
    assert collector.pose_frames[0][3] is image
    batch.close()
    assert image.getpixel((0,0))==(0,0,0)
    with pytest.raises(RuntimeError,match='bad pose'):collector.flush()
    assert collector.pending_bytes()==0 and not collector.pose_frames
    with pytest.raises(ValueError):image.getpixel((0,0))
