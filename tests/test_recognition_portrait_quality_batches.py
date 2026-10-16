from dataclasses import replace
from io import BytesIO
import threading

import numpy as np
from PIL import Image
import pytest

from stash_ai.recognition.config import plain
from stash_ai.recognition.visibility import FaceVisibility,LazyVisibility,VisibilityGeometryError,CONTRACT_ID,prepare,measurement,device_measurements,_constants
from stash_ai.recognition.portrait_threads import NativeBudget,encoding_workers,native_threads
from stash_ai.recognition.portrait_encoding import Encoder
from tests.test_recognition_clustering import setup
from tests.test_recognition_vtt_portraits import FACE


def test_fixed_preprocessing_constants_preserve_exact_arrays_and_masks():
    face=FACE
    pixels=np.random.default_rng(9).integers(0,256,(350,500,3),dtype=np.uint8)
    with Image.fromarray(pixels) as image:
        blob,region=prepare(image,face)
        x,y,w,h=face.box;side=max(w,h)*1.5
        left,top=x+w/2-side/2,y+h/2-side/2
        with image.convert('RGB') as rgb,rgb.transform((256,256),Image.Transform.AFFINE,
            (side/256,0,left,0,side/256,top),resample=Image.Resampling.BILINEAR) as crop:
            expected=np.asarray(crop,dtype=np.float32)/255.
        expected=(expected-np.array([.485,.456,.406],dtype=np.float32))/np.array([.229,.224,.225],dtype=np.float32)
        rows,columns=np.indices((256,256),dtype=np.float32)
        mask=(((columns+.5-128)/(w*.45*256/side))**2+((rows+.5-128)/(h*.48*256/side))**2<=1)
        assert np.array_equal(blob,np.ascontiguousarray(expected.transpose(2,0,1)[None]))
        assert np.array_equal(region,mask)
    assert all(not value.flags.writeable for value in _constants())
    with pytest.raises(ValueError):_constants()[0][0]=1


def test_device_integer_reduction_matches_numpy_and_transfers_only_counts(monkeypatch):
    import torch
    logits=np.random.default_rng(8).normal(size=(3,1,256,256)).astype('float32')
    logits[:,:,0]=0
    regions=np.indices((256,256)).sum(axis=0)%2==0
    masks=[regions,~regions,np.ones((256,256),dtype=bool)]
    copied=[];original=torch.Tensor.cpu
    def cpu(value,*args,**kwargs):copied.append(value.numel());return original(value,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'cpu',cpu)
    actual=device_measurements(torch.from_numpy(logits),masks,'cuda')
    expected=[measurement(logits[i:i+1],region,'cuda') for i,region in enumerate(masks)]
    assert actual==expected and copied==[3]
    with pytest.raises(ValueError,match='non-finite'):
        device_measurements(torch.full((3,1,256,256),float('nan')),masks,'cuda')
    with pytest.raises(ValueError,match='empty'):
        device_measurements(torch.from_numpy(logits),[np.zeros_like(regions)]*3,'cuda')


def test_visibility_batches_keep_geometry_local_and_equal_scalar_masks():
    import torch
    calls=[]
    class Model:
        def __call__(self,value):calls.append(len(value));return value[:,:1]
    model=object.__new__(FaceVisibility);model.model=Model();model.device='cpu'
    with Image.new('RGB',(500,350),(200,90,60)) as first,Image.new('RGB',(500,350),(20,50,90)) as second:
        expected=[model.measure(first,FACE),model.measure(second,FACE)]
        calls.clear()
        assert model.measure_many([(first,FACE),(second,FACE)])==expected and calls==[2]
        with Image.new('RGB',(8,8)) as invalid:
            values=model.measure_many([(invalid,FACE),(first,FACE)])
            assert isinstance(values[0],VisibilityGeometryError) and values[1]==expected[0]


def test_lazy_batch_failure_disables_batch_and_retains_scalar_recovery(tmp_path):
    calls=[];closed=[]
    class Model:
        def measure_many(self,items):calls.append('batch');raise RuntimeError('batch unsupported')
        def measure(self,image,face):calls.append('single');return dict(visible_face_fraction=.8,visibility=dict(contract=CONTRACT_ID))
        def close(self):closed.append(1)
    config=setup(tmp_path);model=LazyVisibility(config,{},lambda *a:Model())
    with Image.new('RGB',(500,350)) as image:
        assert all(isinstance(v,dict) for v in model.measure_many([(image,FACE)]*4))
        assert all(isinstance(v,dict) for v in model.measure_many([(image,FACE)]*4))
        assert calls==['batch']+['single']*8 and len(closed)==1
    model.close();assert len(closed)==2


def test_explicit_native_budget_is_scoped_and_default_is_lazy(tmp_path):
    import torch
    from threadpoolctl import threadpool_info
    config=setup(tmp_path);data=plain(config.data);data['execution']['portrait_cpu_threads']=4
    config=replace(config,data=data)
    previous=torch.get_num_threads();before={v['filepath']:v['num_threads'] for v in threadpool_info() if v['user_api']=='blas'}
    with NativeBudget(config):
        assert torch.get_num_threads()==native_threads(config)==1
        assert encoding_workers(config)==2
        assert all(v['num_threads']==1 for v in threadpool_info() if v['user_api']=='blas')
    assert torch.get_num_threads()==previous
    assert {v['filepath']:v['num_threads'] for v in threadpool_info() if v['user_api']=='blas'}==before


def test_jpeg_executor_preserves_bytes_and_closes_input_after_failure(tmp_path):
    from stash_ai.recognition.quality import portrait_bytes
    config=setup(tmp_path);policy=config.data['portraits']
    with Image.new('RGB',(500,350),(80,90,100)) as original:
        expected=portrait_bytes(original,FACE,policy,limit=1048576,details={})
        image=original.copy();owners=[]
        def encode(*args,**kwargs):owners.append(threading.current_thread().name);return portrait_bytes(*args,**kwargs)
        with Encoder(config,encode) as encoder:
            data,bounds,details=encoder.encode(image,FACE,policy,1048576)
        assert (data,bounds)==expected and owners[0].startswith('portrait-jpeg')
        with pytest.raises(ValueError):image.getpixel((0,0))
        image=original.copy()
        def fail(*args,**kwargs):raise RuntimeError('encoding failed')
        with Encoder(config,fail) as encoder:
            with pytest.raises(RuntimeError,match='encoding failed'):encoder.encode(image,FACE,policy,1048576)
        with pytest.raises(ValueError):image.getpixel((0,0))


def test_vtt_pipeline_uses_visibility_microbatches_without_new_scanning(tmp_path,monkeypatch):
    from tests.test_recognition_vtt_portraits import setup as vtt_setup
    from tests.test_recognition_run import run
    config,client,services,backends=vtt_setup(tmp_path,monkeypatch)
    calls=[]
    class Visibility:
        def __init__(self,*args):pass
        def measure(self,*args):calls.append(('single',1));return dict(visible_face_fraction=.8,visibility=dict(contract=CONTRACT_ID))
        def measure_many(self,items):
            calls.append(('batch',len(items)))
            assert threading.current_thread().name.startswith('portrait-models')
            return [dict(visible_face_fraction=.8,visibility=dict(contract=CONTRACT_ID)) for _ in items]
        def close(self):pass
    services.visibility=Visibility
    assert run(config,services,'1')==0 and calls==[('batch',3)]
    assert run(config,services,'1')==0 and calls==[('batch',3)] and len(backends)==1


def test_cpu_budget_one_keeps_portrait_inference_serial(tmp_path,monkeypatch):
    from tests.test_recognition_portrait_pipeline import multi,result
    from tests.test_recognition_run import run
    config,client,services,readers,calls=multi(tmp_path,monkeypatch)
    data=plain(config.data);data['execution']['portrait_cpu_threads']=1
    config=replace(config,data=data)
    assert run(config,services,'1')==0 and len(result(config))==2
    assert len(calls)==6 and {tid for tid,_,_ in calls}=={threading.get_ident()}
    assert not readers


def test_jpeg_startup_failure_closes_transferred_pixels(tmp_path,monkeypatch):
    from stash_ai.recognition import portrait_encoding
    config=setup(tmp_path)
    image=Image.new('RGB',(500,350))
    def fail(*args,**kwargs):raise RuntimeError('encoder startup')
    monkeypatch.setattr(portrait_encoding,'ThreadPoolExecutor',fail)
    with Encoder(config,lambda *args:None) as encoder:
        with pytest.raises(RuntimeError,match='encoder startup'):encoder.encode(image,FACE,config.data['portraits'],1048576)
    with pytest.raises(ValueError):image.getpixel((0,0))


def test_portrait_session_threads_are_explicit_and_identity_default_stays_one(tmp_path):
    from types import SimpleNamespace
    from stash_ai.recognition.config import load
    from stash_ai.recognition.runtime import session
    seen=[]
    class Options:pass
    def create(model,*,sess_options,providers):
        seen.append((sess_options.intra_op_num_threads,sess_options.inter_op_num_threads))
        return SimpleNamespace(get_providers=lambda:['CPUExecutionProvider'])
    ort=SimpleNamespace(SessionOptions=Options,InferenceSession=create)
    config=load(root=tmp_path,device='cpu')
    session(config,b'model',runtime_info=(ort,['CPUExecutionProvider']),cpu_threads=4)
    session(config,b'model',runtime_info=(ort,['CPUExecutionProvider']))
    assert seen==[(4,1),(1,1)]
    with pytest.raises(ValueError,match='threads'):session(config,b'model',cpu_threads=0)


def test_native_limit_cleanup_preserves_primary_error(tmp_path,monkeypatch):
    config=setup(tmp_path);budget=NativeBudget(config)
    def failed_cleanup():raise RuntimeError('cleanup')
    monkeypatch.setattr(budget.stack,'close',failed_cleanup)
    with pytest.raises(ValueError,match='primary') as caught:
        with budget:raise ValueError('primary')
    assert any('cleanup' in note for note in caught.value.__notes__)
