"""ONNX model pins, preprocessing and adapter contracts (no model downloads)."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from stash_ai.recognition.config import load
from stash_ai.recognition.models import ASSETS, namespace


@pytest.mark.parametrize('recognizer,order', [
    ('lvface_b', [255, 127, 0]),
    ('adaface_ir50', [0, 127, 255]),
])
def test_model_specific_rgb_bgr_once_normalization(recognizer, order):
    from stash_ai.recognition.onnx_recognizers import input_blob
    image=np.zeros((112,112,3),dtype=np.uint8)
    image[:]=[0,127,255]  # BGR
    blob=input_blob(image,recognizer=recognizer)
    assert blob.shape==(1,3,112,112)
    assert blob.dtype==np.float32 and blob.flags.c_contiguous
    assert blob[0,:,0,0]==pytest.approx([(v-127.5)/127.5 for v in order])
    for bad in (np.ones((112,112,3),dtype=np.float32),
                np.zeros((128,128,3),dtype=np.uint8)):
        with pytest.raises(ValueError,match='112x112 uint8'):
            input_blob(bad,recognizer=recognizer)


@pytest.mark.parametrize('recognizer,sha,size', [
    ('lvface_b','9d834ed8e927fd35b9123b2bf97c40aad05785b1f9ecfb1c4c1f6242d38d1382',455533594),
    ('adaface_ir50','a3c1b24e4271c98526d7b840c9d18e57ee281c6782675ace3e09e79dcedd98e8',174398313),
])
def test_selected_hf_weights_are_fully_pinned_and_licensed(recognizer,sha,size):
    asset=ASSETS[recognizer]
    assert asset.sha256==sha and asset.size==size
    assert len(asset.revision)==40
    assert asset.repo and asset.filename.endswith('.onnx')
    assert 'research' in asset.license
    assert asset.dimensions==512


def config(tmp_path,recognizer,device='cpu'):
    path=tmp_path/(recognizer+'.yml')
    path.write_text(f'version: 1\nmodels: {{detector: yunet, recognizer: {recognizer}, device: {device}}}\n'
                    'detection: {min_visible_face_fraction: 0}\nportraits: {enabled: false}\n')
    return load(path)


def test_model_namespace_is_unique_even_for_identical_embedding_dimension(tmp_path):
    original=namespace(config(tmp_path,'auraface'))
    values=[namespace(config(tmp_path,model)) for model in
            ('sface','auraface','lvface_b','adaface_ir50')]
    assert len(set(values))==4
    assert original==values[1]
    # Device, model directory and similarity tuning change neither identity
    # nor the model's embedding namespace.
    alt=tmp_path/'alt.yml'
    alt.write_text('version: 1\nmodels: {detector: yunet, recognizer: lvface_b, device: cuda, directory: other}\n'
                   'recognition: {match_threshold: 0.81}')
    assert namespace(load(alt))==values[2]


@pytest.mark.parametrize('recognizer',('lvface_b','adaface_ir50'))
def test_only_requested_model_weights_are_downloaded(tmp_path,monkeypatch,recognizer):
    import stash_ai.recognition.runtime as face_runtime
    import stash_ai.recognition.models as models
    import stash_ai.backends.downloads as downloads
    monkeypatch.setattr(face_runtime,'runtime',lambda cfg: (None,None))
    monkeypatch.setattr(models,'verify',lambda *args: None)
    calls=[]
    monkeypatch.setattr(downloads,'install_hf_asset',
        lambda repo,filename,revision,destination,**kwargs: calls.append(
            (repo,filename,revision,kwargs['expected_sha256'],kwargs['max_bytes'])))
    selected=models.ensure_assets(config(tmp_path,recognizer),write=lambda _:None)
    assert set(selected)=={'yunet',recognizer}
    assert calls==[(ASSETS[key].repo,ASSETS[key].filename,ASSETS[key].revision,
                     ASSETS[key].sha256,ASSETS[key].size) for key in ('yunet',recognizer)]


class FakeSession:
    def __init__(self,*,output_dim=512,input_shape=None,output_shape=None,device='cpu'):
        self.input=SimpleNamespace(type='tensor(float)',name='input',
            shape=input_shape if input_shape is not None else ['batch',3,112,112])
        self.output=SimpleNamespace(type='tensor(float)',name='embedding',
            shape=output_shape if output_shape is not None else ['batch',output_dim])
        self.device=device
        self.samples=[]
        self.outputs=None

    def get_inputs(self):
        return [self.input]

    def get_outputs(self):
        return [self.output]

    def get_providers(self):
        return ['CUDAExecutionProvider','CPUExecutionProvider'] if self.device=='cuda' else ['CPUExecutionProvider']

    def run(self,outputs,tensors):
        self.samples.append(tensors['input'])
        if self.outputs is not None:
            return self.outputs
        return [np.ones((1,512),dtype=np.float32)]


@pytest.mark.parametrize('recognizer',('lvface_b','adaface_ir50'))
def test_adapter_runs_five_point_alignment_and_normalized_512d_output(tmp_path,monkeypatch,recognizer):
    from stash_ai.recognition import onnx_recognizers as adapter
    from stash_ai.recognition.models import YuNetDetector
    class CV:
        def warpAffine(self,image,transform,size,borderValue):
            assert size==(112,112) and transform.shape==(2,3)
            return np.tile(np.array([0,127,255],dtype=np.uint8),(112,112,1))
    def fake_init(self,cfg,paths,**kwargs):
        self.cv=CV()
        self.provenance={}
    monkeypatch.setattr(YuNetDetector,'__init__',fake_init)
    monkeypatch.setattr(YuNetDetector,'close',lambda self: None)
    monkeypatch.setattr(YuNetDetector,'_bgr',lambda self,image: image)
    monkeypatch.setattr(adapter,'runtime',lambda config:(SimpleNamespace(__version__='test'),['CPUExecutionProvider']))
    monkeypatch.setattr(adapter,'verified_bytes',lambda path,asset:b'mock-verified')
    fake=FakeSession()
    monkeypatch.setattr(adapter,'session',
        lambda cfg,blob,**kwargs:(fake,'cpu'))
    cfg=config(tmp_path,recognizer)
    backend=adapter.OnnxEmbeddingFaces(cfg,{recognizer:object()})
    source=np.asarray([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],
                       [41.5493,92.3655],[70.7299,92.2041]])
    detection=SimpleNamespace(landmarks=source)
    vec=backend.embed(np.zeros((112,112,3),dtype=np.uint8),detection)
    assert vec.shape==(512,) and vec.dtype==np.float32
    assert np.linalg.norm(vec)==pytest.approx(1.0)
    assert fake.samples[0][0,:,0,0]==pytest.approx(
        [1,(127-127.5)/127.5,-1] if recognizer=='lvface_b'
        else [-1,(127-127.5)/127.5,1])
    assert backend.provenance['weights']['sha256']==ASSETS[recognizer].sha256


@pytest.mark.parametrize('recognizer',('lvface_b','adaface_ir50'))
@pytest.mark.parametrize('input_shape,output_shape',[
    ([1,3,224,224],[1,512]),([1,1,112,112],[1,512]),
    ([1,3,112,112],[1,128]),([2,3,112,112],[2,512]),
])
def test_adapter_rejects_wrong_onnx_contract_before_inference(tmp_path,monkeypatch,recognizer,input_shape,output_shape):
    from stash_ai.recognition import onnx_recognizers as adapter
    from stash_ai.recognition.models import YuNetDetector
    monkeypatch.setattr(YuNetDetector,'__init__',
        lambda self,cfg,paths,**kw: setattr(self,'provenance',{}))
    monkeypatch.setattr(YuNetDetector,'close',lambda self: None)
    monkeypatch.setattr(adapter,'runtime',lambda cfg:(SimpleNamespace(__version__='test'),['CPUExecutionProvider']))
    monkeypatch.setattr(adapter,'verified_bytes',lambda *args:b'verified')
    monkeypatch.setattr(adapter,'session',lambda *a,**k:(
        FakeSession(input_shape=input_shape,output_shape=output_shape),'cpu'))
    with pytest.raises(ValueError,match='ONNX'):
        adapter.OnnxEmbeddingFaces(config(tmp_path,recognizer),{recognizer:object()})


@pytest.mark.parametrize('recognizer',('lvface_b','adaface_ir50'))
def test_cuda_fallback_is_refused(tmp_path,monkeypatch,recognizer):
    from stash_ai.recognition import onnx_recognizers as adapter
    from stash_ai.recognition.models import YuNetDetector
    monkeypatch.setattr(YuNetDetector,'__init__',
        lambda self,cfg,paths,**kw: setattr(self,'provenance',{}))
    monkeypatch.setattr(YuNetDetector,'close',lambda self: None)
    monkeypatch.setattr(adapter,'runtime',lambda cfg:(SimpleNamespace(__version__='test'),['CUDAExecutionProvider']))
    monkeypatch.setattr(adapter,'verified_bytes',lambda *args:b'verified')
    monkeypatch.setattr(adapter,'session',lambda *a,**k:(FakeSession(),'cpu'))
    with pytest.raises(ValueError,match='CUDA'):
        adapter.OnnxEmbeddingFaces(config(tmp_path,recognizer,device='cuda'),
                                   {recognizer:object()})


@pytest.mark.parametrize('recognizer',('lvface_b','adaface_ir50'))
def test_loader_routes_to_selected_backend(tmp_path,monkeypatch,recognizer):
    from stash_ai.recognition import models, onnx_recognizers
    import stash_ai.runtime.shutdown as shutdown
    monkeypatch.setattr(shutdown,'require_idle',lambda:None)
    marker=object()
    monkeypatch.setattr(onnx_recognizers,'OnnxEmbeddingFaces',lambda *args: marker)
    assert models.load_backend(config(tmp_path,recognizer),{}) is marker
