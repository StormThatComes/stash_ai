"""Verified source imports, offline assets and per-frame SAM lifetime."""
from contextlib import contextmanager
import hashlib
import importlib
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from stash_ai.recognition import sam_pose2seg as sam
from tests.test_recognition_pmpose import configuration


@contextmanager
def fresh_namespace():
    names=lambda: [k for k in sys.modules if k=='bboxmaskpose' or k.startswith('bboxmaskpose.')]
    previous={k:sys.modules.pop(k) for k in names()}
    try:
        yield
    finally:
        for k in names():sys.modules.pop(k)
        sys.modules.update(previous)


def test_imports_only_verified_bytes_without_package_initializers_or_sys_path_changes():
    sources={'bboxmaskpose/sam2/example.py':b'value = 42\n'}
    before=list(sys.path)
    with fresh_namespace():
        with sam.source_imports(sources):
            module=importlib.import_module('bboxmaskpose.sam2.example')
        assert module.value==42 and sys.path==before
        assert sys.modules['bboxmaskpose'].__path__==[]
        with sam.source_imports(sources):
            assert importlib.import_module('bboxmaskpose.sam2.example') is module
        with pytest.raises(RuntimeError,match='foreign Python package'):
            with sam.source_imports({'bboxmaskpose/sam2/example.py':b'value = 43\n'}):pass
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module('bboxmaskpose.training')


def test_foreign_upstream_package_is_never_silently_reused():
    with fresh_namespace():
        sys.modules['bboxmaskpose']=SimpleNamespace()
        with pytest.raises(RuntimeError,match='foreign Python package'):
            with sam.source_imports({}):pass


def test_missing_and_corrupted_source_and_checkpoint_policy(tmp_path,monkeypatch):
    from stash_ai.recognition import sapiensid_downloads
    config=configuration(tmp_path)
    settings=config.data['sapiensid']['pose']['refinement']
    settings['directory']='local/sam'
    sources={'bboxmaskpose/sam2/example.py':b'value = 42\n', 'config.yaml':b'model: {}\n'}
    manifest={p:dict(size=len(raw),sha256=hashlib.sha256(raw).hexdigest()) for p,raw in sources.items()}
    monkeypatch.setattr(sam,'source_manifest',lambda:manifest)
    calls=[]
    def install(url,path,size,checksum):
        calls.append(url)
        path.parent.mkdir(parents=True,exist_ok=True)
        raw=next((raw for name,raw in sources.items() if url.endswith(name)), b'weights')
        path.write_bytes(raw)
    monkeypatch.setattr(sapiensid_downloads,'install',install)
    config.data['models']['download_missing']=False
    with pytest.raises(FileNotFoundError,match='download_missing'):
        sam.assets(config)
    config.data['models']['download_missing']=True
    actual,checkpoint=sam.assets(config)
    assert actual==sources
    assert checkpoint==tmp_path/'local/sam'/sam.registry.NAME
    assert calls[-1]==sam.registry.URL and len(calls)==3
    config.data['models']['download_missing']=False
    assert sam.assets(config)==(actual,checkpoint)
    (tmp_path/'local/sam/source/config.yaml').write_text('corrupt')
    with pytest.raises(ValueError,match='SHA256'):
        sam.assets(config)
    assert len(calls)==3


@pytest.mark.parametrize('fail_at',[None,'image','predict'])
def test_predictor_uses_rgb_points_without_box_and_releases_features_on_all_exits(fail_at):
    torch=pytest.importorskip('torch')
    class Predictor:
        reset=0
        def set_image(self,image):
            assert torch.is_inference_mode_enabled()
            assert image[0,0].tolist()==[10,20,30]
            if fail_at=='image':raise RuntimeError('image failed')
        def predict(self,**kwargs):
            assert torch.is_inference_mode_enabled()
            assert kwargs['box'] is None and not kwargs['multimask_output']
            assert kwargs['point_labels'].tolist()==[1,1]
            if fail_at=='predict':raise RuntimeError('predict failed')
            return np.ones((1,30,40),dtype=bool),np.array([.9]),None
        def reset_predictor(self):self.reset+=1
    predictor=Predictor();segmenter=sam.Segmenter(predictor,'cpu')
    def run():
        with Image.new('RGB',(40,30),(10,20,30)) as image, segmenter.frame(image):
            mask,score=segmenter.predict(np.array([[3,4],[5,6]]))
            assert mask.shape==(30,40) and score==.9
    if fail_at:
        with pytest.raises(RuntimeError):run()
    else:run()
    assert predictor.reset==1


@pytest.mark.parametrize('masks,scores',[(np.zeros((2,4,4)),np.array([.5,.5])),
    (np.full((1,4,4),np.nan),np.array([.9])),(np.full((1,4,4),.5),np.array([.9])),
    (np.ones((1,4,4)),np.array([np.nan])),(np.ones((1,4,4)),np.array([1.1]))])
def test_bad_mask_outputs_raise_instead_of_falling_back(masks,scores):
    predictor=SimpleNamespace(predict=lambda **_: (masks,scores,None))
    with pytest.raises(ValueError,match='malformed/non-finite'):
        sam.Segmenter(predictor,'cpu').predict(np.array([[3,4]]))


def test_loader_strictly_checks_real_torch_state_dict_before_using_predictor(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch')
    pytest.importorskip('hydra');pytest.importorskip('iopath');pytest.importorskip('torchvision')
    path=tmp_path/'model.pt'
    torch.save({'model':torch.nn.Linear(2,3).state_dict()},path)
    monkeypatch.setattr(sam.registry,'SIZE',path.stat().st_size)
    monkeypatch.setattr(sam.registry,'SHA256',hashlib.sha256(path.read_bytes()).hexdigest())
    sources={sam.registry.CONFIG:b'model: {_target_: torch.nn.Linear, in_features: 2, out_features: 3}\n',
        'bboxmaskpose/sam2/sam2_image_predictor.py':b'class SAM2ImagePredictor:\n def __init__(self, model, **kw): self.model=model\n'}
    # The real loader's SAM decoder options require a constructor accepting them.
    sources[sam.registry.CONFIG]=b'model: {_target_: bboxmaskpose.sam2.example.Model}\n'
    sources['bboxmaskpose/sam2/example.py']=b'import torch\nclass Model(torch.nn.Linear):\n def __init__(self, **kw): super().__init__(2,3)\n'
    monkeypatch.setattr(sam,'assets',lambda _: (sources,path))
    with fresh_namespace():
        segmenter=sam.load(None,'cpu')
        assert not segmenter.predictor.model.training
        path.write_bytes(b'x'*path.stat().st_size)
        with pytest.raises(ValueError,match='SHA256'):
            sam.load(None,'cpu')
