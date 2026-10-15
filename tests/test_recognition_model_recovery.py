import pytest
from tests.test_recognition_clustering import setup as cluster_config
from types import SimpleNamespace
from tests.test_recognition_rebuild import changed
from tests.test_recognition_run import setup as run_setup, run


@pytest.mark.parametrize('kind', ['visibility', 'expression'])
def test_failed_optional_model_is_replaced_for_later_measurements(tmp_path, kind):
    from stash_ai.recognition.visibility import LazyVisibility, CONTRACT_ID
    from stash_ai.recognition.expression import LazyExpression, measurement, REQUIRED
    models = []
    class Model:
        def __init__(self): self.failed = not models; self.closed = False
        def measure(self, *a):
            if self.failed: raise RuntimeError('poisoned quality session')
            return ({'visible_face_fraction':1., 'visibility':{'contract':CONTRACT_ID}} if kind == 'visibility'
                    else measurement({key:0. for key in REQUIRED}))
        def close(self): self.closed = True
    def factory(*a):
        model = Model(); models.append(model); return model
    owner = (LazyVisibility if kind == 'visibility' else LazyExpression)(cluster_config(tmp_path), {}, factory)
    with pytest.raises(RuntimeError, match='poisoned'):
        owner.measure(None, None)
    for _ in range(2):
        assert isinstance(owner.measure(None, None), dict)
    assert len(models) == 2 and models[0].closed and not models[1].closed
    owner.close()


def test_auto_preload_failure_falls_back_but_strict_cuda_still_fails(tmp_path, monkeypatch):
    import sys
    import stash_ai.recognition.runtime as runtime
    calls = []
    def version(name):
        if name == 'onnxruntime': raise runtime.metadata.PackageNotFoundError(name)
        return '1'
    def preload(): raise OSError('CUDA DLL preload failed')
    ort = SimpleNamespace(get_available_providers=lambda:['CUDAExecutionProvider','CPUExecutionProvider'],
        preload_dlls=preload, SessionOptions=lambda:SimpleNamespace(),
        InferenceSession=lambda *a,**kw:calls.append(kw) or SimpleNamespace(get_providers=lambda:['CPUExecutionProvider']))
    monkeypatch.setattr(runtime.metadata, 'version', version)
    monkeypatch.setitem(sys.modules, 'onnxruntime', ort)
    result,device=runtime.session(changed(cluster_config(tmp_path), models={'device':'auto'}), b'model')
    assert device=='cpu' and calls[0]['providers']==['CPUExecutionProvider']
    with pytest.raises(OSError, match='preload'):
        runtime.session(changed(cluster_config(tmp_path), models={'device':'cuda'}), b'model')
    assert len(calls)==1


@pytest.mark.parametrize('adapter', ['OnnxYuNet', 'OnnxSFace'])
def test_failed_adapter_contract_releases_owned_session(tmp_path,monkeypatch,adapter):
    import weakref
    import stash_ai.recognition.onnx_models as module
    references=[]
    class Session:
        def get_inputs(self):return []
        def get_outputs(self):return []
    def session(*a,**kw):
        result=Session();references.append(weakref.ref(result));return result,'cpu'
    monkeypatch.setattr(module,'prepared_model',lambda *a,**kw:(b'model','hash'))
    monkeypatch.setattr(module,'session',session)
    with pytest.raises(ValueError,match='contract') as caught:
        args=(None,(SimpleNamespace(__version__='test'),[])) if adapter=='OnnxYuNet' else ((SimpleNamespace(__version__='test'),[]),)
        getattr(module,adapter)(cluster_config(tmp_path),None,*args)
    assert references[0]() is None


def test_cached_run_does_not_prepare_unused_identity_runtime(tmp_path, monkeypatch):
    from stash_ai.recognition.models import ensure_assets
    import stash_ai.recognition.runtime as runtime
    config,client,services,models=run_setup(tmp_path,monkeypatch)
    assert run(config,services,'1') == 0
    def unavailable(*a): raise ValueError('identity runtime unavailable for cached run')
    monkeypatch.setattr(runtime,'runtime',unavailable)
    services.assets=ensure_assets
    assert run(config,services,'1') == 0
    assert len(models) == 1


def test_portrait_quality_assets_do_not_require_identity_weights(tmp_path,monkeypatch):
    from stash_ai.recognition.models import ensure_assets
    from stash_ai.recognition.portrait_refresh import LazyAssets
    import stash_ai.recognition.runtime as runtime
    config=changed(cluster_config(tmp_path),models={'download_missing':False},portraits={'enabled':True})
    import stash_ai.recognition.models as models
    asset=config.model_dir/'faceocc'/models.ASSETS['faceocc'].filename
    asset.parent.mkdir(parents=True);asset.write_bytes(b'verified fixture')
    verified=[]
    monkeypatch.setattr(models,'verify',lambda path,spec:verified.append(path))
    monkeypatch.setattr(runtime,'runtime',lambda _:pytest.fail('FaceOcc assets do not consume ORT'))
    paths=LazyAssets(config,ensure_assets,SimpleNamespace(write=lambda _:None))
    assert paths['faceocc']==asset and verified==[asset]
    assert not (config.model_dir/'yunet').exists()


@pytest.mark.parametrize('kind', ['visibility', 'expression'])
def test_persistent_optional_inference_failure_has_bounded_recovery(tmp_path, kind):
    from stash_ai.recognition.visibility import LazyVisibility
    from stash_ai.recognition.expression import LazyExpression
    models = []
    class Broken:
        def __init__(self): self.closed = False
        def measure(self, *args): raise RuntimeError('persistent model failure')
        def close(self): self.closed = True
    def factory(*args):
        model = Broken()
        models.append(model)
        return model
    owner = (LazyVisibility if kind == 'visibility' else LazyExpression)(cluster_config(tmp_path), {}, factory)
    for _ in range(8):
        with pytest.raises(RuntimeError):
            owner.measure(None, None)
    assert len(models) == 2 and all(model.closed for model in models)
    owner.close()
