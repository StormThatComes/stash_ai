from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from stash_ai.recognition.config import load
from stash_ai.recognition.models import namespace, ensure_assets
from stash_ai.recognition.onnx_models import OnnxYuNet, sface_blob
from stash_ai.recognition.runtime import runtime, session


def one_variant(monkeypatch):
    import stash_ai.recognition.runtime as face_runtime
    def version(name):
        if name == 'onnxruntime':
            raise face_runtime.metadata.PackageNotFoundError(name)
        return '1.26.0'
    monkeypatch.setattr(face_runtime.metadata, 'version', version)
    monkeypatch.setitem(sys.modules, 'torch', None)
    monkeypatch.setitem(sys.modules, 'cv2', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'onnx', SimpleNamespace())


def test_default_is_gpu_and_cpu_override_keeps_existing_namespace(tmp_path):
    gpu = load(root=tmp_path)
    cpu = load(root=tmp_path, device='cpu')
    assert gpu.data['models']['recognizer'] == 'sface'
    assert gpu.data['models']['device'] == 'cuda'
    assert cpu.data['models']['device'] == 'cpu'
    assert namespace(cpu) == namespace(gpu)
    assert namespace(cpu) == 'b035e353b98c2b02a42e61e7cacd9244a0a25b228828116a83554723e7366ab0'
    shipped = load(Path(__file__).resolve().parents[1] / 'facial_recognition.yml')
    assert shipped.data['models']['device'] == 'cuda'
    from stash_ai.recognition.cli import parser
    assert parser().parse_args(['run','--scene-id','1','--device','cpu']).device == 'cpu'


def test_gpu_unavailable_stops_before_assets_or_model_directory(tmp_path, monkeypatch):
    one_variant(monkeypatch)
    monkeypatch.setitem(sys.modules, 'onnxruntime', SimpleNamespace(get_available_providers=lambda: ['CPUExecutionProvider']))
    with pytest.raises(ValueError, match='no CUDA'):
        ensure_assets(load(root=tmp_path))
    assert not (tmp_path / 'recognition_models').exists()
    assert not (tmp_path / 'faces_state.sqlite3').exists()


def test_face_runtime_reuses_application_cuda_preload_and_requires_actual_provider(tmp_path, monkeypatch):
    one_variant(monkeypatch)
    events = []
    ort = SimpleNamespace(get_available_providers=lambda: ['CUDAExecutionProvider','CPUExecutionProvider'],
        preload_dlls=lambda: events.append('preload'), SessionOptions=lambda: SimpleNamespace(),
        InferenceSession=lambda model, **kwargs: SimpleNamespace(get_providers=lambda: ['CPUExecutionProvider']))
    monkeypatch.setitem(sys.modules, 'onnxruntime', ort)
    config = load(root=tmp_path)
    info = runtime(config)
    assert info[1] == ['CUDAExecutionProvider','CPUExecutionProvider'] and events == ['preload']
    with pytest.raises(ValueError, match='fell back'):
        session(config, b'model', runtime_info=info)


def test_only_explicit_auto_allows_failed_cuda_initialization_to_use_cpu(tmp_path, monkeypatch):
    one_variant(monkeypatch)
    calls = []
    def create(model, *, providers, **kwargs):
        calls.append(providers)
        if providers[0] == 'CUDAExecutionProvider':
            raise RuntimeError('CUDA libraries unavailable')
        return SimpleNamespace(get_providers=lambda: providers)
    ort = SimpleNamespace(get_available_providers=lambda: ['CUDAExecutionProvider','CPUExecutionProvider'],
        preload_dlls=lambda: None, SessionOptions=lambda: SimpleNamespace(), InferenceSession=create)
    monkeypatch.setitem(sys.modules, 'onnxruntime', ort)
    with pytest.raises(RuntimeError, match='libraries'):
        session(load(root=tmp_path), b'model')
    _, device = session(load(root=tmp_path, device='auto'), b'model')
    assert device == 'cpu' and calls[-1] == ['CPUExecutionProvider']


def test_sface_blob_does_not_repeat_graph_normalization():
    bgr = np.empty((112,112,3), dtype='uint8')
    bgr[:] = [0,127,255]
    result = sface_blob(bgr)
    assert result.shape == (1,3,112,112) and result.flags.c_contiguous
    assert result[0,:,0,0] == pytest.approx([255,127,0])


def test_yunet_decoding_matches_stride_geometry_and_rejects_bad_shapes(tmp_path):
    detector = OnnxYuNet.__new__(OnnxYuNet)
    detector.config = load(root=tmp_path, device='cpu')
    outputs = {}
    for stride in (8,16,32):
        count = (32//stride)**2
        for name, channels in (('cls',1),('obj',1),('bbox',4),('kps',10)):
            outputs[f'{name}_{stride}'] = np.zeros((1,count,channels), dtype='float32')
    outputs['cls_8'][0,5,0] = outputs['obj_8'][0,5,0] = .99
    outputs['bbox_8'][0,5] = [1,2,0,0]
    faces = detector.decode(outputs,32,32)
    assert len(faces) == 1 and faces[0].box == pytest.approx([12,20,8,8])
    outputs['bbox_8'] = np.zeros((1,1,4))
    with pytest.raises(ValueError, match='malformed'):
        detector.decode(outputs,32,32)


def test_default_sface_routes_both_neural_models_through_cuda_not_opencv(tmp_path, monkeypatch):
    from stash_ai.recognition.models import load_backend, checked_detection
    import stash_ai.recognition.models as models
    import stash_ai.recognition.onnx_models as onnx_models
    from PIL import Image
    one_variant(monkeypatch)
    events = []
    class Session:
        def __init__(self, model, *, providers, **kwargs):
            self.detector = model == b'yunet'
            self.providers = providers
            events.append(('load', model, providers))

        def get_providers(self):
            return self.providers

        def get_inputs(self):
            return [SimpleNamespace(name='data', type='tensor(float)',
                    shape=[1,3,'height','width'] if self.detector else [1,3,112,112])]

        def get_outputs(self):
            return ([SimpleNamespace(name=n, type='tensor(float)') for n in OnnxYuNet.names] if self.detector else
                    [SimpleNamespace(name='fc1', type='tensor(float)', shape=[1,128])])

        def run(self, names, inputs):
            events.append(('infer', 'detector' if self.detector else 'recognizer'))
            if not self.detector:
                return [np.ones((1,128), dtype='float32')]
            height, width = inputs['data'].shape[2:]
            return [np.zeros((1,(height//int(n.split('_')[1]))*(width//int(n.split('_')[1])),
                     {'cls':1,'obj':1,'bbox':4,'kps':10}[n.split('_')[0]]), dtype='float32') for n in names]
    ort = SimpleNamespace(__version__='1.26.0', get_available_providers=lambda: ['CUDAExecutionProvider','CPUExecutionProvider'],
                          preload_dlls=lambda: None, SessionOptions=lambda: SimpleNamespace(), InferenceSession=Session)
    monkeypatch.setitem(sys.modules, 'onnxruntime', ort)
    aligner = SimpleNamespace(alignCrop=lambda data, face: data[:112,:112],
                             feature=lambda data: pytest.fail('GPU SFace executed CPU feature()'))
    cv = SimpleNamespace(__version__='mock', FaceRecognizerSF=SimpleNamespace(create=lambda *args: aligner),
                         FaceDetectorYN=SimpleNamespace(create=lambda *args: pytest.fail('GPU YuNet executed CPU detector')))
    monkeypatch.setitem(sys.modules, 'cv2', cv)
    monkeypatch.setattr(models, 'verify', lambda *args: None)
    monkeypatch.setattr(onnx_models, 'prepared_model', lambda path, **kwargs: (str(path).encode(), 'sha'))
    backend = load_backend(load(root=tmp_path), {'yunet': 'yunet', 'sface': 'sface'})
    try:
        with Image.new('RGB',(128,128)) as image:
            assert backend.detect(image) == []
            face = checked_detection([0,0,128,128,38,44,88,44,64,66,42,90,86,90,.99])
            assert backend.embed(image,face).shape == (128,)
        assert backend.device == backend.detector_device == 'cuda'
        assert ('infer','detector') in events and ('infer','recognizer') in events
        assert all(e[2][0] == 'CUDAExecutionProvider' for e in events if e[0] == 'load')
    finally:
        backend.close()
