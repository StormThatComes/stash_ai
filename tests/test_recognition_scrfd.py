"""SCRFD KPS model policy, geometry, ONNX contracts and bounded decoding."""
from types import SimpleNamespace

import numpy as np
import pytest

from stash_ai.recognition.config import load
from stash_ai.recognition.models import ASSETS, asset_selection, namespace, YuNetDetector
from stash_ai.recognition.scrfd import ScrfdDetector


def policy(score=.5, maximum=5):
    return SimpleNamespace(data={'detection': {
        'score_threshold': score, 'nms_threshold': .3,
        'max_faces_per_frame': maximum,
    }})


def predictions():
    """SCRFD has two anchors per stride and channels [1,4,10] per output."""
    output = []
    for channels in (1, 4, 10):
        for stride in (8, 16, 32):
            count = (640 // stride) ** 2 * 2
            output.append(np.zeros((1, count, channels), dtype=np.float32))
    return output


def test_scrfd_10g_is_new_default_and_existing_yunet_stays_supported(tmp_path):
    cfg = load(root=tmp_path)
    assert cfg.data['models']['detector'] == 'scrfd_10g'
    selected, identities = asset_selection(cfg)
    assert selected[:2] == ['scrfd_10g', 'sface']
    assert {'scrfd_10g', 'sface'}.issubset(identities)
    assert 'yunet' not in selected
    assert ASSETS['scrfd_10g'].size == 16923827
    assert ASSETS['scrfd_2_5g'].size == 3291737
    assert all(len(ASSETS[key].sha256) == 64 for key in ('scrfd_10g','scrfd_2_5g'))

    path = tmp_path / 'detector.yml'
    path.write_text('version: 1\nmodels: {detector: scrfd_2_5g}\n')
    second = load(path)
    assert asset_selection(second)[0][:2] == ['scrfd_2_5g', 'sface']
    assert namespace(second) != namespace(cfg)
    path.write_text('version: 1\nmodels: {detector: yunet}\n')
    legacy = load(path)
    assert asset_selection(legacy)[0][:2] == ['yunet', 'sface']
    assert namespace(legacy) != namespace(cfg)
    assert namespace(legacy) != namespace(second)
    path.write_text('version: 1\nmodels: {detector: scrfd_10g}\n')
    assert namespace(load(path)) == namespace(cfg)


@pytest.mark.parametrize('name', ['scrfd_10g','scrfd_2_5g'])
def test_scrfd_bgr_to_rgb_normalization_and_aspect_padding(name):
    # Three independent corners validate channel order, normalization and
    # top-left letterbox; a non-square frame must not stretch face landmarks.
    bgr = np.empty((160, 80, 3), dtype=np.uint8)
    bgr[:] = [0, 127, 255]
    tensor, sx, sy = ScrfdDetector._blob(bgr)
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32 and tensor.flags.c_contiguous
    assert (sx,sy) == pytest.approx((4,4))
    assert tensor[0,:,100,100] == pytest.approx([(255-127.5)/128,
                                                (127-127.5)/128,
                                                (0-127.5)/128])
    assert tensor[0,:,100,500] == pytest.approx([-127.5/128]*3)
    assert name in ASSETS


def test_scrfd_decode_landmarks_two_anchors_scales_and_nms():
    output = predictions()
    stride = 8
    # Feature-map position (x=8, y=10), second of two SCRFD anchors.
    index = ((10 * 80 + 8) * 2 + 1)
    output[0][0,index,0] = .95
    output[3][0,index,:] = [1,2,3,4]
    output[6][0,index,:] = [-1,0,1,0,0,1,-1,2,1,2]
    faces = ScrfdDetector.decode(output,scale_x=2,scale_y=2,config=policy())
    assert len(faces)==1
    assert faces[0].score==pytest.approx(.95)
    assert faces[0].box==pytest.approx((28,32,16,24))
    assert np.asarray(faces[0].landmarks)==pytest.approx([
        [28,40], [36,40], [32,44], [28,48], [36,48]
    ])
    # A duplicate detection from the same cell must be suppressed by NMS.
    output[0][0,index-1,0] = .94
    output[3][0,index-1,:] = output[3][0,index,:]
    output[6][0,index-1,:] = output[6][0,index,:]
    assert len(ScrfdDetector.decode(output,scale_x=2,scale_y=2,config=policy())) == 1
    assert len(ScrfdDetector.decode(output,scale_x=2,scale_y=2,config=policy(score=.99))) == 0


def test_scrfd_rejects_invalid_output_and_nonfinite_tensors():
    outputs = predictions()
    with pytest.raises(ValueError,match='nine'):
        ScrfdDetector.decode(outputs[:-1],scale_x=1,scale_y=1,config=policy())
    outputs[0] = np.zeros((1,1,1),dtype=np.float32)
    with pytest.raises(ValueError,match='malformed'):
        ScrfdDetector.decode(outputs,scale_x=1,scale_y=1,config=policy())
    outputs = predictions()
    outputs[6][0,0,0] = np.nan
    with pytest.raises(ValueError,match='malformed'):
        ScrfdDetector.decode(outputs,scale_x=1,scale_y=1,config=policy())
    with pytest.raises(ValueError,match='uint8 BGR'):
        ScrfdDetector._blob(np.zeros((160,160,3),dtype=np.float32))


def test_scrfd_nms_enforces_final_limit():
    outputs=predictions()
    for n in range(20):
        index=((10 * 80 + 2*n) * 2)
        outputs[0][0,index,0]=.99
        outputs[3][0,index,:]=[1,1,1,1]
        outputs[6][0,index,:]=[0,0,1,0,0,1,0,2,1,2]
    faces=ScrfdDetector.decode(outputs,scale_x=1,scale_y=1,config=policy(maximum=3))
    assert len(faces)==3


def test_scrfd_adapter_uses_verified_weights_and_explicit_ort_contract(tmp_path,monkeypatch):
    import stash_ai.recognition.scrfd as scrfd
    input_row=SimpleNamespace(name='scrfd_input',type='tensor(float)',shape=[1,3,640,640])
    output_rows=[SimpleNamespace(name=f'out{n}',type='tensor(float)',shape=[1,None,channels])
                 for channels in (1,4,10) for n in range(3)]
    calls=[]
    class FakeSession:
        def get_inputs(self): return [input_row]
        def get_outputs(self): return output_rows
        def get_providers(self): return ['CPUExecutionProvider']
        def run(self,names,feed):
            calls.append((names,feed))
            return predictions()
    session=FakeSession()
    monkeypatch.setattr(scrfd,'verified_bytes', lambda *args:b'pinned')
    monkeypatch.setattr(scrfd,'session',lambda *args,**kwargs:(session,'cpu'))
    info=(SimpleNamespace(__version__='fake'),['CPUExecutionProvider'])
    cfg=load(root=tmp_path,device='cpu')
    detector=ScrfdDetector(cfg,{'scrfd_10g':tmp_path/'test.onnx'},runtime_info=info)
    try:
        assert detector.device=='cpu' and detector.provenance['detector']=='scrfd_10g'
        assert detector.detect(np.zeros((120,160,3),dtype=np.uint8))==[]
        assert calls[0][0]==[f'out{n}' for n in range(3)]*3
        assert calls[0][1]['scrfd_input'].shape==(1,3,640,640)
    finally:
        detector.close()
    assert detector.session is None


def test_shared_face_adapter_selects_scrfd_and_retains_yunet_api(tmp_path,monkeypatch):
    import stash_ai.recognition.models as models
    import stash_ai.recognition.scrfd as scrfd
    state={'loaded':[],'closed':[]}
    class Stub:
        device='cpu'
        provenance={'format':'scrfd-kps-test'}
        def __init__(self,config,paths,*,runtime_info=None):
            state['loaded'].append(config.data['models']['detector'])
        def detect(self, bgr):
            return []
        def close(self): state['closed'].append(True)
    monkeypatch.setattr(scrfd,'ScrfdDetector',Stub)
    cfg=load(root=tmp_path,device='cpu')
    backend=YuNetDetector(cfg, {'scrfd_10g':'stub'})
    try:
        assert backend.detector_device=='cpu'
        from PIL import Image
        with Image.new('RGB',(80,80)) as img:
            assert backend.detect(img)==[]
        assert state['loaded']==['scrfd_10g']
    finally:
        backend.close()
    assert state['closed']==[True]
