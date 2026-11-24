"""Independent prior decoding and source-coordinate contract for RetinaFace."""
from types import SimpleNamespace

import numpy as np
import pytest

from stash_ai.recognition.config import load, defaults
from stash_ai.recognition.models import ASSETS, asset_selection, namespace, YuNetDetector
from stash_ai.recognition.retinaface import RetinaFaceDetector


def policy(**changes):
    data=defaults()
    data['models'].update(detector='retinaface_r50',device='cpu')
    data['detection'].update(score_threshold=.5,nms_threshold=.3,**changes)
    return SimpleNamespace(data=data)


def predictions():
    return [np.zeros((1,16800,n),np.float32) for n in (4,2,10)]


def test_new_detector_is_opt_in_and_changes_evidence_namespace(tmp_path):
    from stash_ai.recognition.inference_contracts import face
    from stash_ai.recognition.detection_parameters import apply_edits
    from stash_ai.recognition.tune_editor import CHOICES
    original=load(root=tmp_path,device='cpu')
    config,_=apply_edits(original,{'face.detector.backend':'retinaface_r50'},detection_preview=True)
    assert original.data['models']['detector']=='scrfd_10g'
    assert config.data['models']['detector']=='retinaface_r50'
    assert 'retinaface_r50' in CHOICES['face.detector.backend']
    assert asset_selection(config)[0][:2]==['retinaface_r50','sface']
    assert namespace(config)!=namespace(original) and face(config)!=face(original)
    assert ASSETS['retinaface_r50'].size==109458344
    assert len(ASSETS['retinaface_r50'].sha256)==64
    path=tmp_path/'retina.yml'
    path.write_text('version: 2\nface: {detector: {backend: retinaface_r50}}\n')
    assert load(path).data['models']['detector']=='retinaface_r50'


@pytest.mark.parametrize('shape',[(252,448),(448,252),(159,281)])
def test_preprocessing_preserves_bgr_order_and_uses_actual_integer_resize(shape):
    image=np.full((*shape,3),[110,120,130],np.uint8)
    tensor,sx,sy=RetinaFaceDetector._blob(image)
    assert tensor.shape==(1,3,640,640) and tensor.dtype==np.float32
    assert tensor.flags.c_contiguous
    np.testing.assert_allclose(tensor[0,:,0,0],[6,3,7])
    np.testing.assert_allclose(tensor[0,:,-1,-1],[-104,-117,-123])
    assert sx==int(shape[1]*640/max(shape))/shape[1]
    assert sy==int(shape[0]*640/max(shape))/shape[0]


def test_priors_use_half_cell_centers_sizes_and_level_order():
    priors=RetinaFaceDetector.priors()
    assert priors.shape==(16800,4)
    np.testing.assert_array_equal(priors[:3],[[4,4,16,16],[4,4,32,32],[12,4,16,16]])
    np.testing.assert_array_equal(priors[12800:12802],[[8,8,64,64],[8,8,128,128]])
    np.testing.assert_array_equal(priors[16000:16002],[[16,16,256,256],[16,16,512,512]])


def test_decode_prior_offsets_exponential_box_sizes_and_independent_xy_scales():
    output=predictions()
    index=((10*80+8)*2+1)  # center (68,84), size 32
    output[0][0,index]=[1,-2,np.log(2)/.2,np.log(.5)/.2]
    output[1][0,index]=[.05,.95]
    output[2][0,index]=[-2,-1,2,-1,0,1,-1,2,1,2]
    faces=RetinaFaceDetector.decode(output,scale_x=2,scale_y=4,config=policy())
    assert len(faces)==1 and faces[0].score==pytest.approx(.95)
    np.testing.assert_allclose(faces[0].box,[19.6,17.4,32,4],atol=1e-5)
    np.testing.assert_allclose(faces[0].landmarks,
        [[30.8,20.2],[37.2,20.2],[34,21.8],[32.4,22.6],[35.6,22.6]],atol=1e-5)


def test_nms_candidate_floor_and_face_limit():
    output=predictions()
    for index,score in [(1,.9),(3,.8),(41,.7),(81,.4)]:
        output[1][0,index]=[1-score,score]
    decode=lambda cfg:RetinaFaceDetector.decode(output,scale_x=1,scale_y=1,config=cfg)
    assert len(decode(policy()))==2
    assert len(decode(policy(max_faces_per_frame=1)))==1
    assert len(decode(policy(candidate_score_threshold=.3)))==3


@pytest.mark.parametrize('defect',['count','shape','nan','probability','overflow','scale'])
def test_malformed_model_results_fail_explicitly(defect):
    outputs=predictions();scale=1
    outputs[1][0,1]=[.1,.9]
    if defect=='count':outputs.pop()
    elif defect=='shape':outputs[2]=np.zeros((1,16800,6))
    elif defect=='nan':outputs[2][0,1,0]=np.nan
    elif defect=='probability':outputs[1][0,1,1]=1.1
    elif defect=='overflow':outputs[0][0,1,2]=10000
    elif defect=='scale':scale=0
    with pytest.raises(ValueError,match='RetinaFace'):
        RetinaFaceDetector.decode(outputs,scale_x=scale,scale_y=1,config=policy())


@pytest.mark.parametrize('bad',[np.zeros((0,10,3),np.uint8),np.zeros((10,10),np.uint8),
                               np.zeros((10,10,4),np.uint8),np.zeros((10,10,3),np.float32)])
def test_invalid_source_images_are_rejected(bad):
    with pytest.raises(ValueError,match='uint8 BGR'):
        RetinaFaceDetector._blob(bad)


@pytest.mark.parametrize('bad_contract',[False,True])
def test_shared_adapter_loads_verified_model_and_checks_output_contract(tmp_path,monkeypatch,bad_contract):
    from stash_ai.recognition import retinaface
    calls=[]
    class Session:
        def get_inputs(self):return [SimpleNamespace(name='input',type='tensor(float)',shape=[1,3,640,640])]
        def get_outputs(self):return [SimpleNamespace(name=str(c),type='tensor(float)',
            shape=[1,16800,c]) for c in (4,2,6 if bad_contract else 10)]
        def get_providers(self):return ['CPUExecutionProvider']
        def run(self,names,feed):
            calls.append(feed['input'].shape)
            return predictions()
    config=load(root=tmp_path,device='cpu')
    from stash_ai.recognition.detection_parameters import apply_edits
    config,_=apply_edits(config,{'face.detector.backend':'retinaface_r50'},detection_preview=True)
    def verify(path,asset):
        assert asset is ASSETS['retinaface_r50'] and path=='pinned-model'
        return b'verified'
    def create(cfg,blob,**kwargs):
        assert blob==b'verified'
        return Session(),'cpu'
    monkeypatch.setattr(retinaface,'verified_bytes',verify)
    monkeypatch.setattr(retinaface,'session',create)
    monkeypatch.setattr(retinaface,'runtime',lambda c:(SimpleNamespace(__version__='test'),['CPUExecutionProvider']))
    if bad_contract:
        with pytest.raises(ValueError,match='RetinaFace ONNX contract'):
            YuNetDetector(config,{'retinaface_r50':'pinned-model'})
        return
    from PIL import Image
    backend=YuNetDetector(config,{'retinaface_r50':'pinned-model'})
    try:
        assert backend.provenance['detector']['detector']=='retinaface_r50'
        with Image.new('RGB',(448,252)) as a,Image.new('RGB',(252,448)) as b:
            assert backend.detect_faces_many([a,b])==[[],[]]
        assert calls==[(1,3,640,640)]*2
    finally:backend.close()


def test_retinaface_uses_rotation_repair_and_numeric_audit():
    from stash_ai.recognition import landmark_repair_audit as audit
    from tests.test_recognition_landmark_repair import OBSERVED, OBSERVED_CROP, synthetic_retry, project, detector
    pixels=np.arange(252*448*3,dtype=np.uint8).reshape(252,448,3)
    expected=synthetic_retry(1);answer=project(expected,OBSERVED_CROP,1)
    upright=np.rot90(pixels[22:223,:337],1)
    stub,_=detector([OBSERVED],lambda img:[answer] if np.array_equal(img,upright) else [])
    model=object.__new__(RetinaFaceDetector)
    model.config=stub.config;model.config.data['models']['detector']='retinaface_r50'
    model._raw_many=stub._raw_many
    with audit.capture() as reports:
        result=model.detect(pixels)
    np.testing.assert_allclose(result[0].row,expected.row,atol=1e-6)
    assert reports[0]['rotation_enabled']
    assert reports[0]['faces'][0]['status']=='repaired'
