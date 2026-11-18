from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from stash_ai.recognition.config import defaults
from stash_ai.recognition.rtmo_detector import RTMO,decode,SeededPose


class Session:
    def __init__(self,batch='batch'):
        self.batch=batch;self.calls=[]
    def get_inputs(self):return [SimpleNamespace(name='images',type='tensor(float)',shape=[self.batch,3,640,640])]
    def run(self,_,inputs):
        pixels=inputs['images'];self.calls.append(pixels)
        boxes=np.tile(np.array([[[64,32,320,320,.9],[64,32,320,320,.8]]],np.float32),(len(pixels),1,1))
        poses=np.zeros((len(pixels),2,17,3),np.float32);poses[:,:,:]=[128,64,.8]
        return boxes,poses


def settings():
    result=defaults()['sapiensid']['pose'];result.update(detector='rtmo',backend='rtmo',batch_size=2)
    return result


def test_batched_one_stage_output_restores_source_coordinates_and_bgr():
    session=Session();model=RTMO(session,'cpu',settings())
    with Image.new('RGB',(320,160),(200,100,50)) as image:
        values=model.predict([image]*3,conf=.5,iou=.5,max_det=4)
    assert [len(v) for v in session.calls]==[2,1]
    assert session.calls[0][0,:,0,0].tolist()==[50,100,200]
    assert session.calls[0][0,:,500,0].tolist()==[114,114,114]
    assert values[0].boxes.xyxy.tolist()==[[32.,16.,160.,160.]]
    assert values[0].keypoints.xy[0,0].tolist()==[64.,32.]
    assert len(values)==3


def test_static_graph_uses_single_frame_without_padding_fake_outputs():
    session=Session(1);model=RTMO(session,'cpu',settings())
    with Image.new('RGB',(100,100)) as image:model.predict([image]*3,conf=.5,iou=.5,max_det=4)
    assert [len(v) for v in session.calls]==[1,1,1]


def test_nonfinite_or_out_of_range_confidence_is_rejected():
    boxes=np.array([[0,0,100,100,.9]],np.float32);pose=np.zeros((1,17,3),np.float32)
    pose[0,0,2]=2
    with pytest.raises(ValueError,match='Malformed RTMO'):decode(boxes,pose,(100,100),1,.5,.5,4)
    boxes[0,0]=np.nan
    with pytest.raises(ValueError):decode(boxes,pose,(100,100),1,.5,.5,4)


def test_rtmo_selectors_and_mask_only_contract():
    from stash_ai.recognition.person_signal import validate,contract
    from stash_ai.recognition.inference_contracts import detector
    cfg=SimpleNamespace(data=defaults());cfg.data['sapiensid']['pose'].update(settings())
    validate(cfg.data['sapiensid'])
    before=contract(cfg);first=detector(cfg)
    cfg.data['sapiensid']['pose']['rtmo_directory']='elsewhere'
    assert contract(cfg)==before and detector(cfg)==first
    cfg.data['sapiensid']['pose']['rtmo_torso_nms']=True
    assert contract(cfg)!=before and detector(cfg)!=first
    cfg.data['sapiensid']['pose']['refinement'].update(enabled=True,generic_second_pass=True)
    with pytest.raises(ValueError,match='mask-only'):validate(cfg.data['sapiensid'])
    cfg.data['sapiensid']['pose']['refinement']['generic_second_pass']=False
    validate(cfg.data['sapiensid'])
    cfg.data['sapiensid']['pose']['detector']='yolo'
    with pytest.raises(ValueError,match='both'):validate(cfg.data['sapiensid'])


def test_mask_seed_cannot_reuse_points_for_different_box():
    with Image.new('RGB',(100,100)) as image:
        image._rtmo_points=([[0,0,100,100]],np.zeros((1,17,2)),np.ones((1,17)))
        points,scores=SeededPose().predict(image,[[0,0,100,100]])
        points[:]=-1
        assert not (image._rtmo_points[1]==-1).any()
        with pytest.raises(ValueError):SeededPose().predict(image,[[0,0,50,50]])


def test_tune_switches_both_one_stage_selectors_and_returns_to_top_down():
    pytest.importorskip('tkinter')
    from pathlib import Path
    from stash_ai.recognition.config import load
    from stash_ai.recognition.tune_editor import Settings
    editor=Settings(load(Path(__file__).resolve().parents[1]/'facial_recognition.yml'))
    editor.edit('pose.estimator.backend','rtmo')
    assert editor.value('pose.detector.backend')=='rtmo'
    editor.edit('pose.refinement.generic_second_pass','false')
    editor.edit('pose.refinement.bidirectional_prompts','false')
    assert editor.effective().data['sapiensid']['pose']['backend']=='rtmo'
    editor.edit('pose.estimator.backend','vitpose_plus_large')
    assert editor.value('pose.detector.backend')=='rfdetr_xl'
    assert 'pose.deduplication' in editor.groups()


def test_recognition_batch_assembles_one_stage_with_optional_masks(monkeypatch):
    from contextlib import nullcontext
    from stash_ai.recognition import rtmo_detector,sam_pose2seg
    from stash_ai.recognition.sapiensid_backend import SapiensIDFaces
    data=defaults();data['models']['device']='cpu'
    data['sapiensid']['enabled']=True;data['sapiensid']['collection']['enabled']=True
    data['sapiensid']['pose'].update(settings())
    data['sapiensid']['pose']['refinement']['enabled']=True
    session=Session()
    monkeypatch.setattr(rtmo_detector,'load',lambda cfg:RTMO(session,'cpu',cfg.data['sapiensid']['pose']))
    segmenter=SimpleNamespace(frame=lambda image:nullcontext(),predict=lambda *args:(np.ones((160,320),bool),.99))
    monkeypatch.setattr(sam_pose2seg,'load',lambda *args:segmenter)
    model=SapiensIDFaces(SimpleNamespace(data=data),{},load_identity=False,load_detector=False)
    try:
        with Image.new('RGB',(320,160)) as a,Image.new('RGB',(320,160)) as b:
            model.detect_many([(a,[]),(b,[])])
            for image in (a,b):
                model.detect(image,faces=[])
                assert len(model.frame_people)==1
                assert len(model.frame_people[0]['keypoints'])==17
        assert len(session.calls)==1
    finally:model.close()
