from types import SimpleNamespace
import numpy as np
from PIL import Image
from stash_ai.recognition.pose_masks import PoseMasks
from stash_ai.recognition.pose_refinement_config import defaults


def test_generic_mask_preserves_geometry_and_marks_unknown_visibility():
    xy = np.zeros((1,17,2)); xy[0] = [20,20]
    xy[0,5] = [15,30]; xy[0,6] = [25,30]
    scores = np.full((1,17),.9)
    pose = SimpleNamespace(batch_size=1,predict_many=lambda frames:[(xy,scores)])
    mask = np.ones((50,50),bool)
    segmenter = SimpleNamespace(frame=lambda image: __import__('contextlib').nullcontext(),
                                predict=lambda *args:(mask,.99))
    model = PoseMasks(pose,segmenter,dict(defaults(),enabled=True))
    with Image.new('RGB',(50,50)) as image:
        result = model.predict(image,[[0,0,50,50]])
    np.testing.assert_array_equal(result[0],xy)
    assert result[3][0]['status'] == 'masked'
    assert result[3][0]['visibility_measured'] is False
    assert result[3][0]['overlap_mask']


def test_generic_copied_skeleton_is_suppressed_without_measured_visibility():
    from stash_ai.recognition.pose_masks import ConfidencePose
    from stash_ai.recognition.pose_deduplication import DeduplicatedPMPose
    from stash_ai.recognition.pose_deduplication_config import defaults as dedup_defaults
    from tests.test_recognition_pose_refinement import pose_points
    xy=np.array([pose_points()]*2); scores=np.full((2,17),.9)
    pose=ConfidencePose(SimpleNamespace(predict_many=lambda frames:[(xy,scores)]))
    model=DeduplicatedPMPose(pose,dedup_defaults())
    with Image.new('RGB',(100,100)) as image:
        result=model.predict(image,[[0,0,100,100]]*2)
    assert result[-1]['kept_indices']==[0] and result[0].shape==(1,17,2)


def test_generic_sam_and_optional_second_pass_keep_cross_frame_batches():
    from stash_ai.recognition.pose_masks import ConfidencePose
    class Raw:
        batch_size=4
        def __init__(self):self.calls=[]
        def predict_many(self,frames):
            self.calls.append(len(frames))
            return [(np.tile([[[20.,20.]]],(len(boxes),17,1)),np.full((len(boxes),17),.9))
                    for image,boxes in frames]
    class SAM:
        def __init__(self):self.calls=[]
        def predict_frames(self,requests):
            self.calls.append(len(requests))
            return [[(np.ones((im.height,im.width),bool),.99) for _ in prompts] for im,prompts in requests]
    for second in (False,True):
        raw=Raw();sam=SAM();model=PoseMasks(ConfidencePose(raw),sam,dict(defaults(),generic_second_pass=second))
        with Image.new('RGB',(50,50)) as a,Image.new('RGB',(50,50)) as b:
            results=model.predict_many([(a,[[0,0,50,50]]),(b,[[0,0,50,50]])])
        assert sam.calls==[2] and raw.calls==([2,2] if second else [2])
        assert all(not result[3][0]['visibility_measured'] for result in results)
