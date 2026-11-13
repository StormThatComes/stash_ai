"""Scheduling proofs using deterministic models, independent of CUDA hardware."""
from contextlib import contextmanager
import numpy as np
import pytest
from PIL import Image
from tests.test_recognition_pose_refinement import Pose, Segmenter, BOX, prediction, pose_points
from stash_ai.recognition.pose_refinement import RefinedPMPose
from stash_ai.recognition.pose_refinement_config import defaults


class BatchedPose(Pose):
    supports_masked_batches=True
    def predict_many(self, frames):
        if frames and len(frames[0])==3:
            self.calls.append(('masked_batch',sum(len(bounds) for _,bounds,_ in frames)))
            return [prediction([pose_points()+[box[0]-20,0] for box in boxes]) for _,boxes,_ in frames]
        return super().predict_many(frames)
    def predict(self,image,bounds,masks=None):
        return self.predict_many([(image,bounds,masks)])[0]


class BatchedSegmenter(Segmenter):
    def predict_frames(self, frames):
        self.windows=getattr(self,'windows',[])+[len(frames)]
        results=[]
        for image,prompts in frames:
            with self.frame(image):
                results.append([self.predict(p,n) for p,n in prompts])
        return results


def test_second_pass_spans_frames_matches_scalar_and_flushes_tail():
    with Image.new('RGB',(160,100)) as image:
        frames=[(image,[BOX]),(image,[]),(image,[BOX]),(image,[BOX]),(image,[BOX])]
        pose=BatchedPose();sam=BatchedSegmenter();model=RefinedPMPose(pose,sam,defaults())
        rows=model.predict_many(frames)
        assert pose.calls==[('first',5),('masked_batch',2),('masked_batch',1),('masked_batch',1)]
        assert sam.windows==[3,1]
        scalar=RefinedPMPose(BatchedPose(),Segmenter(),defaults())
        expected=scalar.predict_many(frames)
        for actual,reference in zip(rows,expected):
            np.testing.assert_array_equal(actual[0],reference[0])
            np.testing.assert_array_equal(actual[1],reference[1])
            assert actual[3]==reference[3]


def test_oversized_frame_uses_scalar_path_and_failure_propagates():
    with Image.new('RGB',(160,100)) as image:
        sam=BatchedSegmenter();model=RefinedPMPose(BatchedPose(),sam,defaults(),queue_bytes=1)
        result=model.predict_many([(image,[BOX])])
        assert result[0][3][0]['status']=='refined' and not hasattr(sam,'windows')
        sam.failure=True
        with pytest.raises(RuntimeError,match='inference failed'):
            model.predict_many([(image,[BOX])])
        assert sam.frames==sam.resets==2


def test_deferred_conflict_cannot_restore_a_quality_mask():
    model=RefinedPMPose(BatchedPose(),BatchedSegmenter(),defaults())
    model.pose.batch_size=1  # First mask is finalized before its rival is seen.
    model.settings.update(masked_identity_crop=True,max_foreign_coverage=1.)
    def shared_mask(points,negative=()):
        mask=np.zeros((100,150),dtype=bool)
        x=20 if points[0,0]<65 else 67
        mask[10:65,x:x+40]=True
        mask[65:90,35:97]=True
        return mask,.99
    model.segmenter.predict=shared_mask
    with Image.new('RGB',(150,100)) as image:
        rows=model.predict_many([(image,[[20,10,80,90],[64,10,125,90]])]*2)
    for row in rows:
        assert all(r['mask_conflict'] for r in row[3])
        assert all(not {'encoder_mask','overlap_mask'}.intersection(r) for r in row[3])


def test_sam_batches_equal_length_prompts_and_retries_images_without_reordering():
    torch=pytest.importorskip('torch')
    from stash_ai.recognition.sam_pose2seg import Segmenter as RealSegmenter
    class Predictor:
        def __init__(self):self.calls=[];self.resets=0;self.groups=[]
        def reset_predictor(self):self.resets+=1
        def set_image_batch(self,arrays):
            self.calls.append(len(arrays))
            if len(arrays)>2:raise torch.cuda.OutOfMemoryError('simulated')
            self.arrays=arrays.copy()
        def _prep_prompts(self,coordinates,labels,box,mask,normalize,*,img_idx):
            assert normalize and labels.shape==coordinates.shape[:2]
            assert (labels>=0).all()  # No synthetic padding tokens.
            return None,coordinates,labels,None
        def _predict(self,coords,labels,boxes,mask,*,img_idx,**kwargs):
            self.groups.append((img_idx,len(coords),coords.shape[1]))
            h,w=self.arrays[img_idx].shape[:2]
            masks=torch.zeros(len(coords),1,h,w,dtype=torch.bool)
            for i,row in enumerate(coords):masks[i,0,int(row[0,1]),int(row[0,0])]=True
            return masks,torch.full((len(coords),1),.95),object()
    predictor=Predictor();model=RealSegmenter(predictor,'cuda',frame_batch_size=4,prompt_batch_size=2)
    with Image.new('RGB',(20,10)) as image:
        prompts=[(np.array([[1,2]]),[]),(np.array([[3,4],[5,6]]),[]),(np.array([[7,8]]),[])]
        results=model.predict_frames([(image,prompts)]*3)
    assert predictor.calls==[3,2,1] and predictor.resets==3
    assert len(results)==3
    assert predictor.groups==[(0,2,1),(0,1,2),(1,2,1),(1,1,2),(0,2,1),(0,1,2)]
    for row in results:
        assert [np.argwhere(mask).tolist() for mask,_ in row]==[[[2,1]],[[4,3]],[[8,7]]]


def test_fused_flip_equals_two_forwards_with_one_model_call():
    torch=pytest.importorskip('torch')
    from stash_ai.recognition.pmpose_pose import PMPose
    class Model:
        def __init__(self):self.calls=[]
        def __call__(self,x):
            self.calls.append(len(x))
            maps=torch.sigmoid(torch.nn.functional.avg_pool2d(x[:,:1],4)).expand(-1,17,-1,-1).contiguous()
            scores=maps.mean((2,3))
            return maps,scores,scores.clone(),scores.clone()
    rng=np.random.default_rng(14)
    with Image.fromarray(rng.integers(0,255,(100,180,3),dtype=np.uint8)) as image:
        models=[Model(),Model()]
        rows=[PMPose(m,device='cpu',batch_size=2,fused_flip=fused).predict(image,[[10,10,70,90],[80,10,150,90]])
              for m,fused in zip(models,[False,True])]
    assert models[0].calls==[2,2] and models[1].calls==[4]
    for a,b in zip(rows[0][:2],rows[1][:2]):np.testing.assert_allclose(a,b,atol=1e-5)
