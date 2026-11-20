"""SAM masks for COCO17 estimators without model-specific visibility heads."""
import numpy as np
from .pose_refinement import RefinedPMPose


class ConfidencePose:
    """Common COCO17 contract, with explicitly unmeasured probability proxies."""
    def __init__(self,pose):self.pose=pose
    supports_masked_batches=True
    def __getattr__(self,name):return getattr(self.pose,name)
    def close(self):
        from .resident_models import release
        release(self.pose)
    def predict_many(self,frames):
        from contextlib import ExitStack
        from PIL import Image
        frames=list(frames)
        with ExitStack() as stack:
            jobs=[];positions=[]
            for frame,(image,boxes,*extra) in enumerate(frames):
                masks=extra[0] if extra else None
                if masks is None:
                    jobs.append((image,boxes));positions.append((frame,None))
                else:
                    if len(masks)!=len(boxes):raise ValueError('Masked pose requires one mask per person.')
                    with image.convert('RGB') as converted:rgb=np.asarray(converted).copy()
                    for index,(box,mask) in enumerate(zip(boxes,masks)):
                        if mask.shape!=(image.height,image.width):raise ValueError('Invalid pose conditioning mask.')
                        pixels=np.where(mask[:,:,None],rgb,(rgb*.25).astype(np.uint8))
                        conditioned=stack.enter_context(Image.fromarray(pixels))
                        jobs.append((conditioned,[box]));positions.append((frame,index))
            values=self.pose.predict_many(jobs)
            if len(values)!=len(jobs):raise ValueError('Invalid generic pose batch count.')
            result=[(np.empty((len(f[1]),17,2),np.float32),np.empty((len(f[1]),17),np.float32)) for f in frames]
            for (frame,index),(xy,score) in zip(positions,values):
                if index is None:result[frame]=(xy,score)
                else:result[frame][0][index],result[frame][1][index]=xy[0],score[0]
        return [(xy,score,dict(presence=score.copy(),visibility=score.copy())) for xy,score in result]
    def predict(self,image,bounds,masks=None):return self.predict_many([(image,bounds,masks)])[0]


class PoseMasks(RefinedPMPose):
    """Confidence proxies select prompts; they never become measured visibility."""
    def predict_many(self, frames):
        frames = list(frames)
        from .refinement_budget import begin
        begin(frames,self.settings)
        values = self.pose.predict_many(frames)
        predictions = [value if len(value)==3 else
                       (value[0],value[1],dict(presence=value[1].copy(),visibility=value[1].copy())) for value in values]
        return self.refine_many(frames,predictions)

    generic_estimator = True

    def _apply_second_pass(self,pending,result,records,original,refined):
        super()._apply_second_pass(pending,result,records,original,refined)
        for i,*_ in pending:
            records[i].update(point_evidence='confidence_proxy',visibility_measured=False)



def wrap(pose, config, device):
    from .sam_pose2seg import load
    from .resident_models import current, Stage
    settings=config.data['sapiensid']['pose']
    pose=ConfidencePose(pose)
    refined=None
    if settings['refinement']['enabled']:
        segmenter = (Stage('pose_segmenter', lambda: load(config, device), device)
                     if current() is not None else load(config, device))
        refined=PoseMasks(pose,segmenter,settings['refinement'],
            joint_quality=dict(min_keypoint_score=settings['keypoint_score_threshold']),
            frame_batch_size=config.data['execution']['sam_frame_batch_size'],
            queue_bytes=config.data['execution']['refinement_queue_bytes'])
    if settings['deduplication']['enabled']:
        from .pose_deduplication import DeduplicatedPMPose
        return DeduplicatedPMPose(pose,settings['deduplication'],refined)
    return refined or pose
