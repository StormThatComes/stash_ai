"""SAM masks for COCO17 estimators without model-specific visibility heads."""
import numpy as np
from .pose_refinement import RefinedPMPose, retain_quality_mask


class ConfidencePose:
    """Common COCO17 contract, with explicitly unmeasured probability proxies."""
    def __init__(self,pose):self.pose=pose
    def __getattr__(self,name):return getattr(self.pose,name)
    def close(self):
        from .resident_models import release
        release(self.pose)
    def predict_many(self,frames):
        return [(xy,score,dict(presence=score.copy(),visibility=score.copy()))
                for xy,score in self.pose.predict_many(frames)]
    def predict(self,image,bounds):return self.predict_many([(image,bounds)])[0]


class PoseMasks(RefinedPMPose):
    """Confidence proxies select prompts; they never become measured visibility."""
    def predict_many(self, frames):
        frames = list(frames)
        values = self.pose.predict_many(frames)
        predictions = [value if len(value)==3 else
                       (value[0],value[1],dict(presence=value[1].copy(),visibility=value[1].copy())) for value in values]
        # Without a masked second estimator pass, temporal borrowed prompts cannot
        # be independently verified. Current-frame recovery still runs.
        results = []
        for (image, bounds), prediction in zip(frames, predictions):
            self.previous_cue = None
            results.append(self.refine(image, bounds, prediction))
        return results

    def _second_pass(self, image, pending, result, records, original):
        for i, box, mask, indices in pending:
            record = records[i]
            from .limb_ownership import fuse, PARENT
            # A single pass can remove foreign points, but cannot corroborate a
            # disconnected endpoint by comparing the prediction with itself.
            own=(original[0][i:i+1],original[1][i:i+1],{k:v[i:i+1] for k,v in original[2].items()})
            xy,score,prob,_=fuse(i,0,original,own,mask,box,self.joint_quality)
            for k,p in PARENT.items():
                if score[p]<=self.joint_quality['min_keypoint_score']:
                    xy[k],score[k]=-1,0
                    for value in prob.values():value[k]=0
            removed=np.flatnonzero((original[1][i]>0)&(score==0)).tolist()
            if removed:record['mask_ownership_withheld']=removed
            result[0][i],result[1][i]=xy,score
            for key in result[2]:result[2][key][i]=prob[key]
            retain_quality_mask(record, mask, box)
            if record.get('separated') and self.settings.get('masked_identity_crop'):
                record['encoder_mask'] = record.get('overlap_mask')
            record.update(status='masked', reason=None, bounds=box,
                          point_evidence='confidence_proxy', visibility_measured=False,
                          box_expanded=box != record['detector_bounds'])


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
