"""One counterfactual pose pass for nearly coincident detections in contact."""
import numpy as np

from .limb_ownership import PARENT


def _recover_one(pose,image,bounds,prediction,reserves,settings):
    if len(bounds) < 2:
        return prediction, None
    from .pose_deduplication import _measurements, _compare
    boxes,valid,_ = _measurements(bounds,prediction,settings)
    points,scores,prob = prediction
    pairs = [(parent,i) for i,parent in reserves.items()]
    if not pairs:
        pairs = ((i,j) for i in range(len(bounds)) for j in range(i+1,len(bounds)))
    for i,j in pairs:
        intersection = np.maximum(0,np.minimum(boxes[i,2:],boxes[j,2:])-np.maximum(boxes[i,:2],boxes[j,:2])).prod()
        if not intersection:
            continue
        reason, pair = _compare(i,j,boxes,points,valid,settings)
        if (reason not in ('insufficient_shared_core','duplicate_core') or not
                settings.get('contact_min_iou',.70) < pair['box_iou'] < settings.get('contact_max_iou',.94)):
            continue
        scale = min(np.linalg.norm(boxes[n,2:]-boxes[n,:2]) for n in (i,j))
        paired = (scores[i]>.5)&(scores[j]>.5)
        same = np.linalg.norm(points[i]-points[j],axis=1) < max(3.,.035*scale)
        if np.count_nonzero(paired & same) < 7:
            continue
        from PIL import ImageDraw
        with image.convert('RGB') as residual:
            ring = np.asarray(residual)
            border = np.concatenate([ring[0],ring[-1],ring[:,0],ring[:,-1]])
            fill = tuple(np.percentile(border,50,axis=0).astype(int))
            draw = ImageDraw.Draw(residual)
            for tip,root in {**PARENT,5:6,6:12,12:11,11:5}.items():
                if min(scores[i,tip],scores[i,root]) > .45:
                    draw.line([tuple(points[i,root]),tuple(points[i,tip])],fill=fill,
                              width=max(3,round(.05*scale)))
            result = pose.predict_many([(residual,[bounds[i],bounds[j]])])[0]
        alternatives=[]
        for offset,owner,peer in ((0,i,j),(1,j,i)):
            xy,conf,evidence = result[0][offset],result[1][offset],result[2]
            visible = (conf>.5)&(evidence['presence'][offset]>.5)&(evidence['visibility'][offset]>.4)
            distinct = np.linalg.norm(xy-points[peer],axis=1) > max(4.,.065*scale)
            head = np.median(xy[:5],axis=0)
            peer_head = np.median(points[peer,:5],axis=0)
            core = [5,6,11,12]
            head_confirmed = (np.linalg.norm(head-peer_head)>.065*scale
                              and np.count_nonzero((conf[:5]>.42)&distinct[:5]) >= 2)
            torso_confirmed = np.count_nonzero(visible[core]&distinct[core]) >= 2
            new_limbs = int(np.count_nonzero((visible & distinct)[list(PARENT)]))
            if (head_confirmed or torso_confirmed) and new_limbs>=2:
                box = boxes[owner]
                inside = ((xy>=box[:2]) & (xy<box[2:])).all(axis=1)
                alternatives.append((int(np.count_nonzero(inside & distinct & (conf>.5))),new_limbs,offset,owner))
        if alternatives:
            _,_,offset,owner=max(alternatives)
            output=(points.copy(),scores.copy(),{key:value.copy() for key,value in prob.items()})
            output[0][owner],output[1][owner]=result[0][offset],result[1][offset]
            for key in output[2]:
                output[2][key][owner]=result[2][key][offset]
            return output, (owner,i,j)
        # The counterfactual pass is bounded to one pair per cue.
        break
    return prediction, None


def recover_pair(pose,image,bounds,prediction,reserves,settings):
    """Try both owners, prioritizing the most overlapping physical pair."""
    pairs = [(parent,i) for i,parent in reserves.items()] or [
        (i,j) for i in range(len(bounds)) for j in range(i+1,len(bounds))]
    def overlap(pair):
        a,b = (np.asarray(bounds[i]) for i in pair)
        area = np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
        return area/max(1,min(np.prod(a[2:]-a[:2]),np.prod(b[2:]-b[:2])))
    pairs.sort(key=overlap,reverse=True)
    budget = settings.get('max_contact_attempts',4)
    original=pose
    class Counted:
        def predict_many(self,frames):
            nonlocal budget
            budget-=1
            from .profiling import pipeline_count
            pipeline_count('contact_pose_model_attempts')
            return original.predict_many(frames)
    pose=Counted()
    for a,b in pairs:
        for first,second in ((a,b),(b,a)):
            if budget <= 0:
                return prediction,None
            result,recovered = _recover_one(pose,image,bounds,prediction,{second:first},settings)
            if recovered is not None:
                owner,_,_ = recovered
                return result,(owner,a,b)
        if budget and settings.get('contact_crop_variants',False):
            from .pose_crop_variants import Variant
            for mode in ('expanded','sideways'):
                if budget <= 0:
                    break
                result,recovered = _recover_one(Variant(pose,mode),image,bounds,prediction,{b:a},settings)
                if recovered is not None:
                    return result,(recovered[0],a,b)
    return prediction,None
