"""Fuse two anatomical COCO17 poses within one owned instance mask."""
import numpy as np

from .mask_candidates import inside

PARENT = {7: 5, 8: 6, 9: 7, 10: 8, 13: 11, 14: 12, 15: 13, 16: 14}
OPPOSITE = {7: 8, 8: 7, 9: 10, 10: 9, 13: 14, 14: 13, 15: 16, 16: 15}


def fuse(i, offset, original, refined, mask, box, settings):
    raw_xy, raw_scores, raw_probs = original
    new_xy, new_scores, new_probs = refined
    joints = new_xy[offset].copy()
    confidence = new_scores[offset].copy()
    evidence = {k: v[offset].copy() for k, v in new_probs.items()}
    diagonal = np.hypot(box[2]-box[0], box[3]-box[1])
    tolerance = max(settings.get('limb_peer_min_pixels',2.), diagonal * settings.get('limb_peer_distance_fraction',.035))
    peer_valid = (raw_scores > settings['min_keypoint_score']) & (
        raw_probs['visibility'] > settings['min_visibility']) & (raw_probs['presence'] > settings['min_presence'])
    peer_valid[i] = False
    choices = []

    def qualifies(point, score, probs, joint, source):
        if (score <= settings['min_keypoint_score'] or probs['presence'] <= settings['min_presence']
                or not (box[0] <= point[0] < box[2] and box[1] <= point[1] < box[3])):
            return False
        if (joint in PARENT and rival_chain(original, i, joint, point, tolerance) <= tolerance
                and (confidence[PARENT[joint]] <= 0 or
                     np.linalg.norm(point-joints[PARENT[joint]]) > settings.get('limb_contact_reach_ratio',5.)*tolerance)):
            return False
        nearest_peer = np.inf
        if joint in PARENT:
            pair = [joint, OPPOSITE[joint]]
            others = raw_xy[:,pair][peer_valid[:,pair]]
            if len(others):
                nearest_peer = np.linalg.norm(others-point, axis=1).min()
        if not inside(mask, [point])[0]:
            if nearest_peer <= tolerance or joint not in PARENT:
                return False
            # One pixel of boundary uncertainty is common on small VTT thumbnails.
            pixels = point + np.array([[0,-1],[0,1],[-1,0],[1,0]])
            if inside(mask, pixels).sum() < 2:
                return False
        if joint not in PARENT:
            return True
        parent = PARENT[joint]
        if confidence[parent] > 0:
            samples = np.linspace(joints[parent], point, 7)[1:-1]
            return inside(mask, samples).sum() >= 4
        if parent in (5,6,11,12):
            # At contact a hip or shoulder can fall just outside its own mask.
            # Two agreeing roots and an uninterrupted foreground segment support
            # the next joint even if the root itself was withheld.
            root = new_xy[offset,parent]
            if (new_scores[offset,parent] > settings['min_keypoint_score']
                    and raw_scores[i,parent] > settings['min_keypoint_score']
                    and np.linalg.norm(raw_xy[i,parent]-root) <= tolerance
                    and inside(mask,np.linspace(root,point,7)[1:]).all()):
                return True
        # A disconnected endpoint requires agreement of two independent crops.
        other_xy = raw_xy[i,joint] if source == 'masked' else new_xy[offset,joint]
        other_score = raw_scores[i,joint] if source == 'masked' else new_scores[offset,joint]
        other_probs, index = (raw_probs, i) if source == 'masked' else (new_probs, offset)
        return (nearest_peer > tolerance and probs['visibility'] > settings['min_visibility']
                and other_score > settings['min_keypoint_score']
                and other_probs['presence'][index,joint] > settings['min_presence']
                and other_probs['visibility'][index,joint] > settings['min_visibility']
                and np.linalg.norm(other_xy-point) <= tolerance)

    for joint in range(17):
        masked = {k:v[offset,joint] for k,v in new_probs.items()}
        first = {k:v[i,joint] for k,v in raw_probs.items()}
        contested = (joint in PARENT and rival_chain(original,i,joint,joints[joint],tolerance) <= tolerance)
        own_first = (contested and np.linalg.norm(raw_xy[i,joint]-joints[joint]) > tolerance
                     and rival_chain(original,i,joint,raw_xy[i,joint],tolerance) > tolerance
                     and qualifies(raw_xy[i,joint],raw_scores[i,joint],first,joint,'first'))
        if contested and not own_first:
            other = alternate(refined,offset,joint,(mask.shape[1],mask.shape[0]))
            if (other is not None and confidence[PARENT[joint]] > 0
                    and rival_chain(original,i,joint,other,tolerance) > tolerance
                    and np.linalg.norm(other-joints[PARENT[joint]]) < np.linalg.norm(joints[joint]-joints[PARENT[joint]])
                    and qualifies(other,confidence[joint]*masked['alt_strength'],masked,joint,'masked')):
                joints[joint] = other
                confidence[joint] *= masked['alt_strength']
        if not own_first and qualifies(joints[joint], confidence[joint], masked, joint, 'masked'):
            continue
        if qualifies(raw_xy[i,joint], raw_scores[i,joint], first, joint, 'first'):
            joints[joint] = raw_xy[i,joint]
            confidence[joint] = raw_scores[i,joint]
            for key in evidence:
                evidence[key][joint] = first[key]
            choices.append(joint)
        else:
            joints[joint], confidence[joint] = -1, 0
            for values in evidence.values():
                values[joint] = 0
    return joints, confidence, evidence, choices


def rival_chain(original, owner, joint, point, tolerance):
    """Return the nearest endpoint supported by a visible, distinct rival chain."""
    xy, scores, probs = original
    parent = PARENT[joint]
    best = np.inf
    for other in range(len(xy)):
        if other == owner:
            continue
        for side in (joint, OPPOSITE[joint]):
            root = PARENT[side]
            if (min(scores[other,side], scores[other,root], probs['visibility'][other,side],
                    probs['presence'][other,side], probs['presence'][other,root]) <= .5
                    or np.linalg.norm(xy[owner,parent]-xy[other,root]) <= 2*tolerance):
                continue
            best = min(best, np.linalg.norm(point-xy[other,side]))
    return best


def alternate(original, i, joint, size):
    evidence = original[2]
    if 'alt_strength' not in evidence or evidence['alt_strength'][i,joint] < .55:
        return None
    return np.array([evidence['alt_x'][i,joint]*size[0],evidence['alt_y'][i,joint]*size[1]])


def maskless(original, i, box, size, result, settings):
    xy, scores, probs = original
    if len(xy) < 2:
        return []
    limit = max(settings.get('limb_peer_min_pixels',2.), settings.get('limb_peer_distance_fraction',.035)*np.hypot(box[2]-box[0],box[3]-box[1]))
    withheld = []
    for k,parent in PARENT.items():
        point = xy[i,k]
        if scores[i,k] <= 0 or rival_chain(original,i,k,point,limit) > limit:
            continue
        root = xy[i,parent]
        tip_distance = np.linalg.norm(point-root)
        # Retain credible contact when both people can anatomically reach it.
        if scores[i,parent] > settings['min_keypoint_score'] and tip_distance <= limit*settings.get('limb_contact_reach_ratio',5.):
            continue
        other = alternate(original,i,k,size)
        if (other is not None and scores[i,parent] > settings['min_keypoint_score'] and
                box[0]<=other[0]<box[2] and box[1]<=other[1]<box[3] and
                np.linalg.norm(other-root) < tip_distance and rival_chain(original,i,k,other,limit)>limit):
            result[0][i,k] = other
            result[1][i,k] *= probs['alt_strength'][i,k]
        else:
            result[0][i,k],result[1][i,k] = -1,0
            for values in result[2].values():
                values[i,k] = 0
            withheld.append(k)
    return withheld


def conditioning(image, jobs, original):
    """Darken confirmed rival limb pixels outside our validated foreground."""
    import cv2
    xy, scores, probs = original
    result = []
    for i,box,mask,_ in jobs:
        foreign = np.zeros(mask.shape,np.uint8) if len(xy)>1 else None
        width = max(2,round(.03*np.hypot(box[2]-box[0],box[3]-box[1])))
        for j in range(len(xy)):
            if i==j:
                continue
            for k,p in PARENT.items():
                if (min(scores[j,k],scores[j,p],probs['visibility'][j,k],probs['visibility'][j,p])>.75
                        and np.isfinite(xy[j,[p,k]]).all() and (xy[j,[p,k]]>=0).all()
                        and not inside(mask,xy[j,[p,k]]).any()):
                    cv2.line(foreign,tuple(xy[j,p].round().astype(int)),tuple(xy[j,k].round().astype(int)),1,width)
        if foreign is not None:
            # Crop preparation uses only pixels near this person's crop.
            foreign[mask] = 0
        result.append((mask,foreign.astype(bool)) if foreign is not None and foreign.any() else mask)
    return result
