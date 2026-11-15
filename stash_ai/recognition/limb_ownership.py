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
    tolerance = max(2., diagonal * .035)
    peer_valid = (raw_scores > settings['min_keypoint_score']) & (
        raw_probs['visibility'] > settings['min_visibility']) & (raw_probs['presence'] > settings['min_presence'])
    peer_valid[i] = False
    choices = []

    def qualifies(point, score, probs, joint, source):
        if (score <= settings['min_keypoint_score'] or probs['presence'] <= settings['min_presence']
                or not (box[0] <= point[0] < box[2] and box[1] <= point[1] < box[3])):
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
        if qualifies(joints[joint], confidence[joint], masked, joint, 'masked'):
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
