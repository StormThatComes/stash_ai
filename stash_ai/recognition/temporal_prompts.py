"""Use at most one preceding VTT cue as a source of missing SAM torso prompts."""
import math

import numpy as np

TORSO = (5, 6, 11, 12)


def capture(image, bounds, points, cores):
    signature = getattr(image, '_recognition_cue_signature', None)
    scan = getattr(image, '_recognition_scan', None)
    if scan is None or signature is None or signature[1] is None:
        return None
    people = [(tuple(box), {k: points[i,k].copy() for k in TORSO if k in cores[i]})
              for i,box in enumerate(bounds)]
    return (scan, *signature[:3]), people


def adjacent(previous, current):
    if not previous or not current:
        return False
    (scan, cue, shot, pixels), _ = previous
    (other, index, group, image), _ = current
    if (other, index, group) != (scan, cue+1, shot):
        return False
    from .perspective_shots import change
    def decode(value):
        return np.frombuffer(bytes.fromhex(value), dtype='uint8').reshape(16,3).astype(float)
    return change(decode(pixels),decode(image),True)=='continuous'


def borrow(previous, current, i, indices, minimum, record):
    if len(indices)>=minimum or not adjacent(previous,current):
        return indices
    box, joints = current[1][i]
    peers = [other for j,(other,_) in enumerate(current[1]) if j!=i]
    old_people = previous[1]
    def intersects(a,b):
        return min(a[2],b[2])>max(a[0],b[0]) and min(a[3],b[3])>max(a[1],b[1])
    # A crowded or interleaved torso cannot determine which way to transport a joint.
    if any(intersects(box,b) for b in peers):
        return indices
    paths = [(b,old) for b,old in old_people if intersects(box,b)]
    if len(paths)!=1:
        return indices
    old_box,old = paths[0]
    stable = [k for k in joints if k in old and k in indices]
    if not stable or len(old)<2:
        return indices
    tolerance = max(3., .09*math.hypot(box[2]-box[0],box[3]-box[1]))
    delta = np.median([joints[k]-old[k] for k in stable],axis=0)
    if (any(np.linalg.norm(joints[k]-old[k]-delta)>tolerance for k in stable) or
            max(abs(np.asarray(old_box)+np.tile(delta,2)-box))>tolerance):
        return indices
    added = []
    for k in TORSO:
        if k in old and k not in indices:
            x,y = old[k]+delta
            if box[0]<x<box[2] and box[1]<y<box[3]:
                added.append((k,[float(x),float(y)]))
                if len(indices)+len(added)>=minimum:
                    record['temporal_prompts'] = added
                    record['temporal_source_cue'] = previous[0][1]
                    return np.append(indices,[k for k,_ in added])
    return indices
