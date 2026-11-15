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
    peers = [other for j,(other,_) in enumerate(current[1]) if j != i]
    tolerance = max(3., .09*math.hypot(box[2]-box[0],box[3]-box[1]))
    def inside(point, rect, pad=0):
        x,y = point
        return rect[0]-pad <= x <= rect[2]+pad and rect[1]-pad <= y <= rect[3]+pad

    def transport(owner, old_box, old):
        stable = [k for k in joints if k in old and k in indices and
                  not any(inside(joints[k],b,2) for b in peers)]
        if not stable or len(old)<2:
            return None
        before = np.array([old[k] for k in stable])
        after = np.array([joints[k] for k in stable])
        origin, destination = before.mean(axis=0), after.mean(axis=0)
        matrix = np.eye(2)
        if len(stable) >= 2:
            u,v = before-origin, after-destination
            length = (u*u).sum()
            if length < tolerance**2/2:
                return None
            a = (u*v).sum()/length
            b = (u[:,0]*v[:,1]-u[:,1]*v[:,0]).sum()/length
            if not .72 <= np.hypot(a,b) <= 1.4 or abs(math.atan2(b,a)) > math.pi/4:
                return None
            matrix = np.array([[a,b],[-b,a]])
        def move(xy):
            return (xy-origin) @ matrix + destination
        if np.max(np.linalg.norm(move(before)-after,axis=1)) > tolerance:
            return None
        left,top,right,bottom = old_box
        corners = move(np.array([[left,top],[right,top],[left,bottom],[right,bottom]]))
        predicted = np.r_[corners.min(axis=0),corners.max(axis=0)]
        if np.max(abs(predicted-np.array(box))) > tolerance:
            return None
        added = []
        for joint in TORSO:
            if joint not in old or joint in indices:
                continue
            point = move(old[joint])
            path = np.linspace(old[joint],point,5)
            if (inside(point,box,-.01) and not any(inside(p,b,2) for p in path for b in peers)
                    and not any(inside(old[joint],b,2) for j,(b,_) in enumerate(previous[1]) if j != owner)):
                added.append((joint,point.tolist()))
            if len(indices)+len(added)>=minimum:
                return added
        return None

    paths = []
    for owner,(old_box,old) in enumerate(previous[1]):
        if min(box[2],old_box[2])>max(box[0],old_box[0]) and min(box[3],old_box[3])>max(box[1],old_box[1]):
            added = transport(owner,old_box,old)
            if added:
                paths.append(added)
    if len(paths) == 1:
        record['temporal_prompts'] = paths[0]
        record['temporal_source_cue'] = previous[0][1]
        return np.append(indices,[k for k,_ in paths[0]])
    return indices
