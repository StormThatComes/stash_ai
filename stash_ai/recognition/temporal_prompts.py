"""Use at most one preceding VTT cue as a source of regional SAM prompts."""
import math

import numpy as np

TORSO = (5, 6, 11, 12)
# Mobile landmarks need current-cue support along the same anatomical chain.
SUPPORT = {7: (5,), 8: (6,), 13: (11,), 14: (12,),
           9: (7, 5), 10: (8, 6), 15: (13, 11), 16: (14, 12),
           0: (1, 2), 1: (0,), 2: (0,), 3: (1, 0), 4: (2, 0)}


def capture(image, bounds, points, valid):
    signature = getattr(image, '_recognition_cue_signature', None)
    scan = getattr(image, '_recognition_scan', None)
    if scan is None or signature is None or signature[1] is None:
        return None
    people = [(tuple(box), {int(k): points[i,k].copy() for k in np.flatnonzero(valid[i])})
              for i,box in enumerate(bounds)]
    return (scan, *signature[:3], bool(len(signature)>4 and signature[4])), people


def adjacent(previous, current):
    if not previous or not current:
        return False
    scan,cue,shot,pixels = previous[0][:4]
    other,index,group,image = current[0][:4]
    if other != scan or group != shot or abs(index-cue)!=1:
        return False
    later = current if index>cue else previous
    from .cue_continuity import continuous
    return continuous(pixels,image,verified=len(later[0])>4 and later[0][4])


def borrow(previous, current, i, indices, minimum, record, *, local=None):
    if not previous or not current:
        return indices
    box, joints = current[1][i]
    sparse = len(indices) < minimum
    # A complete face does not fill a missing shoulder or hip region.
    wanted = [k for group in (TORSO[:2], TORSO[2:]) for k in group
              if k not in joints and (sparse or all(other not in joints for other in group))]
    local = local or {}
    wanted.extend(k for k, chain in SUPPORT.items()
                  if k in local and all(root in joints for root in chain))
    if not wanted or not adjacent(previous, current):
        return indices
    peers = [other for j,(other,_) in enumerate(current[1]) if j != i]
    tolerance = max(3., .09*math.hypot(box[2]-box[0],box[3]-box[1]))
    def inside(point, rect, pad=0):
        x,y = point
        return rect[0]-pad <= x <= rect[2]+pad and rect[1]-pad <= y <= rect[3]+pad

    from .cue_continuity import intrudes
    peer_joints = {tuple(b): js for _,members in (previous,current) for b,js in members}
    def blocked(point,b):
        return intrudes(np.asarray(point),b,peer_joints.get(tuple(b),{}))

    def transport(owner, old_box, old):
        stable = [k for k in TORSO if k in joints and k in old and k in indices and
                  not any(blocked(joints[k],b) for b in peers)]
        if not stable or sum(k in old for k in TORSO) < 2:
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
        old_peers = [other for j, (other, _) in enumerate(previous[1]) if j != owner]
        def uncontested(xy, boxes):
            return all(not blocked(xy, other) for other in boxes)
        added = []
        for joint in wanted:
            if joint not in old or joint in indices:
                continue
            point = move(old[joint])
            if joint in SUPPORT:
                chain = SUPPORT[joint]
                if any(k not in old or not uncontested(old[k], old_peers)
                       or not uncontested(joints[k], peers) for k in chain):
                    continue
                # Follow the measured parent, then require a nearby peak from THIS cue.
                parent = chain[0]
                expected = joints[parent] + move(old[joint]) - move(old[parent])
                point = local[joint]
                if np.linalg.norm(point - expected) > tolerance:
                    continue
            path = np.linspace(old[joint],point,5)
            if (inside(point, box, -.01) and uncontested(old[joint], old_peers)
                    and all(uncontested(p, peers) for p in path)):
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
