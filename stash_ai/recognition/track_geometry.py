"""Shared physical gates for VTT transitions in replay and solver probes."""
import math
import numpy as np


def torso_points(person):
    points = person.get("keypoints") or ()
    visibility = person.get("keypoint_visibility")
    return {i: np.asarray(points[i]) for i in (5, 6, 11, 12) if i < len(points)
            and min(points[i]) >= 0 and (visibility is None or visibility[i] >= .5)}


def rejection(ba, bb, ta, tb, *, translation=None):
    if translation is not None:
        dx,dy = translation
        bb = [bb[0]-dx,bb[1]-dy,bb[2]-dx,bb[3]-dy]
        tb = {i: point-np.asarray(translation) for i,point in tb.items()}
    common = sorted(ta.keys() & tb.keys())
    if len(common)>=3:
        a,b = (np.stack([points[i] for i in common]) for points in (ta,tb))
        span = min(np.linalg.norm(np.ptp(points,axis=0)) for points in (a,b))
        # Subtract common translation. One performer's torso cannot jump
        # to the partner while the remaining anatomical points stay still.
        offsets = b-a
        error = np.linalg.norm(offsets-np.median(offsets,axis=0),axis=1).max()
        if span>=6 and error > .65*span:
            return (4,'torso_discontinuity',float(error/span),.65)

    wa, ha = ba[2]-ba[0], ba[3]-ba[1]
    wb, hb = bb[2]-bb[0], bb[3]-bb[1]
    if min(wa,ha,wb,hb)<=0:
        return (2,'empty_box',min(wa,ha,wb,hb),0)
    scale = max(wa/wb,wb/wa,ha/hb,hb/ha)
    displacement = math.hypot((ba[0]+ba[2]-bb[0]-bb[2])/2,
                              (ba[1]+ba[3]-bb[1]-bb[3])/2)
    relative = displacement/min(math.hypot(wa,ha), math.hypot(wb,hb))
    if scale>1.5 or relative>.30:
        common = sorted(ta.keys() & tb.keys())
        if len(common)>=3:
            origin,destination = (np.stack([torso[i] for i in common]) for torso in (ta,tb))
            spans = [np.linalg.norm(points.max(axis=0)-points.min(axis=0)) for points in (origin,destination)]
            stable = (min(spans)>=6 and max(spans)<=1.35*min(spans) and
                      np.linalg.norm(destination-origin,axis=1).max() <= min(spans)*.25)
        else:
            stable = False
        if not stable:
            if scale>1.5:
                return (3,'box_scale',scale,1.5)
            return (4,'displacement',relative,.30)
    return None


def translation(first,second,left,right):
    from .perspective_shots import camera_shift
    size = first.get('frame_size')
    if not size or size != second.get('frame_size'):
        return None
    shift = camera_shift(left,right)
    if shift:
        return shift[0]*size[0]/4,shift[1]*size[1]/4
    return None
