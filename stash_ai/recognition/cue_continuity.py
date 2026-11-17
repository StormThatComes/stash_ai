"""One continuity decision shared by contact recovery and identity tracks."""
import numpy as np


def continuous(left,right,*,verified=False):
    from .perspective_shots import change
    def decode(value):
        return np.frombuffer(bytes.fromhex(value),np.uint8).reshape(16,3).astype(float)
    return bool(verified or change(decode(left),decode(right),True)=='continuous')


def intrudes(point, box, joints):
    """Use reliable skeleton support when available; otherwise preserve box veto."""
    if not (box[0]-2<=point[0]<=box[2]+2 and box[1]-2<=point[1]<=box[3]+2):
        return False
    core=[joints[k] for k in (5,6,11,12) if k in joints]
    if len(core)<3:
        return True
    import cv2
    hull=cv2.convexHull(np.asarray(core,np.float32))
    tolerance=max(2.,.035*np.linalg.norm(np.subtract(box[2:],box[:2])))
    if cv2.pointPolygonTest(hull,tuple(map(float,point)),True)>=-tolerance:
        return True
    from .limb_ownership import PARENT
    for k,p in PARENT.items():
        if k not in joints or p not in joints:continue
        a,b=np.asarray(joints[p]),np.asarray(joints[k]); d=b-a
        fraction=np.clip(np.dot(point-a,d)/max(1.,np.dot(d,d)),0,1)
        if np.linalg.norm(point-(a+fraction*d))<=tolerance:return True
    return False
