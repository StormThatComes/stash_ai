"""Bounded spatial checks for missing VTT observations."""
import json
import math


def blockers(db, scan, contract):
    cues = {}
    for cue, raw, kind in db.execute(
        "SELECT cue,geometry,'body' FROM body_observations WHERE scan=? AND contract=? "
        "UNION ALL SELECT cue,geometry,'face' FROM observations WHERE scan=?", (scan,contract,scan)
    ):
        value = json.loads(raw)
        box = value.get('bounds') if kind == 'body' and isinstance(value,dict) else None
        if kind == 'face' and isinstance(value,list) and len(value)>=4:
            x,y,w,h = value[:4]
            box = [x,y,x+w,y+h]
        valid = box and len(box)==4 and all(math.isfinite(v) for v in box) and box[0]<box[2] and box[1]<box[3]
        cues.setdefault(cue,[]).append(box if valid else None)
    return cues


def clear(start, end, cues):
    # Sweep the entire corridor. Unknown geometry and grazing rivals both veto.
    x,y = min(start[0],end[0]),min(start[1],end[1])
    r,b = max(start[2],end[2]),max(start[3],end[3])
    pad = .1*max(r-x,b-y)
    return all(peer is not None and (peer[2]<x-pad or peer[0]>r+pad or
                                     peer[3]<y-pad or peer[1]>b+pad)
               for boxes in cues for peer in boxes)


def camera_continuity(before, after):
    """Verify a bounded camera warp on textured background in three corners."""
    import cv2
    import numpy as np
    a,b = (np.frombuffer(bytes.fromhex(raw),np.uint8).reshape(48,48) for raw in (before,after))
    mask = np.full(a.shape,255,np.uint8)
    mask[10:-10,10:-10] = 0
    points = cv2.goodFeaturesToTrack(a,48,.03,3,mask=mask)
    if points is None or len(points)<8:
        return False
    dest,ok,_ = cv2.calcOpticalFlowPyrLK(a,b,points,None,winSize=(9,9),maxLevel=1)
    if dest is None:
        return False
    original,moved = points[ok.ravel()!=0,0],dest[ok.ravel()!=0,0]
    if len(original)<8:
        return False
    transform,_ = cv2.estimateAffinePartial2D(original,moved,method=cv2.LMEDS)
    if transform is None:
        return False
    scale = np.linalg.norm(transform[0,:2])
    if not .8<=scale<=1.25 or abs(transform[0,1])>.15 or max(abs(transform[:,2]))>12:
        return False
    warped = cv2.warpAffine(a,transform,(48,48),borderMode=cv2.BORDER_CONSTANT)
    real = cv2.warpAffine(np.ones_like(a),transform,(48,48),flags=cv2.INTER_NEAREST).astype(bool)
    agree = 0
    for y in (2,32):
        for x in (2,32):
            region = np.s_[y:y+14,x:x+14]
            valid = real[region]
            left,right = warped[region][valid],b[region][valid]
            if len(left)>=90 and min(left.std(),right.std())>=12 and np.abs(left.astype(float)-right).mean()<=9:
                agree += 1
    return agree>=3
