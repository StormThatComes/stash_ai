"""Pure COCO17 preparation shared by inference and retained quality admission."""
import math
import numpy as np


def adjust_pose(points, *, source_indices=None):
    """Match upstream image-left/right canonicalization; keep missing pairs."""
    points = [list(p) for p in points]
    for left, right in zip((1,3,5,7,9,11,13,15), (2,4,6,8,10,12,14,16)):
        a, b = points[left], points[right]
        av, bv = a != [-1., -1.], b != [-1., -1.]
        if (av and bv and a[0] < b[0]) or (av and not bv and a[0] < .5) or (bv and not av and b[0] > .5):
            points[left], points[right] = b, a
            if source_indices is not None:
                source_indices[left], source_indices[right] = source_indices[right], source_indices[left]
    # Upstream drops nose/eye outliers when both ears and shoulders are known.
    if all(points[i] != [-1., -1.] for i in (3,4,5,6)):
        lo, hi = min(points[i][0] for i in (3,4,5,6)), max(points[i][0] for i in (3,4,5,6))
        for i in (0,1,2):
            if points[i] != [-1., -1.] and not lo <= points[i][0] <= hi:
                points[i] = [-1., -1.]
    return points



def body_points(person, *, anatomical=False):
    """Encoder's surviving crop points; optionally restore detector-side labels."""
    bounds=person['bounds'];points=person['keypoints']
    if (len(bounds)!=4 or any(type(v) not in (int,float) or not math.isfinite(v) for v in bounds)
            or len(points)!=17):raise ValueError('Invalid retained person geometry.')
    left,top,right,bottom=bounds
    if left>=right or top>=bottom:raise ValueError('Empty retained person geometry.')
    values=np.asarray(points,dtype='float64')
    if values.shape!=(17,2) or not np.isfinite(values).all():raise ValueError('Invalid retained pose points.')
    side=max(right-left,bottom-top)
    padx,pady=(side-(right-left))//2,(side-(bottom-top))//2
    normalized=[[(x-left+padx)/side,(y-top+pady)/side] if x>=0 and y>=0 else [-1.,-1.] for x,y in values]
    source_indices=list(range(17)) if anatomical else None
    adjusted=adjust_pose(normalized, source_indices=source_indices)
    result=np.asarray([[x*side-padx,y*side-pady] if x>=0 else [-1.,-1.] for x,y in adjusted],dtype='float32')
    outside=((result<0)|(result>=[right-left,bottom-top])).any(axis=1)
    result[outside]=-1
    if anatomical:
        # Keep exactly the encoder's surviving points, with detector-side labels.
        # Sorting by image x would otherwise turn a left arm into a right arm.
        result=result[np.argsort(source_indices)]
    return result.tolist()


def used_keypoints(person):
    return sum(x>=0 and y>=0 for x,y in body_points(person))
