"""Conservative same-cue duplicate evidence, never similarity alone."""
import json
from .run_cache import metadata
import math


def duplicate(left, right, enabled=False):
    if left['geometry']==right['geometry'] and left['embedding']==right['embedding']:
        return True
    if not enabled:
        return False
    try:
        import numpy as np
        a=metadata(left['geometry']);b=metadata(right['geometry'])
        if len(a)!=15 or len(b)!=15 or not all(math.isfinite(v) for v in a+b):
            return False
        if min(a[2:4]+b[2:4])<=0:
            return False
        intersection=max(0,min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0])) * max(
            0,min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]))
        if intersection/(a[2]*a[3]+b[2]*b[3]-intersection)<.9:
            return False
        scale=min(math.hypot(*a[2:4]),math.hypot(*b[2:4]))
        if any(math.hypot(a[i]-b[i],a[i+1]-b[i+1])>.03*scale for i in range(4,14,2)):
            return False
        x=np.frombuffer(left['embedding'],dtype='<f4');y=np.frombuffer(right['embedding'],dtype='<f4')
        return x.shape==y.shape and x.size>0 and bool(float(np.dot(x,y))>=.995)
    except (ValueError,TypeError,KeyError,OverflowError,ZeroDivisionError):
        return False


def distinct_pairs(rows, enabled):
    """Consume joined geometry/embedding pairs without materializing all cues."""
    for row in rows:
        if not duplicate(dict(geometry=row[0],embedding=row[1]),dict(geometry=row[2],embedding=row[3]),enabled):
            return True
    return False
