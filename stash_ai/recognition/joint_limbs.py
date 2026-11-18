"""Conservative endpoint ownership across a simultaneous contact group."""
import numpy as np
from .limb_ownership import PARENT, OPPOSITE


def settle(result,bounds,records,settings):
    xy,scores,prob=result[:3]
    snapshot=xy.copy(); confidence=scores.copy(); remove=set()
    for i,box in enumerate(bounds):
        for k,parent in PARENT.items():
            if confidence[i,k]<=settings['min_keypoint_score']:
                continue
            for j in range(i):
                scale=min(np.linalg.norm(np.subtract(b[2:],b[:2])) for b in (box,bounds[j]))
                tolerance=max(settings.get('limb_peer_min_pixels',2.),
                              settings.get('limb_peer_distance_fraction',.035)*scale)
                for other in (k,OPPOSITE[k]):
                    root=PARENT[other]
                    if confidence[j,other]<=settings['min_keypoint_score'] or np.linalg.norm(snapshot[i,k]-snapshot[j,other])>tolerance:
                        continue
                    if np.linalg.norm(snapshot[i,parent]-snapshot[j,root])<=2*tolerance:
                        continue
                    def chain(owner,tip,p):
                        grand=PARENT.get(p)
                        if grand is None or min(confidence[owner,p],confidence[owner,grand])<=settings['min_keypoint_score']:
                            return None
                        proximal=np.linalg.norm(snapshot[owner,p]-snapshot[owner,grand])
                        distal=np.linalg.norm(snapshot[owner,tip]-snapshot[owner,p])
                        return distal/max(proximal,tolerance)
                    a,b=chain(i,k,parent),chain(j,other,root)
                    limit=settings.get('max_limb_length_ratio',3.)
                    if a is not None and a>limit and b is not None and b<=limit:
                        remove.add((i,k))
                    elif b is not None and b>limit and a is not None and a<=limit:
                        remove.add((j,other))
                    elif a is None or b is None:
                        remove.update(((i,k),(j,other)))
    for i,k in sorted(remove):
        xy[i,k],scores[i,k]=-1,0
        for value in prob.values():value[i,k]=0
        records[i].setdefault('joint_ownership_withheld',[]).append(k)
