"""Frozen complete-link body modes; no unsupported witness is discarded."""
import numpy as np
from .independent_evidence import Votes


def body_modes(rows,config, *, partial=False):
    from .hybrid_references import body_vector
    threshold=config.data['sapiensid']['matching']['person_threshold']
    minimum=config.data['sapiensid']['matching']['min_support']
    clusters=[]
    for row in rows:
        value=body_vector(row)
        compatible=[c for c in clusters if all(float(np.dot(value,v))>=threshold for _,v in c)]
        if len(compatible)!=1:
            clusters.append([(row,value)])
        else:compatible[0].append((row,value))
    result=[]
    for cluster in clusters:
        votes=Votes()
        for row,_ in cluster:votes.add(row)
        # All witnesses must belong to an independently confirmed mode.
        if len(votes)<minimum:
            if partial:
                continue
            return ()
        result.append(np.asarray([v for _,v in cluster]))
    return tuple(result)


def covered_score(left,right,settings):
    from .hybrid_evidence import scores
    if not left or not right:return None
    measurements=[[scores(a,b) for b in right] for a in left]
    threshold=settings['person_threshold']
    edges=[m for row in measurements for m in row if m['minimum']>=threshold]
    if (any(not any(m['minimum']>=threshold for m in row) for row in measurements) or
        any(not any(row[j]['minimum']>=threshold for row in measurements) for j in range(len(right)))):
        return None
    return dict(minimum=min(m['minimum'] for m in edges),mean=min(m['mean'] for m in edges),
                mode_coverage=[len(left),len(right)],strategy='all_supported_modes_covered')


def competitor_score(left,right,fallback):
    from .hybrid_evidence import scores
    if left.body_modes and right.body_modes:
        return max(scores(a,b)['mean'] for a in left.body_modes for b in right.body_modes)
    return fallback


def query_score(query,bank,settings,modes=()):
    from .hybrid_evidence import scores
    measurements=[dict(scores(query,mode),mode_index=i) for i,mode in enumerate(modes or (bank,)) if len(mode)]
    admitted=[m for m in measurements if m['minimum']>=settings['person_threshold']]
    return max(admitted or measurements,key=lambda m:m['mean'],default=None)


def guard_modes(bank,threshold):
    """Partition veto witnesses without inventing support or dropping outliers."""
    groups=[]
    for value in bank:
        target=next((g for g in groups if all(float(np.dot(value,v))>=threshold for v in g)),None)
        if target is None:groups.append([value])
        else:target.append(value)
    return tuple(np.asarray(g) for g in groups)


class RivalBounds:
    """Screen only rivals which cannot improve the exact reported runner-up."""
    def __init__(self,budget):
        self.means={}
        self.remaining=budget

    def get(self,owner):
        if owner.person in self.means:return self.means[owner.person]
        from .disk_body_modes import Matrix
        if not owner.body_modes:return None
        cost=sum(128+mode.shape[1]*8 for mode in owner.body_modes)
        if cost>self.remaining:return None
        means=[]
        for mode in owner.body_modes:
            if not len(mode) or len(mode)>1000000:return None
            if isinstance(mode,Matrix):
                total=np.zeros(mode.shape[1],dtype='float64');weight=0.
                for start in range(0,len(mode),32):
                    block=slice(start,start+32)
                    weights=mode.weights(block)
                    total+=(mode[block].astype('float64')*weights[:,None]).sum(axis=0)
                    weight+=weights.sum()
                if not weight:return None
                means.append(total/weight)
            else:means.append(mode.mean(axis=0,dtype='float64'))
        self.remaining-=cost
        self.means[owner.person]=means
        return means

    def upper(self,left,right):
        a,b=self.get(left),self.get(right)
        if not a or not b:return None
        # Unit vectors, float64 products and at most 1M rows per mean.
        return max(float(np.dot(x,y)) for x in a for y in b)+1e-6
