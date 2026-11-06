"""Bounded matrix screens may certify rejection; acceptance remains scalar."""
import numpy as np


def contradiction(evidence,left,right,threshold,kind):
    if not left or not right:return None
    if len(left)*len(right)<64:return None
    from stash_ai.runtime.cancellation import current
    cancellation=current()
    for i in range(0,len(left),8):
        if cancellation is not None:cancellation.check()
        a=np.stack([evidence.feature(r,kind) for r in left[i:i+8]])
        for j in range(0,len(right),64):
            b=np.stack([evidence.feature(r,kind) for r in right[j:j+64]])
            if a.shape[1]!=b.shape[1]:raise ValueError('Incompatible face namespaces in body consolidation.')
            scores=a@b.T
            evidence.stats['certificate_products']=evidence.stats.get('certificate_products',0)+scores.size
            # Conservative error band for two float32 reductions on unit vectors.
            slack=8*a.shape[1]*np.finfo(np.float32).eps
            index=np.unravel_index(int(np.argmin(scores)),scores.shape)
            if float(scores[index])+slack<threshold:
                evidence.charge(1,kind)
                score=float(np.dot(a[index[0]],b[index[1]]))
                if score<threshold:
                    evidence.stats['rejection_certificates']=evidence.stats.get('rejection_certificates',0)+1
                    return score
    return None


def pairs(candidates,bank,config,evidence):
    from .body_global_search import _pair_result
    settings=config.data['sapiensid']['global_matching']
    for a,b in candidates:
        first,second=bank[a],bank[b]
        mean,minimum=evidence.compare(first['refs'],second['refs']);blocked=False;bounded=False
        for left in first['cores']:
            for right in second['cores']:
                witness=contradiction(evidence,left,right,settings['cohesion_threshold'],'body')
                if witness is not None:
                    minimum=min(minimum,witness);blocked=bounded=True;break
                average,low=evidence.compare(left,right)
                minimum=min(minimum,low)
                blocked |= average<settings['match_threshold'] or low<settings['cohesion_threshold']
            if bounded:break
        face=None;face_bound=False
        if first['faces'] and second['faces']:
            face=contradiction(evidence,first['faces'],second['faces'],config.data['sapiensid']['matching']['face_floor'],'face')
            face_bound=face is not None
            if face is None:face=evidence.compare(first['faces'],second['faces'],'face')[1]
        result=_pair_result(a,b,bank,config,mean=mean,minimum=minimum,face_min=face,core_blocked=blocked)
        if bounded:result['minimum_is_upper_bound']=True
        if face_bound:result['face_minimum_is_upper_bound']=True
        yield result
