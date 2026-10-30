"""The local cohort algorithm over shared arrays, with no worker SQLite copy."""
from .resource_errors import ResourceLimit
import numpy as np
from .independent_evidence import Votes
from .pose_policy import is_anchor
from .view_prototypes import _Cluster, best_cohort, coverage_threshold, witness_limit
from .reference_usage import ReferenceUsage
from .local_cohorts import _check_cancelled


def priorities(values, settings):
    grouping=settings['clustering']
    minimum=settings['recognition']['min_identity_support']
    fields=('id','scan','cue','seconds','geometry','quality','dimensions','embedding')
    rows=[dict(zip(fields,value)) for value in values]
    slots, assigned, samples = [], {}, {}
    usage=ReferenceUsage(grouping['max_reference_bytes'])
    # Parent SQL supplied exactly the serial canonical input order.
    for row in rows:
        _check_cancelled()
        blob=row['embedding'];dimensions=row['dimensions']
        if type(dimensions) is not int or not 1<=dimensions<=1024 or len(blob)!=dimensions*4:
            raise ValueError('Invalid retained face embedding shape/length.')
        sample=np.frombuffer(blob,dtype='<f4')
        if not np.isfinite(sample).all() or abs(float(np.linalg.norm(sample))-1)>.001:
            raise ValueError('Invalid retained face embedding values/normalization.')
        samples[row['id']]=sample
        slot=best_cohort(slots,row,sample,coverage_threshold(grouping))
        if slot is None:
            if len(slots)>=grouping['max_people_per_lookup']:
                raise ResourceLimit('Local appearance cohorts exceed configured person budget.')
            slot=len(slots);slots.append(_Cluster('other',row['id']))
        cohort=slots[slot];before=len(cohort.examples)
        cohort.add(row,sample,maximum=witness_limit(grouping))
        usage.add_counts((len(cohort.examples)-before)*sample.nbytes,len(cohort.examples)-before,
                         kind='Local appearance witnesses')
        assigned[row['id']]=slot
    anchors=[Votes() for _ in slots]
    for row in sorted(rows,key=lambda r:r['id']):
        _check_cancelled()
        slot=best_cohort(slots,row,samples[row['id']],coverage_threshold(grouping))
        slot=assigned[row['id']] if slot is None else slot
        assigned[row['id']]=slot
        if is_anchor(row['quality'],grouping):
            anchors[slot].add(row)
    if not slots:
        return ()
    centres=[s.medoid()[0] for s in slots]
    centre=np.mean(centres,axis=0,dtype='float64')
    remaining=set(range(len(slots)));placed={};connected=[False]*len(slots)
    ranks=[(len(anchors[i])>=minimum,max(slots[i].qualities,default=0.),
            -float(np.dot(c,centre)),tuple(float(x) for x in c),slots[i].pose)
           for i,c in enumerate(centres)]
    while remaining:
        _check_cancelled()
        winner=max(remaining,key=lambda i:(connected[i],)+ranks[i])
        placed[winner]=len(placed);remaining.remove(winner)
        for i in remaining:
            if not connected[i] and float(np.dot(centres[i],centres[winner]))>=settings['recognition']['match_threshold']:
                connected[i]=True
    return tuple((identifier,placed[slot]) for identifier,slot in sorted(assigned.items()))
