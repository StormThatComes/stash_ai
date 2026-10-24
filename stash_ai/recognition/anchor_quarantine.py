"""An optional attachment-only core without isolated unsupported anchors.

Original strict samples stay in the bank as rivals and merge witnesses. This
creates a query route, never edits membership or certifies a cleaned identity.
"""
from .clustering import Group
from .database import encoded
from .global_views import IdentityView
from .view_prototypes import _Cluster, best_cohort, pose_view


def views(db, groups, config, usage):
    if not config.data['clustering'].get('anchor_outlier_quarantine'):
        return ()
    import numpy as np
    grouping=config.data['clustering']
    minimum=max(3,grouping['view_min_support'],config.data['recognition']['min_merge_support'])
    result=[]
    for group in groups:
        if len(group.sample_ids)!=len(group.samples) or len(group.samples)<=minimum:
            continue
        records={r['id']:r for r in db.execute('SELECT id,scan,seconds,geometry,quality FROM observations '
            'WHERE id IN (SELECT value FROM json_each(?))', (encoded(group.sample_ids),))}
        if len(records)!=len(group.sample_ids):
            continue
        slots=[];assigned={}
        for identifier,sample in zip(group.sample_ids,group.samples):
            row=records[identifier]
            index=best_cohort(slots,row,sample,grouping['view_cohesion_threshold'])
            if index is None:
                index=len(slots);slots.append(_Cluster(pose_view(row['geometry']),identifier))
            slots[index].add(row,sample,maximum=128)
            assigned[identifier]=index
        supported={i for i,s in enumerate(slots) if len(s.cues)>=minimum}
        if not supported:
            continue
        supported_samples=[v for i,v in zip(group.sample_ids,group.samples) if assigned[i] in supported]
        excluded={i for i,v in zip(group.sample_ids,group.samples) if assigned[i] not in supported and
                  all(float(np.dot(v,s))<grouping['cohesion_threshold'] for s in supported_samples)}
        if not excluded:
            continue
        retained=[(i,v) for i,v in zip(group.sample_ids,group.samples) if i not in excluded]
        ids,samples=zip(*retained)
        # Complete original-witness proof, including small non-isolated looks.
        if any(float(np.dot(a,b))<grouping['cohesion_threshold'] for a in samples for b in samples):
            continue
        cues={(records[i]['scan'],records[i]['seconds']) for i in ids}
        if len(cues)<minimum:
            continue
        centre=np.mean(samples,axis=0)
        index=max(range(len(samples)),key=lambda i:(float(np.dot(samples[i],centre)),ids[i]))
        medoid=ids[index]
        usage.add(samples,kind='Quarantined core witnesses')
        proof=dict(format=1,kind='anchor_outlier_quarantine',excluded_ids=sorted(excluded),
                   original_ids=list(group.sample_ids),minimum_support=minimum,attachment_only=True)
        result.append(IdentityView(Group(group.identifier,(samples[index],),True,(medoid,)),
            pose_view(records[medoid]['geometry']),medoid,ids,len(cues),samples,'quarantined_core',proof))
    return tuple(result)
