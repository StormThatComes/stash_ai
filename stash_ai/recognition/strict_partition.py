"""Resolve independently established scene cores using one frozen partition.

Only complete mutually compatible components can consolidate. The existing
consolidation engine also preserves outsider competition and co-presence; no
merged output becomes another nomination's evidence. This pass supplies strict
thresholds, rather than the stronger thresholds for appearance-only views.
"""
from dataclasses import replace

from .clustering import Group, _groups, matching_policy
from .config import plain
from .global_views import IdentityView, merge_views
from .reference_usage import ReferenceUsage


def resolve(database, selected, config, *, progress=None):
    if not selected or not config.data['clustering']['anchor_view_matching']:
        return
    import numpy as np
    db=database.connection
    scope=db.execute('SELECT scope,namespace FROM scans WHERE id=?',(selected[0]['scan'],)).fetchone()
    usage=ReferenceUsage(config.data['clustering']['max_reference_bytes'],progress)
    groups=_groups(db,*scope,config.data['clustering'],usage=usage,stage='frozen_local_cores')
    if len(groups)<2:
        return
    summaries=[]
    for group in groups:
        centre=np.mean(group.samples,axis=0,dtype='float64')
        length=float(np.linalg.norm(centre))
        if not length:
            continue
        sample=np.asarray(centre/length,dtype='float32')
        usage.add((sample,),kind='Strict core summary')
        summaries.append(IdentityView(Group(group.identifier,(sample,),True,group.sample_ids),
            'other',group.sample_ids[0],group.sample_ids,len(group.samples),tuple(group.samples),'strict_core'))
    data=plain(config.data)
    data['recognition']['min_merge_support']=1 # Each local core already passed identity support.
    data['clustering'].update(view_match_threshold=data['recognition']['match_threshold'],
        view_cohesion_threshold=data['clustering']['cohesion_threshold'],
        view_ambiguity_margin=data['recognition']['ambiguity_margin'])
    merge_views(database,tuple(summaries),replace(config,data=data),matching_policy(config),
                evidence_field='strict_partition')
