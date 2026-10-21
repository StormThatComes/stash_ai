"""One generation of attachment views with two disjoint rooted confirmations.

Promotion is read-only and rebuilt from latest durable memberships. Neither a
corroborated view nor this replay's output can supply a promotion root. Keeping
the full source lineage makes lost ownership/support revoke the next capture.
"""
from dataclasses import dataclass,field
import math

from .clustering import vector,Group,co_presence_allows
from .database import encoded
from .global_views import IdentityView
from .view_prototypes import _Cluster,coverage_threshold,pose_matches,pose_view


@dataclass
class _Cohort:
    cluster: _Cluster
    roots: set
    pivot: object
    scene_witnesses: dict = field(default_factory=dict)


def _independent(roots):
    for i,left in enumerate(roots):
        for right in roots[i+1:]:
            if not set(left.support).intersection(right.support):
                return left,right
    return None


def enrich(db,config,scope,namespace,selected,history,usage):
    """Freeze extra query-only views; no membership or reference-role writes."""
    grouping,settings=config.data['clustering'],config.data['recognition']
    if not grouping['corroborated_view_enrichment'] or not history.groups or not history.views:
        return ()
    from .historical_references import History,rank,_source,parameters
    from .reference_compatibility import fallback_sql
    import numpy as np
    minimum=max(grouping['view_min_support'],settings['min_merge_support'])
    roots={v.medoid:v for v in history.views if v.authority in ('strict','verified_view')
           and v.support_count>=minimum}
    by_owner={}
    for v in roots.values():by_owner.setdefault(v.group.identifier,[]).append(v)
    by_owner={p:vs for p,vs in by_owner.items() if _independent(vs) is not None}
    if not by_owner:
        return ()
    # Promotion always uses the original rooted bank, even when called again.
    bank=History(history.groups,tuple(v for v in history.views if v.authority in ('strict','verified_view')),
                 history.people,history.context)
    references=bank.groups+tuple(v.group for v in bank.views)
    threshold=max(settings['match_threshold'],grouping['view_match_threshold'])
    margin=max(settings['ambiguity_margin'],grouping['view_ambiguity_margin'])
    radius=max(coverage_threshold(grouping),math.sqrt((1+grouping['view_cohesion_threshold'])/2))
    rows=db.execute('SELECT m.person,o.*,s.scene '+
        _source(config,fallback_sql(db,config,namespace),attachment=True)+
        ' ORDER BY m.person,o.embedding,o.geometry,o.id',parameters(selected,config,scope,namespace))
    output,owner,slots,scratch=[],None,[],0

    def emit():
        if owner is None:return
        strict_ids=tuple(i for g in bank.groups if g.identifier==owner for i in g.sample_ids)
        for item in slots:
            cluster=item.cluster
            if len(cluster.cues)<minimum or len(item.scene_witnesses)<2:continue
            parents=_independent([roots[i] for i in sorted(item.roots)])
            if parents is None:continue
            sample,medoid=cluster.medoid()
            scenes=sorted(item.scene_witnesses)[:2]
            queries=tuple(dict.fromkeys(tuple(cluster.observations)+tuple(item.scene_witnesses[s] for s in scenes)))
            source_ids=tuple(dict.fromkeys(strict_ids+tuple(i for v in parents for i in v.source_ids)))
            proof=dict(format=1,frozen=True,generation=1,identity=owner,
                query_support=len(cluster.cues),required_support=minimum,scenes=scenes,
                scene_count=len(item.scene_witnesses),query_ids=queries,source_reference_ids=source_ids,
                parents=[dict(medoid=v.medoid,authority=v.authority,support_ids=v.source_ids) for v in parents],
                threshold=threshold,cohesion_threshold=grouping['view_cohesion_threshold'],
                validation='two_disjoint_rooted_views_and_two_query_scenes')
            usage.add(cluster.examples,kind='Corroborated attachment view')
            output.append(IdentityView(Group(owner,(sample,),True,(medoid,)),cluster.pose,medoid,
                queries,len(cluster.cues),tuple(cluster.examples),'corroborated_view',proof))

    for raw in rows:
        row=dict(raw);row['person']=history.aliases.get(row['person'],row['person'])
        if owner!=row['person']:
            emit();owner,slots,scratch=row['person'],[],0
        if owner not in by_owner:continue
        result=rank([row],bank,config)
        if result is None or result[0]!=owner:continue
        query=vector(row);qualified=[]
        for parent in by_owner[owner]:
            medoid=parent.medoid
            if not pose_matches(row['geometry'],parent.pose):continue
            score=float(np.dot(query,parent.group.samples[0]))
            if score<threshold:continue
            if any(float(np.dot(query,s))<grouping['view_cohesion_threshold'] for s in parent.examples):continue
            # Each root must independently beat all mixed rival identities.
            rival=result[1]['rival_score']
            if rival is not None and score-rival<margin:continue
            from .anchor_views import _competing_anchor
            if _competing_anchor(bank.groups,owner,query,score,margin):continue
            if not co_presence_allows(db,row,parent.group,references,query,config):continue
            qualified.append(medoid)
        if _independent([roots[i] for i in qualified]) is None:continue
        item=next((s for s in slots if float(np.dot(s.pivot,query))>=radius and
                   s.cluster.accepts_pose(row['geometry']) and s.cluster.accepts(query,radius)),None)
        if item is None:
            item=_Cohort(_Cluster(pose_view(row['geometry']),row['id']),set(qualified),query);slots.append(item)
            scratch+=query.nbytes
        item.roots.intersection_update(qualified)
        before=len(item.cluster.examples);item.cluster.add(row,query)
        scratch+=(len(item.cluster.examples)-before)*query.nbytes
        item.scene_witnesses.setdefault(row['scene'],row['id'])
        if usage.bytes+scratch>usage.limit:
            raise ValueError('Corroborated view evidence exceeds clustering.max_reference_bytes.')
    emit()
    return tuple(output)
