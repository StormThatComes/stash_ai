"""One generation of attachment views with two disjoint rooted confirmations.

Promotion is read-only and rebuilt from latest durable memberships. Neither a
corroborated view nor this replay's output can supply a promotion root. Keeping
the full source lineage makes lost ownership/support revoke the next capture.
"""
import math

from .clustering import vector,Group,co_presence_allows
from .database import encoded
from .global_views import IdentityView
from .view_prototypes import coverage_threshold,pose_matches
from .promotion_cohorts import PromotionCohorts


def _independent(roots):
    for i,left in enumerate(roots):
        for right in roots[i+1:]:
            if not set(left.support).intersection(right.support):
                return left,right
    return None


def enabled(config):
    grouping=config.data['clustering']
    return grouping['corroborated_view_enrichment'] and (grouping['anchor_view_matching'] or grouping['view_prototypes_enabled'])


def previous_queries(db,selected,config,scope,namespace):
    """Freeze eligibility only; previous labels never supply positive evidence."""
    if not enabled(config):return ()
    from .historical_references import _source,parameters
    from .reference_compatibility import fallback_sql
    rows=db.execute('SELECT o.id '+_source(config,fallback_sql(db,config,namespace),attachment=True)+
        ' AND s.id IN (SELECT value FROM json_each(?)) ORDER BY o.id',
        parameters([],config,scope,namespace)+(encoded([s['scan'] for s in selected]),))
    return tuple(r['id'] for r in rows)


def previous_strict(db,selected,config,scope,namespace):
    if not enabled(config):return ()
    from .historical_references import _source,parameters
    from .reference_compatibility import fallback_sql
    return tuple(r['id'] for r in db.execute('SELECT o.id '+_source(config,fallback_sql(db,config,namespace))+
        ' AND s.id IN (SELECT value FROM json_each(?)) ORDER BY o.id',
        parameters([],config,scope,namespace)+(encoded([s['scan'] for s in selected]),)))


def previous_hints(db,config,scope,namespace,previous):
    """A previous plan can defer a birth, never authorize an assignment."""
    if not enabled(config):return ()
    import json
    from .config import digest
    row=db.execute('SELECT value FROM metadata WHERE key=?',('identity-rebuild:'+digest([scope,namespace]),)).fetchone()
    try:state=json.loads(row[0]) if row else {}
    except (ValueError,TypeError):state={}
    if not isinstance(state,dict):state={}
    ids=state.get('corroborated_plan')
    if ids is None and previous:
        # Bootstrap an existing library once. Later plans are persisted with the
        # completed rebuild, avoiding another whole-bank capture on warm runs.
        from .historical_references import capture,refresh_verified
        from .reference_usage import ReferenceUsage
        usage=ReferenceUsage(config.data['clustering']['max_reference_bytes'])
        history=capture(db,[],config,scope,namespace,'prior-corroboration-plan',resolve_aliases=False)
        for g in history.groups:usage.add(g.samples,kind='Prior plan strict')
        for v in history.views:usage.add(v.examples,kind='Prior plan views')
        refresh_verified(db,[],config,scope,namespace,history,usage)
        ids=[v.medoid for v in history.views if v.authority=='corroborated_view']
    if not isinstance(ids,(list,tuple)) or any(not isinstance(i,str) for i in ids):return ()
    from .historical_references import _source,parameters
    from .reference_compatibility import fallback_sql
    return tuple(db.execute('SELECT o.id,o.geometry,o.dimensions,o.embedding '+_source(config,fallback_sql(db,config,namespace),attachment=True)+
        ' AND o.id IN (SELECT value FROM json_each(?)) ORDER BY o.id',
        parameters([],config,scope,namespace)+(encoded(ids),)))


def defer_new_groups(db,config,hints,established):
    if not hints:return ()
    from .clustering import _groups,matching_policy
    from .pose_policy import anchor_sql
    from .view_prototypes import pose_view
    scope,namespace=db.execute('SELECT scope,namespace FROM scans LIMIT 1').fetchone()
    grouping,settings=config.data['clustering'],config.data['recognition']
    threshold=max(settings['match_threshold'],grouping['view_match_threshold'])-max(settings['ambiguity_margin'],grouping['view_ambiguity_margin'])
    import numpy as np
    frozen=[(r['id'],vector(r),pose_view(r['geometry'])) for r in hints]
    used=set()
    for group in _groups(db,scope,namespace,grouping):
        if set(group.sample_ids).intersection(established):continue
        rows=db.execute('SELECT o.id,o.geometry,o.dimensions,o.embedding FROM members m JOIN observations o ON o.id=m.observation WHERE m.person=?',
                        (group.identifier,))
        common=None
        for row in rows:
            q=vector(row)
            matches={i for i,v,p in frozen if float(np.dot(q,v))>=threshold and pose_matches(row['geometry'],p)}
            common=matches if common is None else common.intersection(matches)
            if not common:break
        if not common:continue
        used.update(common)
        db.execute("UPDATE candidates SET person=NULL,status='withheld',payload=json_set(payload,'$.prior_reason',"
            "json_extract(payload,'$.reason'),'$.reason','prior_appearance_pending','$.prior_appearance_hints',json(?)) "
            'WHERE policy=? AND person=?',(encoded(sorted(common)),matching_policy(config),group.identifier))
        db.execute('DELETE FROM members WHERE person=?',(group.identifier,))
    return tuple(sorted(used))


def enrich(db,config,scope,namespace,selected,history,usage,*,eligible_ids=None,query_rows=None):
    """Freeze extra query-only views; no membership or reference-role writes."""
    grouping,settings=config.data['clustering'],config.data['recognition']
    if not enabled(config) or not history.groups or not history.views:
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
    rows=(query_rows() if query_rows is not None else db.execute('SELECT m.person,o.*,s.scene '+
        _source(config,fallback_sql(db,config,namespace),attachment=True)+
        ' ORDER BY m.person,o.embedding,o.geometry,o.id',parameters(selected,config,scope,namespace)))
    output,owner,cohorts=[],None,None

    def emit():
        if owner is None:return
        strict_ids=tuple(i for g in bank.groups if g.identifier==owner for i in g.sample_ids)
        for appearance in cohorts.supported():
            parents=tuple(roots[i] for i in appearance.parents)
            sample,medoid=appearance.sample,appearance.medoid
            queries=appearance.queries
            source_ids=tuple(dict.fromkeys(strict_ids+tuple(i for v in parents for i in v.source_ids)))
            proof=dict(format=2,frozen=True,generation=1,identity=owner,
                query_support=appearance.support,required_support=minimum,scenes=appearance.scenes,
                scene_count=appearance.scene_count,query_ids=queries,source_reference_ids=source_ids,
                parents=[dict(medoid=v.medoid,authority=v.authority,support_ids=v.source_ids) for v in parents],
                threshold=threshold,cohesion_threshold=grouping['view_cohesion_threshold'],support_radius=radius,
                validation='two_disjoint_rooted_views_and_two_query_scenes')
            usage.add(appearance.examples,kind='Corroborated attachment view')
            output.append(IdentityView(Group(owner,(sample,),True,(medoid,)),appearance.pose,medoid,
                queries,appearance.support,appearance.examples,'corroborated_view',proof))

    for raw in rows:
        if eligible_ids is not None and raw['id'] not in eligible_ids:continue
        row=dict(raw);row['person']=history.aliases.get(row['person'],row['person'])
        if owner!=row['person']:
            emit();owner=row['person'];cohorts=PromotionCohorts(roots,radius,minimum,usage)
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
            if not co_presence_allows(row.get('_database',db),row,parent.group,references,query,config):continue
            qualified.append(medoid)
        if _independent([roots[i] for i in qualified]) is None:continue
        cohorts.add(row,query,qualified)
    emit()
    return tuple(output)
