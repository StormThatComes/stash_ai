"""Revalidate prior accepted queries after selected source ownership is final.

The previous decision freezes eligibility only. Fresh rooted views determine
ownership again, so old automatic labels never train or override the partition.
Newly accepted queries wait for the next replay, including full-library runs.
"""
from dataclasses import replace

from .clustering import matching_policy,vector,co_presence_allows
from .database import encoded
from .corroborated_views import enrich,enabled
from .historical_references import History,capture,rank
from .reference_usage import ReferenceUsage
from .pose_policy import visibility_allows
from .reference_roles import observation_sql


def resolve(durable,database,selected,config,external,previous):
    if not enabled(config):return False
    db=database.connection
    source=durable.connection
    from .reference_roles import attachment_sql
    if not previous and not durable._rebuild_corroborated_plan and not db.execute('SELECT 1 FROM members m JOIN observations o ON o.id=m.observation WHERE '+attachment_sql()+' LIMIT 1').fetchone():
        return False
    scope,namespace=db.execute('SELECT scope,namespace FROM scans LIMIT 1').fetchone()
    grouping=config.data['clustering'];policy=matching_policy(config)
    usage=ReferenceUsage(grouping['max_reference_bytes'])
    selected_bank=capture(db,[],config,scope,namespace,'final-selected-roots',resolve_aliases=False)
    from .verified_views import historical_views
    verified=(historical_views(db,config,scope,namespace,[],selected_bank.groups,usage,selected_bank.confirmations)
              if grouping['view_prototypes_enabled'] and grouping['auxiliary_matching'] else ())
    groups=selected_bank.groups+external.groups
    roots=tuple(v for v in selected_bank.views+verified+external.views if v.authority in ('strict','verified_view'))
    people={r['id']:tuple(r) for r in db.execute('SELECT * FROM people')}
    bank=History(groups,roots,people,'frozen-prior-selected-queries',aliases=external.aliases)
    for g in groups:usage.add(g.samples,kind='Final corroboration strict')
    for v in roots:usage.add(v.examples,kind='Final corroboration roots')
    from .historical_references import _source,parameters
    from .reference_compatibility import fallback_sql
    from heapq import merge
    foreign_source=_source(config,fallback_sql(source,config,namespace),attachment=True)
    foreign_args=parameters(selected,config,scope,namespace)
    prior=set(previous)
    prior.update(r['id'] for r in source.execute('SELECT o.id '+foreign_source,foreign_args))
    def query_rows():
        def stream(connection,selection):
            args=parameters(selection,config,scope,namespace)
            rows=connection.execute('SELECT m.person,o.*,s.scene '+
                _source(config,fallback_sql(connection,config,namespace),attachment=True)+
                ' ORDER BY m.person,o.embedding,o.geometry,o.id',args)
            for row in rows:yield dict(row,_database=connection)
        yield from merge(stream(db,[]),stream(source,selected),
                         key=lambda r:(r['person'],r['embedding'],r['geometry'],r['id']))
    allowed=enrich(db,config,scope,namespace,[],bank,usage,eligible_ids=prior,query_rows=query_rows) if prior else ()
    # A planned but not yet eligible generation must not be hidden by a warm
    # completed-result cache on the next explicit rebuild.
    pending_usage=ReferenceUsage(grouping['max_reference_bytes'])
    for g in groups:pending_usage.add(g.samples,kind='Corroboration plan strict')
    for v in roots:pending_usage.add(v.examples,kind='Corroboration plan roots')
    planned=enrich(db,config,scope,namespace,[],bank,pending_usage,query_rows=query_rows)
    centres={(v.group.identifier,v.medoid) for v in allowed}
    pending=any((v.group.identifier,v.medoid) not in centres for v in planned)
    durable._rebuild_corroborated_plan={v.medoid for v in planned}
    durable._rebuild_revalidated_medoids={v.medoid for v in allowed}
    # Ordinary frozen rivals remain visible; promoted output cannot nominate
    # another view or merger in this stage.
    frozen=replace(bank,views=roots+tuple(v for v in external.views if v.authority=='corroborated_view')+allowed)
    if allowed:
        from .cohort_resolution import recover
        recover(db,list(selected_bank.groups),frozen,config,{})
    references=groups+tuple(v.group for v in frozen.views)
    from .pooled_views import assigned_peer_allows
    db.execute('CREATE TEMP TABLE prior_reference_claims(observation TEXT PRIMARY KEY,candidate TEXT,owner TEXT,proof TEXT)')
    for row in db.execute("SELECT o.*,c.id AS candidate,json_extract(c.payload,'$.reason') AS reason FROM candidates c JOIN observations o ON o.id="+observation_sql('c')+
            " WHERE c.policy=? AND c.status='withheld' AND o.rejected IS NULL ORDER BY o.embedding,o.geometry,o.id",(policy,)):
        if not visibility_allows(row['quality'],grouping.get('min_auxiliary_visible_face_fraction')):continue
        match=rank([row],frozen,config)
        if match is None:continue
        owner,proof=match
        if row['reason'] not in ('prior_attachment_query','prior_appearance_pending') and not any(v['authority']=='corroborated_view' for v in proof['view_evidence']):continue
        if not assigned_peer_allows(db,row,owner,{},config):continue
        if not any(v.group.identifier==owner and co_presence_allows(db,row,v.group,references,vector(row),config)
                   for v in frozen.views) and not any(g.identifier==owner and co_presence_allows(db,row,g,references,vector(row),config) for g in groups):continue
        db.execute('INSERT INTO prior_reference_claims VALUES (?,?,?,?)',(row['id'],row['candidate'],owner,encoded(proof)))
    for row in db.execute('SELECT * FROM prior_reference_claims'):
        db.execute("UPDATE candidates SET person=?,status='pending',payload=json_set(payload,'$.reason',"
            "'prior_corroborated_view_pending','$.historical_reference',json(?)) WHERE id=?",
            (row['owner'],row['proof'],row['candidate']))
    from .decision_resolution import resolve_pending
    resolve_pending(db,policy,config)
    from .query_support import eligible
    rows=lambda:db.execute('SELECT o.id,o.scan,o.seconds,o.dimensions,o.embedding,c.person,json_extract(c.payload,\'$.historical_reference.medoids\') AS medoids,json_extract(c.payload,\'$.historical_reference.mode\') AS mode '
        'FROM candidates c JOIN observations o ON o.id='+observation_sql('c')+
        " WHERE c.policy=? AND c.status='pending' AND json_extract(c.payload,'$.reason')='prior_corroborated_view_pending' "
        'ORDER BY o.embedding,o.geometry,o.id',(policy,))
    supported=eligible(rows,config,key=lambda r:(r['person'],r['mode']),prototype=lambda r:r['medoids'],minimum=lambda f:1 if f[1]=='strict' else grouping['view_attach_min_support'])
    db.execute("UPDATE candidates SET status='accepted',payload=json_set(payload,'$.reason','prior_reference_identity') "
        "WHERE policy=? AND status='pending' AND json_extract(payload,'$.reason')='prior_corroborated_view_pending' "
        'AND '+observation_sql('candidates')+' IN (SELECT value FROM json_each(?))',(policy,encoded(sorted(supported))))
    db.execute("UPDATE candidates SET status='withheld',person=NULL,payload=json_set(payload,'$.reason','view_insufficient_support') "
        "WHERE policy=? AND status='pending' AND json_extract(payload,'$.reason')='prior_corroborated_view_pending'",(policy,))
    db.execute('INSERT OR IGNORE INTO members SELECT '+observation_sql('c')+',c.person FROM candidates c '
        "WHERE c.policy=? AND c.status='accepted' AND json_extract(c.payload,'$.reason')='prior_reference_identity'",(policy,))
    db.execute('DROP TABLE prior_reference_claims')
    return pending
