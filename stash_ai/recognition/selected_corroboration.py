"""Freeze rooted appearances, then finalize queries inside one private replay.

Stage-one accepted attachments receive fresh two-root confirmations before the
bank freezes. Stage-two assignments cannot train or nominate another promotion.
Previous labels freeze query eligibility only and never authorize ownership.
"""
from dataclasses import replace

from .clustering import matching_policy,vector,co_presence_allows
from .database import encoded
from .corroborated_views import enrich,enabled
from .historical_references import History,capture,rank
from .reference_usage import ReferenceUsage
from .pose_policy import visibility_allows
from .reference_roles import observation_sql
from .progress import work, tracking
from .frozen_ownership import FrozenOwnership


def resolve(durable,database,selected,config,external,previous,*,hints=()):
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
    bank=History(groups,roots,people,'frozen-final-selected-queries',aliases=external.aliases)
    for g in groups:usage.add(g.samples,kind='Final corroboration strict')
    for v in roots:usage.add(v.examples,kind='Final corroboration roots')
    # Prior query/birth hints withheld their observations from strict training.
    # Revalidate those observations against roots before freezing this run's
    # one promoted generation, so a new selected scene can corroborate a
    # previously accepted foreign query scene without seeding a rival core.
    ownership=FrozenOwnership(db,source,external.aliases)
    with work('Revalidate rooted identity queries',1,unit='bank',position=0) as stage:
        _resolve_queries(db,config,bank,ownership,prior_only=True)
        stage.advance()
    from .historical_references import _source,parameters
    from .reference_compatibility import fallback_sql
    from heapq import merge
    def query_rows():
        def stream(connection,selection):
            args=parameters(selection,config,scope,namespace)
            rows=connection.execute('SELECT m.person,o.id,o.scan,o.cue,o.seconds,o.geometry,o.quality,o.dimensions,o.embedding,s.scene '+
                _source(config,fallback_sql(connection,config,namespace),attachment=True)+
                ' ORDER BY m.person,o.embedding,o.geometry,o.id',args)
            for row in rows:yield dict(row,_database=connection)
        yield from merge(stream(db,[]),stream(source,selected),
                         key=lambda r:(r['person'],r['embedding'],r['geometry'],r['id']))
    # Enrichment reads only stage-one memberships and original rooted views.
    # Finish it once before accepting any stage-two query; its output cannot
    # feed the input stream, replace parents, or begin another generation.
    validated_hints={}
    with work('Freeze corroborated appearance bank',1,unit='bank',position=0) as stage:
        allowed=enrich(db,config,scope,namespace,[],bank,usage,query_rows=query_rows,
                       validated_hints=validated_hints,hint_ids={r['id'] for r in hints})
        stage.advance()
    durable._rebuild_corroborated_plan={v.medoid for v in allowed}
    durable._rebuild_revalidated_medoids={v.medoid for v in allowed}|set(validated_hints)
    # Ordinary frozen rivals remain visible; promoted output cannot nominate
    # another view or merger in this stage.
    frozen=replace(bank,views=roots+tuple(v for v in external.views if v.authority=='corroborated_view')+allowed)
    if allowed:
        from .cohort_resolution import recover
        protected={v.group.identifier for v in frozen.views if v.authority=='corroborated_view'}|set(validated_hints.values())
        recover(db,[g for g in selected_bank.groups if g.identifier not in protected],frozen,config,{})
    ownership=FrozenOwnership(db,source,external.aliases)
    with work('Finalize corroborated identity queries',1,unit='bank',position=0) as stage:
        _resolve_queries(db,config,frozen,ownership)
        stage.advance()
    return False


def _resolve_queries(db,config,frozen,ownership,*,prior_only=False):
    policy=matching_policy(config);grouping=config.data['clustering'];groups=frozen.groups
    references=groups+tuple(v.group for v in frozen.views)
    from .pooled_views import assigned_peer_allows
    db.execute('CREATE TEMP TABLE prior_reference_claims(observation TEXT PRIMARY KEY,candidate TEXT,owner TEXT,proof TEXT)')
    condition=(" AND json_extract(c.payload,'$.reason') IN ('prior_attachment_query','prior_appearance_pending')" if prior_only else '')
    rows=db.execute("SELECT o.id,o.scan,o.cue,o.seconds,o.geometry,o.quality,o.dimensions,o.embedding,"
        "c.id AS candidate,json_extract(c.payload,'$.reason') AS reason FROM candidates c JOIN observations o ON o.id="+observation_sql('c')+
        " WHERE c.policy=? AND c.status='withheld' AND o.rejected IS NULL"+condition+" ORDER BY o.embedding,o.geometry,o.id",(policy,))
    with tracking(rows,'Match final frozen appearance bank',unit='observation') as tracked:
        for row in tracked:
            if not visibility_allows(row['quality'],grouping.get('min_auxiliary_visible_face_fraction')):continue
            match=rank([row],frozen,config)
            if match is None:continue
            owner,proof=match
            if not ownership.proof_allows(owner,proof,groups):continue
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
