"""Resolve pending queries with views frozen before this run's pool promotion."""
import itertools

from .clustering import Group,vector,co_presence_allows,matching_policy
from .database import encoded
from .global_views import catalogue,rank_person,unambiguous
from .view_prototypes import pose_matches
from .reference_roles import observation_sql
from .decision_resolution import conflicting_claims
from .scoring import comparisons


def assigned_peer_allows(db,row,target,aliases,config):
    """Known membership is stronger co-presence evidence than one prototype."""
    import numpy as np
    owners=[target]+[s for s,t in aliases.items() if t==target]
    peers=db.execute('SELECT o.id,o.geometry,o.dimensions,o.embedding FROM observations o '
        'JOIN members m ON m.observation=o.id WHERE o.scan=? AND o.cue=? AND o.id<>? '
        'AND o.rejected IS NULL AND m.person IN (SELECT value FROM json_each(?))',
        (row['scan'],row['cue'],row['id'],encoded(owners)))
    threshold=max(config.data['recognition']['match_threshold'],config.data['recognition']['cooccurrence_merge_threshold'])
    query=vector(row)
    return all(peer['geometry']==row['geometry'] and peer['embedding']==row['embedding'] or
               float(np.dot(query,vector(peer)))>=threshold for peer in peers)


def resolve(database,replay,selected,config,history,bindings,confirmed,early,usage):
    from .unresolved_pool import rows as external_rows,query_reasons_sql
    from .anchor_views import _competing_anchor
    scope,namespace=replay.execute('SELECT scope,namespace FROM scans LIMIT 1').fetchone()
    grouping,settings=config.data['clustering'],config.data['recognition']
    owners={g.identifier:g for g in confirmed}
    if not owners or not (grouping['anchor_view_matching'] or grouping['view_prototypes_enabled']):
        return
    strict=(catalogue(replay,config,scope,namespace,usage=usage,complete=True)
            if grouping['anchor_view_matching'] else ())
    views=tuple(strict)+tuple(v for v in early if v.group.identifier in owners)
    if not views:
        return
    aliases=dict(history.aliases,**bindings)
    canonical=lambda p:bindings.get(p,p)
    references=tuple(history.groups)+tuple(history.unavailable)+tuple(Group(canonical(g.identifier),g.samples,True) for g in confirmed)
    by_group={id(v.group):v for v in views}
    view_groups=[v.group for v in views]
    threshold=max(settings['match_threshold'],grouping['view_match_threshold'])
    margin=max(settings['ambiguity_margin'],grouping['view_ambiguity_margin'])
    current=replay.execute('SELECT o.id,o.scan,o.cue,o.seconds,o.geometry,o.dimensions,o.embedding,1 AS current '
        'FROM candidates c JOIN observations o ON o.id='+observation_sql('c')+
        " WHERE c.policy=? AND c.status='withheld' AND json_extract(c.payload,'$.reason') IN ("+query_reasons_sql()+') '
        'AND o.rejected IS NULL',(matching_policy(config),))
    replay.execute('CREATE TEMP TABLE pool_view_claims(observation TEXT PRIMARY KEY,owner TEXT,canonical TEXT,medoid TEXT,scan TEXT,cue INTEGER,seconds REAL,current INTEGER)')
    for row in itertools.chain(current,external_rows(database.connection,selected,config,scope,namespace)):
        query=vector(row);ranked=[]
        for group,score,low in comparisons(view_groups,query,threshold-margin,grouping['view_cohesion_threshold']):
            view=by_group[id(group)]
            if low>=grouping['view_cohesion_threshold'] and pose_matches(row['geometry'],view.pose):
                rank_person(ranked,(score,canonical(group.identifier),view))
        if not unambiguous(ranked,threshold,margin):
            continue
        score,target,view=ranked[0]
        if (_competing_anchor(references,target,query,score,margin) or
            not assigned_peer_allows(database.connection,row,target,aliases,config) or
            not assigned_peer_allows(replay,row,target,aliases,config)):
            continue
        reference=Group(target,view.group.samples,True)
        if not co_presence_allows(database.connection,row,reference,references,query,config,
            peer_threshold=threshold,cohesion_threshold=grouping['view_cohesion_threshold']):
            continue
        is_current=row['current'] if 'current' in row.keys() else 0
        replay.execute('INSERT INTO pool_view_claims VALUES (?,?,?,?,?,?,?,?)',
            (row['id'],view.group.identifier,target,view.medoid,row['scan'],row['cue'],row['seconds'],is_current))
    # One physical cue cannot claim two non-reflection faces of the same owner,
    # including when their matching prototypes differ.
    def claims():
        for claim in replay.execute('SELECT * FROM pool_view_claims ORDER BY canonical,scan,cue,observation'):
            row=dict(database.connection.execute('SELECT id,scan,cue,geometry,dimensions,embedding '
                'FROM observations WHERE id=?',(claim['observation'],)).fetchone())
            row['person']=claim['canonical'];yield row
    stronger=max(settings['match_threshold'],settings['cooccurrence_merge_threshold'])
    for row in conflicting_claims(claims(),stronger):
        replay.execute('DELETE FROM pool_view_claims WHERE observation=?',(row['id'],))
    from .query_support import cohorts
    from .config import digest
    claims_by_id={r['observation']:r for r in replay.execute('SELECT * FROM pool_view_claims')}
    raw=lambda:database.connection.execute('SELECT o.id,o.scan,o.seconds,o.dimensions,o.embedding FROM observations o WHERE id IN (SELECT value FROM json_each(?)) '
        'ORDER BY embedding,geometry,id',(encoded(list(claims_by_id)),))
    eligible=cohorts(raw,config,key=lambda r:claims_by_id[r['id']]['owner'],
        prototype=lambda r:claims_by_id[r['id']]['medoid'])
    known={(v.group.identifier,v.medoid):v for v in views}
    for cohort in eligible:
        owner=cohort.family
        selected_views=[known[owner,medoid] for medoid in cohort.prototypes]
        view=selected_views[0]
        source='view:'+owner+':'+digest(cohort.observations)
        source_ids=tuple(dict.fromkeys(tuple(owners[owner].sample_ids)+tuple(i for v in selected_views for i in v.support)))
        proof=dict(format=2,frozen=True,mode='known_view',independent_support=cohort.independent_support,
            required_support=grouping['view_attach_min_support'],identity=canonical(owner),
            source_owner=owner,owner_reference_ids=owners[owner].sample_ids,
            source_reference_ids=source_ids,view=dict(medoid=view.medoid,authority=view.authority,
                support=view.support_count,support_ids=view.support),context=history.context,
            views=[dict(medoid=v.medoid,authority=v.authority,support=v.support_count,support_ids=v.support) for v in selected_views],
            claim_id=source,query_ids=cohort.observations,query_support_ids=cohort.witnesses,
            threshold=threshold,cohesion_threshold=grouping['view_cohesion_threshold'])
        for row in replay.execute('SELECT * FROM pool_view_claims WHERE observation IN '
                '(SELECT value FROM json_each(?))',(encoded(cohort.observations),)):
            if row['current']:
                replay.execute("UPDATE candidates SET person=?,status='accepted',payload=json_set(payload,"
                    "'$.reason','pooled_view_identity','$.provisional_pool',json(?)) WHERE policy=? AND "+
                    observation_sql('candidates')+'=?',(owner,encoded(proof),matching_policy(config),row['observation']))
                replay.execute('INSERT OR IGNORE INTO members VALUES (?,?)',(row['observation'],owner))
            else:
                replay.execute('INSERT OR IGNORE INTO pool_claims VALUES (?,?,?,?,?)',
                    (row['observation'],source,row['scan'],row['cue'],row['seconds']))
        replay.execute('INSERT INTO pool_promotions VALUES (?,?,?,?)',(source,'known_view',encoded(source_ids),encoded(proof)))


def finalize(replay,config,mapping):
    """A final auxiliary reassignment can revoke an early confirmation's owner."""
    if not replay.execute("SELECT 1 FROM sqlite_temp_master WHERE name='pool_promotions'").fetchone():
        return
    promotions=replay.execute("SELECT * FROM pool_promotions WHERE mode='known_view'").fetchall()
    for promotion in promotions:
        import json
        proof=json.loads(promotion['proof'])
        if 'view' not in proof:
            continue
        expected=json.loads(promotion['selected_ids'])
        memberships=replay.execute('SELECT observation,person FROM members WHERE observation IN '
            '(SELECT value FROM json_each(?))',(encoded(expected),)).fetchall()
        if len(memberships)==len(expected) and len({mapping.get(r['person'],r['person']) for r in memberships})==1:
            continue
        current=replay.execute('SELECT observation FROM pool_view_claims WHERE current=1 AND observation IN '
            '(SELECT value FROM json_each(?))',(encoded(proof['query_ids']),)).fetchall()
        for observation, in current:
            replay.execute("UPDATE candidates SET person=NULL,status='withheld',payload=json_set(payload,"
                "'$.prior_reason',json_extract(payload,'$.reason'),'$.reason','provisional_pool_source_invalidated',"
                "'$.provisional_pool.source_validation','invalidated_after_auxiliary') WHERE policy=? AND "+
                observation_sql('candidates')+"=? AND json_extract(payload,'$.provisional_pool.claim_id')=?",
                (matching_policy(config),observation,proof['claim_id']))
            replay.execute('DELETE FROM members WHERE observation=?',(observation,))
        replay.execute('DELETE FROM pool_promotions WHERE source=?',(promotion['source'],))
        replay.execute('DELETE FROM pool_claims WHERE source=?',(promotion['source'],))
