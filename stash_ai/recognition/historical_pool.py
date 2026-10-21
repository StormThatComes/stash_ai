"""Reassess durable pending queries against the entire frozen external bank.

Reference cues never count as query support and recovered queries never become
references. All backfills wait for the same final durable transaction.
"""
from itertools import chain

from .clustering import Group,vector,co_presence_allows,matching_policy
from .database import encoded
from .historical_references import rank
from .pose_policy import anchor_sql
from .reference_roles import observation_sql
from .decision_resolution import conflicting_claims


def resolve(database,replay,selected,config,history,bindings,confirmed):
    if not history.groups and not history.views:return
    from .unresolved_pool import rows as external_rows,query_reasons_sql
    from .pooled_views import assigned_peer_allows
    from .anchor_views import _competing_anchor
    scope,namespace=replay.execute('SELECT scope,namespace FROM scans LIMIT 1').fetchone()
    policy=matching_policy(config)
    aliases=dict(history.aliases,**bindings)
    peers=tuple(Group(bindings.get(g.identifier,g.identifier),g.samples,True) for g in confirmed)
    references=tuple(history.groups)+tuple(v.group for v in history.views)
    replay.execute('CREATE TEMP TABLE historical_pool_claims(observation TEXT PRIMARY KEY,owner TEXT,mode TEXT,medoids TEXT,'
        'scan TEXT,cue INTEGER,seconds REAL,current INTEGER,proof TEXT)')
    current=replay.execute('SELECT o.*,1 AS current FROM candidates c JOIN observations o ON o.id='+observation_sql('c')+
        " WHERE c.policy=? AND c.status='withheld' AND json_extract(c.payload,'$.reason') IN ("+query_reasons_sql()+') '
        'AND o.rejected IS NULL AND '+anchor_sql('o',config.data['clustering']),(policy,))
    for row in chain(current,external_rows(database.connection,selected,config,scope,namespace)):
        match=rank([row],history,config)
        if match is None:continue
        target,proof=match;query=vector(row)
        if (_competing_anchor(peers,target,query,proof['mean_score'],proof['ambiguity_margin']) or
            not assigned_peer_allows(database.connection,row,target,aliases,config) or
            not assigned_peer_allows(replay,row,target,aliases,config)):
            continue
        target_refs=[g for g in references if g.identifier==target]
        if not any(co_presence_allows(database.connection,row,g,references,query,config) for g in target_refs):continue
        replay.execute('INSERT INTO historical_pool_claims VALUES (?,?,?,?,?,?,?,?,?)',
            (row['id'],target,proof['mode'],encoded(proof['medoids']),row['scan'],row['cue'],row['seconds'],
             row['current'] if 'current' in row.keys() else 0,encoded(proof)))
    def claims():
        for claim in replay.execute('SELECT * FROM historical_pool_claims ORDER BY owner,scan,cue,observation'):
            row=dict(database.connection.execute('SELECT id,scan,cue,geometry,dimensions,embedding FROM observations WHERE id=?',
                (claim['observation'],)).fetchone());row['person']=claim['owner'];yield row
    stronger=max(config.data['recognition']['match_threshold'],config.data['recognition']['cooccurrence_merge_threshold'])
    for row in conflicting_claims(claims(),stronger):
        replay.execute('DELETE FROM historical_pool_claims WHERE observation=?',(row['id'],))
    eligible=replay.execute('SELECT owner,mode,medoids,COUNT(*) FROM (SELECT owner,mode,medoids,scan,seconds '
        'FROM historical_pool_claims GROUP BY owner,mode,medoids,scan,seconds) GROUP BY owner,mode,medoids').fetchall()
    import json
    for target,mode,medoids,support in eligible:
        minimum=config.data['clustering']['view_attach_min_support'] if mode=='view' else 1
        if support<minimum:continue
        medoid_ids=json.loads(medoids)
        witnesses=tuple(dict.fromkeys(i for v in history.views if v.group.identifier==target and v.medoid in medoid_ids for i in v.support))
        if mode=='strict':
            witnesses=tuple(dict.fromkeys(i for g in history.groups if g.identifier==target for i in g.sample_ids))
        if not witnesses:continue
        source='historical:'+target+':'+mode+':'+medoids
        proof=dict(format=1,frozen=True,external_reference=True,identity=target,mode='known_'+mode,
            independent_support=support,required_support=minimum,source_reference_ids=witnesses,
            context=history.context,validation='frozen_historical_bank',medoids=medoid_ids)
        replay.execute('INSERT OR IGNORE INTO people VALUES (?,?,?,?,?,?)',history.people[target])
        bindings[target]=target
        for row in replay.execute('SELECT * FROM historical_pool_claims WHERE owner=? AND mode=? AND medoids=?',(target,mode,medoids)):
            if row['current']:
                reason='pooled_view_identity' if mode=='view' else 'late_strict_identity'
                replay.execute("UPDATE candidates SET person=?,status='accepted',payload=json_set(payload,'$.reason',?,"
                    "'$.provisional_pool',json(?),'$.historical_reference',json(?)) WHERE policy=? AND "+observation_sql('candidates')+'=?',
                    (target,reason,encoded(proof),row['proof'],policy,row['observation']))
                replay.execute('INSERT OR IGNORE INTO members VALUES (?,?)',(row['observation'],target))
            else:
                replay.execute('INSERT OR IGNORE INTO pool_claims VALUES (?,?,?,?,?)',
                    (row['observation'],source,row['scan'],row['cue'],row['seconds']))
        replay.execute('INSERT INTO pool_promotions VALUES (?,?,?,?)',(source,'historical_'+mode,'[]',encoded(proof)))
