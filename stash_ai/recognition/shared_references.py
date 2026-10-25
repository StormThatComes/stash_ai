"""Frozen reference authority shared by anchor queries and auxiliary queries."""
from dataclasses import replace

from .clustering import Group, _groups, matching_policy, co_presence_allows
from .database import encoded
from .decision_resolution import resolve_pending
from .historical_references import History, rank, rank_many
from .pose_policy import anchor_sql, auxiliary_sql
from .reference_roles import strict_sql
from .reference_usage import ReferenceUsage
from .view_prototypes import confirmed_views, ViewPrototype


def retry_strict(database, selected, config):
    """Final frozen strict matching; recovered observations do not train a core."""
    if not selected:
        return
    db=database.connection
    scope=db.execute('SELECT scope,namespace FROM scans WHERE id=?',(selected[0]['scan'],)).fetchone()
    groups=_groups(db,*scope,config.data['clustering'])
    history=History(tuple(groups),(),{},'final-selected-strict')
    policy=matching_policy(config)
    rows=db.execute('SELECT o.id,o.scan,o.cue,o.seconds,o.geometry,o.dimensions,o.embedding,c.id AS candidate '
        "FROM candidates c JOIN observations o ON o.id=json_extract(c.payload,'$.observation') "
        "WHERE c.policy=? AND c.status='withheld' AND json_extract(c.payload,'$.reason') IN "
        "('candidate_group','insufficient_match_or_cohesion','ambiguous_identity','co_present_identity_conflict') AND o.rejected IS NULL AND "+
        anchor_sql('o',config.data['clustering'])+' ORDER BY o.scan,o.cue,o.id',(policy,))
    from .deferred_retries import RetryQueue
    queue=RetryQueue(db,groups,policy)
    for row,result in rank_many(queue.pending(rows),history,config):
        if result is None:
            queue.remember(row)
            continue
        person,proof=result
        group=next(g for g in groups if g.identifier==person)
        from .clustering import vector
        if not co_presence_allows(db,row,group,groups,vector(row),config):
            queue.remember(row)
            continue
        db.execute("UPDATE candidates SET person=?,status='pending',payload=json_set(payload,"
            "'$.reason','late_strict_identity','$.shared_reference',json(?)) WHERE id=?",
            (person,encoded(proof),row['candidate']))
    from .joint_claims import recover as recover_joint_claims
    recover_joint_claims(db,policy,history,config)
    resolve_pending(db,policy,config)
    db.execute("UPDATE candidates SET status='accepted' WHERE policy=? AND status='pending' "
               "AND json_extract(payload,'$.reason')='late_strict_identity'",(policy,))
    db.execute("INSERT OR IGNORE INTO members SELECT json_extract(payload,'$.observation'),person "
               "FROM candidates WHERE policy=? AND status='accepted' "
               "AND json_extract(payload,'$.reason')='late_strict_identity'",(policy,))


def early_views(database, selected, config,*,excluded_views=()):
    """Freeze scene-local confirmations before resolving global hypotheses.

    A scene ID is not a timeline. Cold replay uses the independently frozen
    appearance priority for both core establishment and auxiliary confirmation.
    Already established source cores retain precedence over later fragments.
    Other scenes' provisional
    cores cannot steal that evidence before their identity has been resolved.
    Old automatic ownership participates in label binding, never this partition.
    """
    grouping=config.data['clustering']
    if not grouping['view_prototypes_enabled'] or not grouping['auxiliary_matching'] or not selected:
        return ()
    db=database.connection
    scope=db.execute('SELECT scope,namespace FROM scans WHERE id=?',(selected[0]['scan'],)).fetchone()
    groups=_groups(db,*scope,grouping)
    policy=matching_policy(config)
    minimum=config.data['recognition']['min_identity_support']
    horizons={}
    seen={}
    from .local_cohorts import ordering
    position=ordering(db)
    rows=db.execute('SELECT m.person,s.scene,o.scan,o.quality,o.cue,o.seconds,'+position+' AS appearance_order FROM members m JOIN observations o ON o.id=m.observation '
        'JOIN scans s ON s.id=o.scan WHERE '+strict_sql()+' AND '+anchor_sql('o',grouping)+
        ' ORDER BY CAST(s.scene AS INTEGER),s.scene,appearance_order,o.cue,o.id')
    for row in rows:
        key=(row['person'],row['scene'])
        from .independent_evidence import Votes
        cues=seen.setdefault(key,Votes());cues.add(row)
        if len(cues)>=minimum:
            horizons.setdefault(key,row['appearance_order'])
    db.execute('CREATE TEMP TABLE early_view_support(observation TEXT PRIMARY KEY,person TEXT)')
    rows=db.execute('SELECT o.id,o.scan,o.cue,o.seconds,o.geometry,o.dimensions,o.embedding,s.scene,'+position+' AS appearance_order FROM observations o '
        'JOIN scans s ON s.id=o.scan WHERE o.rejected IS NULL AND '+auxiliary_sql('o',grouping)+
        ' ORDER BY CAST(s.scene AS INTEGER),s.scene,o.cue,o.id')
    for row in rows:
        eligible=tuple(g for g in groups
            if len(seen.get((g.identifier,row['scene']),()))>=minimum)
        earlier=tuple(g for g in eligible if horizons[(g.identifier,row['scene'])]<row['appearance_order'])
        eligible=earlier or eligible
        result=rank([row],History(eligible,(),{},'frozen-scene-local-strict-cores'),config)
        if result is None:
            continue
        person,_=result
        from .clustering import vector
        group=next(g for g in eligible if g.identifier==person)
        if co_presence_allows(db,row,group,eligible,vector(row),config):
            db.execute('INSERT INTO early_view_support VALUES (?,?)',(row['id'],person))
    usage=ReferenceUsage(grouping['max_reference_bytes'])
    for group in groups:
        usage.add(group.samples,people=1,kind='Strict')
    source=db.execute('SELECT e.person,o.id,o.scan,o.seconds,o.geometry,o.dimensions,o.embedding,o.quality '
        'FROM early_view_support e JOIN observations o ON o.id=e.observation JOIN scans s ON s.id=o.scan '
        'ORDER BY e.person,CAST(s.scene AS INTEGER),s.scene,o.cue,o.seconds,o.id')
    promotion=dict(grouping,view_min_support=max(grouping['view_min_support'],grouping['auxiliary_min_support']))
    views=confirmed_views(db,policy,promotion,remaining_bytes=grouping['max_reference_bytes']-usage.bytes,rows=source,complete=True)
    views=tuple(v for v in views if v.medoid not in excluded_views)
    db.execute('DROP TABLE early_view_support')
    # Resolve the observation partition before considering historical labels.
    # A prior automatic label is not independent evidence against a repair.
    history=History((),views,{},'frozen-scene-local-auxiliary-views')
    proposals=[]
    for group in groups:
        rows=db.execute('SELECT id,geometry,dimensions,embedding FROM observations WHERE id IN '
            '(SELECT value FROM json_each(?)) ORDER BY id',(encoded(group.sample_ids),)).fetchall()
        eligible=tuple(v for v in views if v.group.identifier!=group.identifier)
        result=rank(rows,replace(history,views=eligible),config) if rows else None
        if result is None:
            continue
        target,proof=result
        from .global_views import _cooccurs
        from .anchor_views import _competing_anchor
        from .clustering import vector
        rivals=[g for g in groups if g.identifier!=group.identifier]
        if (_cooccurs(db,group.identifier,target,config) or len(rows)<grouping['view_attach_min_support'] or
            any(_competing_anchor(rivals,target,vector(row),proof['mean_score'],proof['ambiguity_margin']) for row in rows)):
            continue
        proposals.append((group.identifier,target,proof))
    sources={p[0] for p in proposals}
    blocked=set()
    for index,(left,target,_) in enumerate(proposals):
        for right,other,_ in proposals[index+1:]:
            if target==other and _cooccurs(db,left,right,config):
                blocked.update((left,right))
    for source,target,proof in proposals:
        if source in blocked or target in sources:
            continue
        db.execute('UPDATE members SET person=? WHERE person=?',(target,source))
        db.execute("UPDATE candidates SET person=?,payload=json_set(payload,'$.prior_reason',"
            "json_extract(payload,'$.reason'),'$.reason','shared_view_identity','$.shared_reference',json(?)) "
            'WHERE person=?',(target,encoded(proof),source))
    return views


def auxiliary_bank(db, config, scope, namespace, history, mapping, early):
    """Alias reference owners, while preserving each independently frozen bank."""
    grouping=config.data['clustering']
    groups=_groups(db,scope,namespace,grouping)
    for person in history.people.values():
        db.execute('INSERT OR IGNORE INTO people VALUES (?,?,?,?,?,?)',person)
    for source,target in list(mapping.items()):
        if target in history.people:
            db.execute('INSERT OR IGNORE INTO people VALUES (?,?,?,?,?,?)',history.people[target])
        else:
            db.execute('INSERT OR IGNORE INTO people SELECT ?,scope,namespace,created,performer,? '
                       'FROM people WHERE id=?',(target,'shared-reference:'+target,source))
        if source!=target:
            db.execute('UPDATE members SET person=? WHERE person=?',(target,source))
            db.execute('UPDATE candidates SET person=? WHERE person=?',(target,source))
        mapping.setdefault(target,target)
    groups=[Group(mapping.get(g.identifier,g.identifier),g.samples,True,g.sample_ids) for g in groups]
    groups.extend(history.groups)
    views=[ViewPrototype(Group(mapping.get(v.group.identifier,v.group.identifier),v.group.samples,True),
        v.pose,v.medoid,v.support,v.support_count,v.authority,getattr(v,'promotion',None)) for v in tuple(v for v in early if v.group.identifier in mapping)+tuple(history.views)]
    usage=ReferenceUsage(grouping['max_reference_bytes'])
    for group in groups:
        usage.add(group.samples,people=1,kind='Shared strict')
    for view in views:
        usage.add(view.group.samples,kind='Shared view')
    if len({g.identifier for g in groups})>grouping['max_people_per_lookup']:
        raise ValueError('Shared identity catalogue exceeds configured person budget.')
    return groups,tuple(views),history.confirmations


def finalize_early(db,config,mapping):
    """A source-confirmed selected query needs its final direct auxiliary owner."""
    import json
    from .reference_roles import observation_sql
    policy=matching_policy(config)
    rows=db.execute("SELECT id,person,payload FROM candidates WHERE policy=? AND status='accepted' "
        "AND json_type(payload,'$.shared_reference.view_evidence')='array'",(policy,)).fetchall()
    for row in rows:
        payload=json.loads(row['payload'])
        views=[v for v in payload['shared_reference']['view_evidence'] if v['authority']=='verified_view']
        if not views:continue
        expected=mapping.get(row['person'],row['person'])
        valid=True
        for view in views:
            ids=view['support_ids']
            witnesses=db.execute('SELECT m.observation,m.person FROM members m JOIN candidates c ON '+
                observation_sql('c')+"=m.observation WHERE m.observation IN (SELECT value FROM json_each(?)) "
                "AND c.policy=? AND c.status='accepted' AND c.person=m.person AND "
                "json_extract(c.payload,'$.reason')='auxiliary_identity'",(encoded(ids),policy)).fetchall()
            if len(witnesses)!=len(ids) or any(mapping.get(w['person'],w['person'])!=expected for w in witnesses):
                valid=False;break
        if valid:continue
        db.execute('DELETE FROM members WHERE observation=?',(payload['observation'],))
        db.execute("UPDATE candidates SET person=NULL,status='withheld',payload=json_set(payload,"
            "'$.prior_reason',json_extract(payload,'$.reason'),'$.reason','shared_view_source_invalidated',"
            "'$.shared_reference.source_validation','invalidated_after_auxiliary') WHERE id=?",(row['id'],))
