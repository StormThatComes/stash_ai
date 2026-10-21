"""Frozen reference authority shared by anchor queries and auxiliary queries."""
from dataclasses import replace

from .clustering import Group, _groups, matching_policy, co_presence_allows
from .database import encoded
from .decision_resolution import resolve_pending
from .historical_references import History, rank
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
        "('candidate_group','insufficient_match_or_cohesion') AND o.rejected IS NULL AND "+
        anchor_sql('o',config.data['clustering'])+' ORDER BY o.scan,o.cue,o.id',(policy,))
    for row in rows:
        result=rank([row],history,config)
        if result is None:
            continue
        person,proof=result
        group=next(g for g in groups if g.identifier==person)
        from .clustering import vector
        if not co_presence_allows(db,row,group,groups,vector(row),config):
            continue
        db.execute("UPDATE candidates SET person=?,status='pending',payload=json_set(payload,"
            "'$.reason','late_strict_identity','$.shared_reference',json(?)) WHERE id=?",
            (person,encoded(proof),row['candidate']))
    resolve_pending(db,policy,config)
    db.execute("UPDATE candidates SET status='accepted' WHERE policy=? AND status='pending' "
               "AND json_extract(payload,'$.reason')='late_strict_identity'",(policy,))
    db.execute("INSERT OR IGNORE INTO members SELECT json_extract(payload,'$.observation'),person "
               "FROM candidates WHERE policy=? AND status='accepted' "
               "AND json_extract(payload,'$.reason')='late_strict_identity'",(policy,))


def early_views(database, selected, config):
    """Corroborate views only against cores established before each source cue.

    Later provisional anchor clusters cannot steal the earlier auxiliary evidence
    needed to recognize them. No recovered membership trains this frozen bank.
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
    rows=db.execute('SELECT m.person,s.scene,o.cue,o.seconds FROM members m JOIN observations o ON o.id=m.observation '
        'JOIN scans s ON s.id=o.scan WHERE '+strict_sql()+' AND '+anchor_sql('o',grouping)+
        ' ORDER BY CAST(s.scene AS INTEGER),s.scene,o.cue,o.id')
    for row in rows:
        person=row['person']
        if person in horizons:
            continue
        cues=seen.setdefault(person,set());cues.add((row['scene'],row['seconds']))
        if len(cues)>=minimum:
            horizons[person]=(int(row['scene']),row['cue'])
    db.execute('CREATE TEMP TABLE early_view_support(observation TEXT PRIMARY KEY,person TEXT)')
    rows=db.execute('SELECT o.id,o.scan,o.cue,o.seconds,o.geometry,o.dimensions,o.embedding,s.scene FROM observations o '
        'JOIN scans s ON s.id=o.scan WHERE o.rejected IS NULL AND '+auxiliary_sql('o',grouping)+
        ' ORDER BY CAST(s.scene AS INTEGER),s.scene,o.cue,o.id')
    for row in rows:
        key=(int(row['scene']),row['cue'])
        eligible=tuple(g for g in groups if g.identifier in horizons and horizons[g.identifier]<key)
        result=rank([row],History(eligible,(),{},'earlier-strict-cores'),config)
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
    source=db.execute('SELECT e.person,o.id,o.scan,o.seconds,o.geometry,o.dimensions,o.embedding '
        'FROM early_view_support e JOIN observations o ON o.id=e.observation JOIN scans s ON s.id=o.scan '
        'ORDER BY e.person,CAST(s.scene AS INTEGER),s.scene,o.cue,o.seconds,o.id')
    promotion=dict(grouping,view_min_support=max(grouping['view_min_support'],grouping['auxiliary_min_support']))
    views=confirmed_views(db,policy,promotion,remaining_bytes=grouping['max_reference_bytes']-usage.bytes,rows=source)
    db.execute('DROP TABLE early_view_support')
    # Reconcile only new local hypotheses, never existing owned global people.
    history=History((),views,{},'frozen-earlier-auxiliary-views')
    proposals=[]
    owners={}
    for row in db.execute('SELECT DISTINCT m.person,o.person FROM members m JOIN sample_keys k ON k.observation=m.observation '
        'JOIN origins o ON o.sample_key=k.sample_key JOIN old_people p ON p.id=o.person WHERE p.retired=0'):
        owners.setdefault(row[0],set()).add(row[1])
    for group in groups:
        rows=db.execute('SELECT id,geometry,dimensions,embedding FROM observations WHERE id IN '
            '(SELECT value FROM json_each(?)) ORDER BY id',(encoded(group.sample_ids),)).fetchall()
        eligible=tuple(v for v in views if v.group.identifier!=group.identifier
            and horizons.get(v.group.identifier,(float('inf'),0))<horizons.get(group.identifier,(0,0)))
        result=rank(rows,replace(history,views=eligible),config) if rows else None
        if result is None:
            continue
        target,proof=result
        if owners.get(group.identifier) and not owners[group.identifier].intersection(owners.get(target,set())):
            continue
        from .global_views import _cooccurs
        from .anchor_views import _competing_anchor
        from .clustering import vector
        rivals=[g for g in groups if g.identifier!=group.identifier]
        if (_cooccurs(db,group.identifier,target) or len(rows)<grouping['view_attach_min_support'] or
            any(_competing_anchor(rivals,target,vector(row),proof['mean_score'],proof['ambiguity_margin']) for row in rows)):
            continue
        proposals.append((group.identifier,target,proof))
    sources={p[0] for p in proposals}
    blocked=set()
    for index,(left,target,_) in enumerate(proposals):
        for right,other,_ in proposals[index+1:]:
            if target==other and _cooccurs(db,left,right):
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
        v.pose,v.medoid,v.support,v.support_count) for v in tuple(v for v in early if v.group.identifier in mapping)+tuple(history.views)]
    usage=ReferenceUsage(grouping['max_reference_bytes'])
    for group in groups:
        usage.add(group.samples,people=1,kind='Shared strict')
    for view in views:
        usage.add(view.group.samples,kind='Shared view')
    if len({g.identifier for g in groups})>grouping['max_people_per_lookup']:
        raise ValueError('Shared identity catalogue exceeds configured person budget.')
    return groups,tuple(views)
