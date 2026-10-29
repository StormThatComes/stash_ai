"""Frozen global label reconciliation, separate from Stash scene publication.

Strict cores and strict appearance cohorts nominate into the same complete
component graph. No intermediate union trains another nomination. Reverse
verified-view claims demote the recovered historical core to attachment authority.
Raw observations, original ownership and performer metadata are preserved.
"""
from dataclasses import replace

from .clustering import Group,vector
from .database import encoded
from .global_views import catalogue,consolidation_plan,_cooccurs
from .strict_partition import summary_views


class InvalidatedSourceViews(RuntimeError):
    def __init__(self,medoids):
        self.medoids=frozenset(medoids)
        super().__init__('Source-view ownership is unstable; no rebuilt memberships were committed.')


def _copresence(db,replay=None,config=None):
    origins={}
    if replay is not None:
        for row in replay.execute('SELECT DISTINCT m.person,o.person FROM members m '
            'JOIN sample_keys k ON k.observation=m.observation JOIN origins o ON o.sample_key=k.sample_key'):
            origins.setdefault(row[0],set()).add(row[1])
    def check(left,right):
        if replay is not None and _cooccurs(replay,left,right,config):
            return True
        a,b=origins.get(left,{left}),origins.get(right,{right})
        enabled=bool(config and config.data['clustering'].get('deduplicate_detections'))
        fields='x.geometry,x.embedding,y.geometry,y.embedding' if enabled else '1'
        rows=db.execute('SELECT '+fields+' FROM members a JOIN observations x ON x.id=a.observation '
            'JOIN observations y ON y.scan=x.scan AND y.cue=x.cue '
            'JOIN members b ON b.observation=y.id WHERE a.person IN (SELECT value FROM json_each(?)) '
            'AND b.person IN (SELECT value FROM json_each(?)) AND a.person<>b.person AND x.id<>y.id '
            'AND x.rejected IS NULL AND y.rejected IS NULL'+('' if enabled else ' LIMIT 1'),
            (encoded(sorted(a)),encoded(sorted(b))))
        if not enabled:return rows.fetchone() is not None
        from .physical_faces import distinct_pairs
        return distinct_pairs(rows,True)
    return check


def resolve(db,history,config,usage,*,replay=None,groups=(),early=(),excluded_views=()):
    """Return selected bindings and remap the read-only historical bank in place."""
    if replay is not None and history.groups:
        from .historical_references import rank
        candidates=[]
        for group in groups:
            rows=replay.execute('SELECT id,geometry,dimensions,embedding FROM observations WHERE id IN '
                '(SELECT value FROM json_each(?))',(encoded(group.sample_ids),)).fetchall()
            match=rank(rows,history,config) if rows else None
            # A new query already recognizable only through a view remains an
            # attachment. Its provisional local label cannot upgrade its role
            # by voting in a global merger before historical matching runs.
            if match is None or match[1]['mode']!='view':
                candidates.append(group)
        groups=tuple(candidates)
    all_groups=tuple(history.groups)+tuple(groups)
    views=tuple(v for v in history.views if v.authority=='strict')
    if replay is not None and groups:
        first=replay.execute('SELECT scope,namespace FROM scans LIMIT 1').fetchone()
        views+=catalogue(replay,config,*first,usage=usage,complete=True)
    summaries=summary_views(all_groups,usage)
    copresent=_copresence(db,replay,config)
    decisions=consolidation_plan(db,views+summaries,config,mixed=True,cooccurs=copresent,subcomponents=False)
    historical=set(history.people)
    selected={g.identifier for g in groups}
    aliases={}
    proposals={}
    actions=[]
    active=set()
    for decision in decisions:
        if decision['outcome']!='merge' or not historical.intersection(decision['people']):
            continue
        people=decision['people'];active.update(people)
        target=min(historical.intersection(people),key=lambda p:(history.people[p][3],p))
        proof=dict(format=1,frozen=True,mode='component',identity=target,context=history.context,
            component=decision)
        for source in people:
            if source in historical:
                aliases[source]=target
                if source!=target:
                    actions.append(dict(source=source,target=target,kind='strict_component',proof=proof))
            elif source in selected:
                proposals[source]=(target,proof)
    # An independently confirmed source profile may recognize a historical core
    # appearing before that source scene was ever selected. This is attachment,
    # not a new strict merger nomination; its old core cannot train afterwards.
    reverse_views=tuple(v for v in views if v.group.identifier in selected)+tuple(early)
    reverse_views=tuple(v for v in reverse_views if v.medoid not in excluded_views)
    if replay is not None and reverse_views:
        from .historical_references import History,rank
        from .anchor_views import _competing_anchor
        claims=[]
        for group in history.groups:
            if group.identifier in active:
                continue
            rows=db.execute('SELECT id,geometry,dimensions,embedding FROM observations WHERE id IN '
                '(SELECT value FROM json_each(?))',(encoded(group.sample_ids),)).fetchall()
            candidates=tuple(v for v in reverse_views if v.group.identifier in selected and v.group.identifier not in active)
            result=rank(rows,History((),candidates,{},'frozen-source-confirmed-views'),config) if rows else None
            if result is None:
                continue
            target,proof=result
            support=db.execute('SELECT COUNT(*) FROM (SELECT o.scan,o.seconds FROM members m '
                'JOIN observations o ON o.id=m.observation WHERE m.person=? GROUP BY o.scan,o.seconds)',
                (group.identifier,)).fetchone()[0]
            rivals=[g for g in all_groups if g.identifier!=group.identifier]
            if (support<config.data['clustering']['view_attach_min_support'] or copresent(group.identifier,target)
                or any(_competing_anchor(rivals,target,vector(row),proof['mean_score'],proof['ambiguity_margin']) for row in rows)):
                continue
            claims.append((group.identifier,target,proof))
        blocked=set()
        for i,(left,target,_) in enumerate(claims):
            for right,other,_ in claims[i+1:]:
                if target==other and copresent(left,right):
                    blocked.update((left,right))
        by_target={}
        for source,target,proof in claims:
            if source not in blocked:
                by_target.setdefault(target,[]).append((source,proof))
        for source_core,claims in by_target.items():
            target=min((s for s,_ in claims),key=lambda p:(history.people[p][3],p))
            proof=dict(claims[0][1],identity=target,mode='reference_owner',source_core=source_core)
            proposals[source_core]=(target,proof)
            for source,claim in claims:
                aliases[source]=target
                actions.append(dict(source=source,target=target,kind='verified_attachment',
                    proof=dict(claim,identity=target,source_core=source_core,
                        source_view_medoids=[v.medoid for v in reverse_views if v.group.identifier==source_core])))
    demoted={a['source'] for a in actions if a['kind']=='verified_attachment'}
    def remap(group):
        return Group(aliases.get(group.identifier,group.identifier),group.samples,True,group.sample_ids)
    history.groups=tuple(remap(g) for g in history.groups if g.identifier not in demoted)
    history.views=tuple(replace(v,group=remap(v.group)) for v in history.views if v.group.identifier not in demoted)
    history.confirmations=tuple(replace(v,group=remap(v.group)) for v in history.confirmations if v.group.identifier not in demoted)
    history.aliases=aliases
    history.actions=tuple(actions)
    return proposals


def invalidated_sources(history,replay,config,mapping):
    """Historical demotions need the final owners of their selected witnesses."""
    from .clustering import matching_policy
    from .reference_roles import observation_sql
    policy=matching_policy(config);invalid=set()
    for action in history.actions:
        if action['kind']!='verified_attachment':continue
        expected=mapping.get(action['target'],action['target'])
        for view in action['proof'].get('view_evidence',()):
            ids=view['support_ids']
            members=replay.execute('SELECT observation,person FROM members WHERE observation IN '
                '(SELECT value FROM json_each(?))',(encoded(ids),)).fetchall()
            valid=len(members)==len(ids) and all(mapping.get(r['person'],r['person'])==expected for r in members)
            if valid and view['authority']=='verified_view':
                count=replay.execute('SELECT COUNT(*) FROM candidates c WHERE '+observation_sql('c')+
                    " IN (SELECT value FROM json_each(?)) AND c.policy=? AND c.status='accepted' "
                    "AND json_extract(c.payload,'$.reason')='auxiliary_identity'",(encoded(ids),policy)).fetchone()[0]
                valid=count==len(ids)
            if not valid:
                invalid.update(action['proof'].get('source_view_medoids',()) or (view['medoid'],))
    return invalid


def apply(database,history,config):
    """Commit derived global aliases/roles in the selected rebuild transaction."""
    from .reference_roles import STRICT_REASONS
    from .sample_keys import sample_key
    from .reference_roles import observation_sql
    from .clustering import matching_policy
    import uuid
    db=database.connection
    dirty=set()
    for action in history.actions:
        source,target=action['source'],action['target']
        # Preserve original ownership before changing derived memberships.
        rows=db.execute('SELECT o.id,o.scan,o.cue,o.seconds,o.source,o.rect,o.geometry,'
            's.scope,s.namespace,s.scene,s.input_sha FROM members m '
            'JOIN observations o ON o.id=m.observation JOIN scans s ON s.id=o.scan WHERE m.person=?',(source,))
        for row in rows:
            dirty.add((row['scope'],row['namespace'],row['scene']))
            db.execute('INSERT OR IGNORE INTO identity_origins VALUES (?,?,?)',(sample_key(row),source,row['id']))
            if action['kind']=='verified_attachment' and not db.execute('SELECT 1 FROM candidates c WHERE '+
                observation_sql('c')+'=? LIMIT 1',(row['id'],)).fetchone():
                # Membership-only legacy evidence has implicit strict authority.
                # Materialize its demotion even when no valid old decision exists.
                policy=matching_policy(config)
                identifier=str(uuid.uuid5(uuid.UUID(row['id']),policy))
                payload=dict(observation=row['id'],reason='global_reference_attachment',prior_reason='membership_only',
                    reference_registry=action,reference_role='attachment',reference_selected=False,policy=policy)
                db.execute('INSERT OR REPLACE INTO candidates VALUES (?,?,?,?,?,?)',
                    (identifier,row['scan'],policy,source,'accepted',encoded(payload)))
        db.execute('UPDATE members SET person=? WHERE person=?',(target,source))
        demote=action['kind']=='verified_attachment'
        db.execute("UPDATE candidates SET person=?,payload=json_set(payload,'$.reference_registry',json(?),'$.publication_policy',?,"
            "'$.strict_confirmation',CASE WHEN json_type(payload,'$.strict_confirmation')='object' THEN "
            "json_set(json_extract(payload,'$.strict_confirmation'),'$.identity',?) ELSE NULL END,"
            "'$.prior_reason',CASE WHEN json_extract(payload,'$.reason')='global_reference_attachment' THEN "
            "coalesce(json_extract(payload,'$.prior_reason'),'membership_only') ELSE json_extract(payload,'$.reason') END,"
            "'$.reason',CASE WHEN ? AND "
            "json_extract(payload,'$.reason') IN (SELECT value FROM json_each(?)) THEN 'global_reference_attachment' "
            "ELSE json_extract(payload,'$.reason') END,'$.reference_role',CASE WHEN ? THEN 'attachment' "
            "ELSE coalesce(json_extract(payload,'$.reference_role'),'strict') END,'$.reference_selected',"
            "json(CASE WHEN ? THEN 'false' WHEN json_extract(payload,'$.reference_selected') THEN 'true' ELSE 'false' END)) "
            "WHERE person=? AND status='accepted' AND json_valid(payload)",
            (target,encoded(action),matching_policy(config),target,demote,encoded(STRICT_REASONS),demote,demote,source))
    from .publication_queue import enqueue
    for scope,namespace,scene in sorted(dirty):
        enqueue(db,scope,namespace,scene,config,'global_reference_repair')
    return bool(history.actions)
