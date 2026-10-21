"""Revalidate durable unresolved anchors against a frozen selected cohort.

The existing observations and latest withheld decisions are the persistent pool.
No inferred image, new schema, or earlier recovered attachment supplies support.
Only the final rebuild transaction backfills confirmed derived assignments;
Stash publication remains limited to selected scenes.
"""
import json,uuid

from .clustering import Group,_groups,_similarities,matching_policy,vector,co_presence_allows
from .database import encoded
from .reference_usage import ReferenceUsage
from .reference_selection import select
from .reference_roles import observation_sql
from .progress import tracking

QUERY_REASONS=('candidate_group','reference_bank_unavailable','anchor_view_insufficient_support',
               'global_view_insufficient_support','view_insufficient_support','provisional_pool_source_invalidated',
               'historical_potential_identity','shared_view_source_invalidated')


def query_reasons_sql():
    return ','.join(repr(reason) for reason in QUERY_REASONS)


def rows(db,selected,config,scope,namespace):
    from .pose_policy import anchor_sql
    from .reference_compatibility import fallback_sql
    from .scanning import scan_identity
    return db.execute('SELECT o.id,o.scan,o.cue,o.seconds,o.geometry,o.quality,o.dimensions,o.embedding,s.scene,'
        "json_extract(c.payload,'$.reason') AS reason FROM candidates c "
        'JOIN observations o ON o.id='+observation_sql('c')+' JOIN scans s ON s.id=o.scan '
        "WHERE s.scope=? AND s.namespace=? AND s.status='complete' AND o.rejected IS NULL "
        'AND s.scene NOT IN (SELECT value FROM json_each(?)) AND (s.policy_sha=? OR '+fallback_sql(db,config,namespace)+') '
        'AND NOT EXISTS (SELECT 1 FROM scans n WHERE n.scope=s.scope AND n.namespace=s.namespace '
        'AND n.scene=s.scene AND n.rowid>s.rowid) AND '+anchor_sql('o',config.data['clustering'])+
        " AND c.scan=o.scan AND c.status='withheld' AND CASE WHEN json_valid(c.payload) THEN "
        "json_extract(c.payload,'$.reason') END IN ("+query_reasons_sql()+') '
        'AND NOT EXISTS (SELECT 1 FROM candidates n WHERE '+observation_sql('n')+'=o.id AND n.rowid>c.rowid) '
        'AND NOT EXISTS (SELECT 1 FROM members m WHERE m.observation=o.id) '
        'ORDER BY CAST(s.scene AS INTEGER),s.scene,o.cue,o.id',
        (scope,namespace,encoded([item['scene'] for item in selected]),scan_identity(config,namespace)))


def hash_context(db,selected,config,scope,namespace,hasher):
    if config.data['recognition']['cross_scene_provisional']:
        from .rebuild_cache import _hash_row
        for row in rows(db,selected,config,scope,namespace):
            _hash_row(hasher,row)


class Witnesses:
    """Bounded quality/diversity witnesses; exact support counts live in SQLite."""
    def __init__(self,source_rows,maximum,usage):
        self.rows=list(source_rows)
        self.maximum,self.usage=maximum,usage
        self.size=sum(len(row['embedding']) for row in self.rows)
        usage.add_counts(self.size,len(self.rows),kind='Unresolved witnesses')

    @property
    def samples(self):
        return tuple(vector(row) for row in self.rows)

    def add(self,row):
        self.rows.append(dict(row))
        self.usage.add_counts(len(row['embedding']),1,kind='Unresolved witnesses')
        self.size+=len(row['embedding'])
        if len(self.rows)>=2*self.maximum:
            chosen={identifier for identifier,_ in select(sorted(self.rows,key=lambda r:(int(r['scene']),r['seconds'],r['id'])),self.maximum)}
            retained=[r for r in self.rows if r['id'] in chosen]
            size=sum(len(r['embedding']) for r in retained)
            self.usage.add_counts(size-self.size,len(retained)-len(self.rows),kind='Unresolved witnesses')
            self.rows,self.size=retained,size


def _source_rows(replay,group):
    return replay.execute('SELECT o.id,o.scan,o.cue,o.seconds,o.geometry,o.quality,o.dimensions,o.embedding,s.scene '
        'FROM observations o JOIN scans s ON s.id=o.scan WHERE o.id IN '
        '(SELECT value FROM json_each(?)) ORDER BY s.scene,o.cue,o.id',(encoded(group.sample_ids),)).fetchall()


def _deferred(replay,policy,source,outcome,**evidence):
    proof=dict(format=1,frozen=True,outcome=outcome,**evidence)
    replay.execute("UPDATE candidates SET payload=json_set(payload,'$.provisional_pool',json(?)) "
        "WHERE policy=? AND person=? AND status='withheld'",(encoded(proof),policy,source))


def _validate_known(database,replay,source,current_ids,target,history,config,bindings):
    """Every claimed observation must match the chosen owner, including outliers
    outside the bounded selector and cue peers belonging to the same owner.
    """
    from .historical_references import rank
    from .pooled_views import assigned_peer_allows
    from itertools import chain,islice
    claimed=replay.execute('SELECT observation FROM pool_claims WHERE source=?',(source,))
    selected=set(current_ids)
    identifiers=chain(iter(current_ids),(r[0] for r in claimed))
    references=tuple(history.groups)+tuple(v.group for v in history.views)
    target_refs=[g for g in references if g.identifier==target]
    aliases=dict(history.aliases,**bindings)
    valid=True
    while batch:=list(islice(identifiers,256)):
        observations=database.connection.execute('SELECT id,scan,cue,seconds,geometry,dimensions,embedding '
            'FROM observations WHERE id IN (SELECT value FROM json_each(?))',(encoded(batch),))
        for row in observations:
            match=rank([row],history,config)
            if (match is not None and match[0]==target
                and assigned_peer_allows(database.connection,row,target,aliases,config)
                and assigned_peer_allows(replay,row,target,aliases,config)
                and any(co_presence_allows(database.connection,
                row,ref,references,vector(row),config) for ref in target_refs)):
                continue
            if row['id'] in selected:
                valid=False
            else:
                replay.execute('DELETE FROM pool_claims WHERE observation=?',(row['id'],))
    return valid


def resolve(database,replay,selected,config,history,bindings,early=()):
    if not selected or not config.data['recognition']['cross_scene_provisional']:
        return
    from .historical_references import rank
    from .global_views import rank_person,unambiguous
    from .pose_policy import anchor_sql
    from .scoring import comparisons
    from .anchor_views import _competing_anchor
    scope,namespace=replay.execute('SELECT scope,namespace FROM scans LIMIT 1').fetchone()
    if rows(database.connection,selected,config,scope,namespace).fetchone() is None:
        return
    policy=matching_policy(config)
    settings,grouping=config.data['recognition'],config.data['clustering']
    usage=ReferenceUsage(grouping['max_reference_bytes'])
    confirmed=_groups(replay,scope,namespace,grouping,usage=usage)
    pending=[]
    for person, in replay.execute("SELECT DISTINCT person FROM candidates WHERE policy=? AND status='withheld' "
            "AND person IS NOT NULL AND json_extract(payload,'$.reason')='candidate_group'",(policy,)):
        source=replay.execute('SELECT o.id,o.scan,o.seconds,o.geometry,o.dimensions,o.embedding,o.quality,s.scene '
            'FROM candidates c JOIN observations o ON o.id='+observation_sql('c')+' JOIN scans s ON s.id=o.scan '
            "WHERE c.policy=? AND c.person=? AND c.status='withheld' AND o.rejected IS NULL AND "+anchor_sql('o',grouping)+
            ' ORDER BY CAST(s.scene AS INTEGER),s.scene,o.cue,o.id',(policy,person))
        chosen=select(source,grouping['max_samples_per_person'])
        if chosen:
            ids,samples=zip(*chosen);usage.add(samples,kind='Selected unresolved')
            pending.append(Group(person,tuple(samples),False,tuple(ids)))
    groups=tuple(confirmed)+tuple(pending)
    if len(groups)>grouping['max_people_per_lookup']:
        raise ValueError('Unresolved cohort lookup exceeds configured person budget.')
    for group in history.groups:
        usage.add(group.samples,kind='Pool historical strict')
    for group in history.unavailable:
        usage.add(group.samples,kind='Pool unavailable witnesses')
    for view in history.confirmations:
        usage.add(view.examples,kind='Pool confirmation evidence')
    for view in (v for v in history.views if v.authority!='strict'):
        usage.add(view.examples,kind='Pool historical view')
    replay.execute('CREATE TEMP TABLE pool_claims(observation TEXT PRIMARY KEY,source TEXT,scan TEXT,cue INTEGER,seconds REAL)')
    replay.execute('CREATE TEMP TABLE pool_promotions(source TEXT PRIMARY KEY,mode TEXT,selected_ids TEXT,proof TEXT)')
    from .historical_pool import resolve as resolve_history
    resolve_history(database,replay,selected,config,history,bindings,confirmed)
    from .pooled_views import resolve as resolve_views
    resolve_views(database,replay,selected,config,history,bindings,confirmed,early,usage)
    pending=[g for g in pending if replay.execute("SELECT 1 FROM candidates WHERE person=? AND status='withheld' "
        "AND json_extract(payload,'$.reason')='candidate_group' LIMIT 1",(g.identifier,)).fetchone()]
    groups=tuple(confirmed)+tuple(pending)
    witnesses={}
    blocked=set()
    support_only=set()
    threshold,cohesion,margin=settings['match_threshold'],grouping['cohesion_threshold'],settings['ambiguity_margin']
    peers=tuple(history.groups)+tuple(Group(bindings.get(g.identifier,g.identifier),g.samples,True) for g in groups)
    with tracking(rows(database.connection,selected,config,scope,namespace),'Collect unresolved identity support',unit='observation') as tracked:
        for row in tracked:
            if replay.execute('SELECT 1 FROM pool_claims WHERE observation=?',(row['id'],)).fetchone():
                continue
            query=vector(row);ranked=[]
            for group,mean,low in comparisons(groups,query,threshold-margin,cohesion):
                if low>=cohesion:
                    rank_person(ranked,(mean,bindings.get(group.identifier,group.identifier),group))
            if not unambiguous(ranked,threshold,margin):
                continue
            _,target,group=ranked[0]
            if group.known and _competing_anchor(tuple(history.groups)+tuple(history.unavailable),target,query,ranked[0][0],margin):
                continue
            if row['reason']!='candidate_group':
                support_only.add(group.identifier)
            reference=Group(target,group.samples,True)
            if not co_presence_allows(database.connection,row,reference,peers,query,config):
                continue
            pool=witnesses.get(group.identifier)
            if pool is None:
                pool=witnesses[group.identifier]=Witnesses(_source_rows(replay,group),grouping['max_samples_per_person'],usage)
            if not group.known and pool.rows:
                mean,low=_similarities(Group(group.identifier,pool.samples,False),query)
                if mean<threshold or low<cohesion:
                    blocked.add(group.identifier)
            if group.identifier not in blocked:
                pool.add(row)
                replay.execute('INSERT INTO pool_claims VALUES (?,?,?,?,?)',
                    (row['id'],group.identifier,row['scan'],row['cue'],row['seconds']))
    with tracking(groups,'Resolve pooled identity support',unit='cohort') as tracked:
        for group in tracked:
            pool=witnesses.get(group.identifier)
            if pool is None:
                continue
            if group.identifier in blocked:
                _deferred(replay,policy,group.identifier,'incoherent_frozen_support')
                continue
            current=replay.execute('SELECT o.id,o.scan,o.seconds FROM candidates c JOIN observations o ON o.id='+observation_sql('c')+
                " WHERE c.policy=? AND c.person=? AND c.status IN ('accepted','withheld') AND o.rejected IS NULL AND "+
                anchor_sql('o',grouping),(policy,group.identifier)).fetchall()
            if group.known:
                current=[row for row in current if row['id'] in group.sample_ids]
            current_ids=[row['id'] for row in current]
            if not current:
                continue
            support=replay.execute('SELECT COUNT(*) FROM (SELECT scan,seconds FROM pool_claims WHERE source=? UNION '
                'SELECT o.scan,o.seconds FROM observations o WHERE o.id IN (SELECT value FROM json_each(?)))',
                (group.identifier,encoded(current_ids))).fetchone()[0]
            mode='known_strict' if group.known else 'birth'
            match=None
            if not group.known:
                potential=[];match=rank(pool.rows,history,config,potential=potential)
                if match is not None:
                    if not _validate_known(database,replay,group.identifier,current_ids,match[0],history,config,bindings):
                        continue
                    support=replay.execute('SELECT COUNT(*) FROM (SELECT scan,seconds FROM pool_claims WHERE source=? UNION '
                        'SELECT o.scan,o.seconds FROM observations o WHERE o.id IN (SELECT value FROM json_each(?)))',
                        (group.identifier,encoded(current_ids))).fetchone()[0]
                    minimum=grouping['view_attach_min_support'] if match[1]['mode']=='view' else 1
                    if support<minimum:
                        continue
                    bindings[group.identifier]=match[0]
                    mode='known_view' if match[1]['mode']=='view' else 'known_strict'
                else:
                    from .identity_birth import potential as plausible
                    if plausible(pool.rows,history,config) is not None:
                        _deferred(replay,policy,group.identifier,'plausible_existing_identity',independent_support=support)
                        continue
                    if support<settings['min_identity_support'] or potential or group.identifier in support_only:
                        _deferred(replay,policy,group.identifier,'insufficient_or_known_view_support',
                            independent_support=support,required_support=settings['min_identity_support'],
                            historical_potential=bool(potential),query_only_support=group.identifier in support_only)
                        continue
                    # Missing references and near-but-incoherent known banks may
                    # defer birth; unavailable observations never vote positively.
                    if any(_similarities(g,q)[0]>=threshold-margin for g in tuple(history.groups)+tuple(history.unavailable)
                           for q in group.samples):
                        _deferred(replay,policy,group.identifier,'known_or_unavailable_reference_competes',
                            independent_support=support)
                        continue
                    samples=pool.samples
                    frozen=Group(group.identifier,samples,False)
                    if any(mean<threshold or low<cohesion for mean,low in (_similarities(frozen,q) for q in samples)):
                        _deferred(replay,policy,group.identifier,'incoherent_frozen_support',independent_support=support)
                        continue
            proof=dict(format=1,frozen=True,mode=mode,independent_support=support,
                required_support=(settings['min_identity_support'] if mode=='birth' else
                    grouping['view_attach_min_support'] if mode=='known_view' else 1),
                validation='coherent_frozen_union' if mode=='birth' else 'frozen_known_reference',
                source_reference_ids=group.sample_ids,
                representative_ids=[r['id'] for r in pool.rows],selected_ids=current_ids,
                context=history.context,threshold=threshold,cohesion_threshold=cohesion,
                historical_reference=match[1] if match else None)
            if not group.known:
                created=replay.execute('SELECT COALESCE(MAX(created),0)+1 FROM people').fetchone()[0]
                replay.execute('INSERT OR IGNORE INTO people VALUES (?,?,?,?,?,?)',
                    (group.identifier,scope,namespace,created,None,'pooled:'+group.identifier))
                reason=('pooled_view_identity' if mode=='known_view' else
                    'historical_identity' if mode=='known_strict' else 'cross_scene_provisional')
                replay.execute("UPDATE candidates SET status='accepted',payload=json_set(payload,'$.reason',?,"
                    "'$.provisional_pool',json(?)) WHERE policy=? AND person=? AND status='withheld'",
                    (reason,encoded(proof),policy,group.identifier))
                replay.execute('INSERT OR IGNORE INTO members SELECT '+observation_sql('c')+
                    ",c.person FROM candidates c WHERE c.policy=? AND c.person=? AND c.status='accepted'",(policy,group.identifier))
            replay.execute('INSERT INTO pool_promotions VALUES (?,?,?,?)',
                (group.identifier,mode,encoded(current_ids),encoded(proof)))


def apply(database,replay,config,mapping):
    """Backfill still-unresolved raw witnesses atomically, preserving their origin."""
    if not replay.execute("SELECT 1 FROM sqlite_temp_master WHERE name='pool_promotions'").fetchone():
        return []
    from .sample_keys import sample_key
    db=database.connection
    policy=matching_policy(config)
    applied=[]
    dirty=set()
    for promotion in replay.execute('SELECT * FROM pool_promotions ORDER BY source'):
        ids=json.loads(promotion['selected_ids']);proof=json.loads(promotion['proof'])
        if proof.get('external_reference'):
            witnesses=proof['source_reference_ids']
            members=db.execute('SELECT observation,person FROM members WHERE observation IN '
                '(SELECT value FROM json_each(?))',(encoded(witnesses),)).fetchall()
            if len(members)!=len(witnesses) or {r['person'] for r in members}!={proof['identity']}:
                continue
            target=proof['identity']
        else:
            members=replay.execute('SELECT observation,person FROM members WHERE observation IN '
                '(SELECT value FROM json_each(?))',(encoded(ids),)).fetchall()
            targets={mapping.get(r['person'],r['person']) for r in members}
            if len(members)!=len(ids) or len(targets)!=1:
                continue
            target=targets.pop()
        reason=('cross_scene_provisional' if promotion['mode']=='birth' else
                'pooled_view_identity' if promotion['mode'] in ('known_view','historical_view') else 'late_strict_identity')
        for observation, in replay.execute('SELECT observation FROM pool_claims WHERE source=? ORDER BY observation',(promotion['source'],)):
            if db.execute('SELECT 1 FROM members WHERE observation=?',(observation,)).fetchone():
                continue
            old=db.execute('SELECT c.* FROM candidates c WHERE '+observation_sql('c')+
                '=? ORDER BY c.rowid DESC LIMIT 1',(observation,)).fetchone()
            if old is None or old['status']!='withheld':
                continue
            row=db.execute('SELECT o.id,o.cue,o.seconds,o.source,o.rect,o.geometry,s.scope,s.namespace,s.scene,s.input_sha '
                'FROM observations o JOIN scans s ON s.id=o.scan WHERE o.id=?',(observation,)).fetchone()
            dirty.add((row['scope'],row['namespace'],row['scene']))
            db.execute('INSERT OR IGNORE INTO identity_origins VALUES (?,?,?)',(sample_key(row),target,observation))
            payload=json.loads(old['payload'])
            payload.update(reason=reason,prior_reason=payload.get('reason'),provisional_pool=json.loads(promotion['proof']),
                identity_basis='frozen_unresolved_pool',policy=policy,reference_role='strict' if reason=='cross_scene_provisional' else 'attachment',
                reference_selected=False)
            payload['provisional_pool']['identity']=target
            candidate=str(uuid.uuid5(uuid.UUID(observation),policy))
            db.execute('DELETE FROM candidates WHERE id=?',(candidate,))
            db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?)',(candidate,old['scan'],policy,target,'accepted',encoded(payload)))
            db.execute('INSERT INTO members VALUES (?,?)',(observation,target));applied.append(candidate)
    from .publication_queue import enqueue
    for scope,namespace,scene in sorted(dirty):
        enqueue(db,scope,namespace,scene,config,'unresolved_pool_backfill')
    return applied
