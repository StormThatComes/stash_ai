"""Read-only, frozen references outside the selected replay partition.

No historical observations or candidates are copied into replay. Only verified
label proposals and recovered selected memberships reach the final transaction.
Strict checkpoints therefore remain independent of this late binding stage.
"""
from dataclasses import dataclass
import hashlib

from .clustering import Group, _groups, matching_policy, vector
from .database import encoded
from .global_views import catalogue, rank_person, unambiguous, _cooccurs
from .pose_policy import anchor_sql
from .reference_roles import strict_sql, verified_sql
from .reference_compatibility import fallback_sql
from .reference_selection import select
from .reference_usage import ReferenceUsage
from .scanning import scan_identity
from .scoring import comparisons
from .view_prototypes import pose_view, pose_matches
from .rebuild_cache import _hash_row


@dataclass
class History:
    groups: tuple
    views: tuple
    people: dict
    context: str
    confirmations: tuple = ()


def _source(config,compatibility,*,verified=False):
    return ('FROM members m JOIN people p ON p.id=m.person JOIN observations o ON o.id=m.observation '
        'JOIN scans s ON s.id=o.scan WHERE p.scope=? AND p.namespace=? '
        'AND s.scene NOT IN (SELECT value FROM json_each(?)) AND (s.policy_sha=? OR '+compatibility+') '
        "AND s.scope=p.scope AND s.namespace=p.namespace AND s.status='complete' AND o.rejected IS NULL "
        'AND NOT EXISTS (SELECT 1 FROM identity_merges WHERE source=p.id) '
        'AND NOT EXISTS (SELECT 1 FROM scans newer WHERE newer.scope=s.scope AND newer.namespace=s.namespace '
        'AND newer.scene=s.scene AND newer.rowid>s.rowid) AND ' +
        ('1=1' if verified else anchor_sql('o',config.data['clustering'])) + ' AND ' +
        (verified_sql() if verified else strict_sql()))


def parameters(selected,config,scope,namespace):
    return (scope,namespace,encoded([item['scene'] for item in selected]),scan_identity(config,namespace))


def context(db,selected,config,scope,namespace):
    hasher=hashlib.sha256(b'frozen-external-references-v2')
    compatibility=fallback_sql(db,config,namespace)
    rows=db.execute("SELECT p.id,p.created,o.id,o.scan,o.cue,o.seconds,o.geometry,o.dimensions,o.embedding,"
        "CASE WHEN json_valid(o.quality) THEN json_extract(o.quality,'$.pixels','$.sharpness',"
        "'$.pose_deviation','$.roll_deviation') END " + _source(config,compatibility) +
        ' ORDER BY p.id,CAST(s.scene AS INTEGER),s.scene,o.cue,o.id',parameters(selected,config,scope,namespace))
    for row in rows:
        _hash_row(hasher,row)
    if config.data['clustering']['view_prototypes_enabled']:
        for row in db.execute('SELECT p.id,o.id,o.scan,o.cue,o.seconds,o.geometry,o.quality,o.dimensions,o.embedding '+
                _source(config,compatibility,verified=True)+' ORDER BY p.id,s.scene,o.cue,o.id',
                parameters(selected,config,scope,namespace)):
            _hash_row(hasher,row)
    return hasher.hexdigest()


def capture(db,selected,config,scope,namespace,fingerprint):
    usage=ReferenceUsage(config.data['clustering']['max_reference_bytes'])
    usage.begin('historical_references')
    source=_source(config,fallback_sql(db,config,namespace))
    args=parameters(selected,config,scope,namespace)
    people={}
    groups=[]
    for person in db.execute('SELECT DISTINCT p.* '+source+' ORDER BY p.created,p.id',args):
        if len(groups)>=config.data['clustering']['max_people_per_lookup']:
            raise ValueError('Historical identity lookup exceeds configured person budget.')
        rows=db.execute('SELECT o.id,o.scan,o.seconds,o.geometry,o.dimensions,o.embedding,o.quality,s.scene '+
            source+' AND p.id=? ORDER BY CAST(s.scene AS INTEGER),s.scene,o.cue,o.seconds,o.geometry,o.embedding,o.id',
            args+(person['id'],))
        selected_samples=select(rows,config.data['clustering']['max_samples_per_person'])
        if selected_samples:
            ids,samples=zip(*selected_samples)
            usage.add(samples,people=1,kind='Historical strict')
            groups.append(Group(person['id'],tuple(samples),True,tuple(ids)))
            people[person['id']]=tuple(person)
    views=confirmations=()
    grouping=config.data['clustering']
    if grouping['anchor_view_matching'] or (grouping['max_anchor_pose_deviation'] is not None and grouping['auxiliary_matching']):
        confirmations=catalogue(db,config,scope,namespace,usage=usage,complete=True,minimum=1,
            excluded_scenes=[item['scene'] for item in selected],scan_policy=scan_identity(config,namespace))
        if grouping['anchor_view_matching']:
            views=tuple(v for v in confirmations if v.support_count>=grouping['view_min_support'])
    if config.data['clustering']['view_prototypes_enabled'] and config.data['clustering']['auxiliary_matching']:
        from .verified_views import historical_views
        views=tuple(views)+historical_views(db,config,scope,namespace,selected,tuple(groups),usage,confirmations)
    return History(tuple(groups),views,people,fingerprint,confirmations)


def rank(rows,history,config,*,decisions=None,potential=None):
    settings,grouping=config.data['recognition'],config.data['clustering']
    threshold=settings['match_threshold']
    view_threshold=max(threshold,grouping['view_match_threshold'])
    margin=max(settings['ambiguity_margin'],grouping['view_ambiguity_margin'])
    views=[view.group for view in history.views]
    by_group={id(view.group):view for view in history.views}
    aggregate={}
    for row in rows:
        query=vector(row)
        choices={}
        for group,score,low in comparisons(history.groups,query,threshold-margin,grouping['cohesion_threshold']):
            if low>=grouping['cohesion_threshold']:
                choices[group.identifier]=(score,'strict',None,score>=threshold)
        for group,score,low in comparisons(views,query,view_threshold-margin,grouping['view_cohesion_threshold']):
            view=by_group[id(group)]
            if low<grouping['view_cohesion_threshold'] or not pose_matches(row['geometry'],view.pose):
                continue
            previous=choices.get(group.identifier)
            if previous is not None and previous[1]=='strict' and previous[3]:
                continue
            if previous is None or score>previous[0]:
                choices[group.identifier]=(score,'view',view.medoid,score>=view_threshold)
        if potential is not None and not potential and any(value[3] for value in choices.values()):
            potential.append(True)
        for person,value in choices.items():
            aggregate.setdefault(person,[]).append(value)
    ranked=[]
    for person,values in aggregate.items():
        if len(values)==len(rows):
            rank_person(ranked,(sum(v[0] for v in values)/len(values),person,values))
    decision_margin = (margin if ranked and any(v[1]=='view' for v in ranked[0][2])
                       else settings['ambiguity_margin'])
    if not unambiguous(ranked,threshold,decision_margin) or not all(v[3] for v in ranked[0][2]):
        if (decisions is not None and ranked and ranked[0][0]>=threshold
                and all(v[3] for v in ranked[0][2])):
            decisions.append(dict(outcome='ambiguous', context=history.context,
                rivals=[dict(identity=r[1],score=r[0]) for r in ranked],
                ambiguity_margin=decision_margin))
        return None
    score,person,values=ranked[0]
    from .anchor_views import _competing_anchor
    if any(_competing_anchor(history.groups,person,vector(row),value[0],decision_margin)
           for row,value in zip(rows,values)):
        if decisions is not None:
            decisions.append(dict(outcome='competing_anchor',context=history.context,
                rivals=[dict(identity=r[1],score=r[0]) for r in ranked],
                ambiguity_margin=decision_margin))
        return None
    return person,dict(format=2,frozen=True,context=history.context,identity=person,
        view_evidence=[dict(medoid=v.medoid,authority=v.authority,support=v.support_count,support_ids=v.support)
            for v in history.views if v.medoid in {value[2] for value in values}],
        mode='view' if any(v[1]=='view' for v in values) else 'strict',mean_score=score,
        medoids=sorted({v[2] for v in values if v[2] is not None}),ambiguity_margin=decision_margin)


def reconcile(database,replay,selected,config,scope,namespace,fingerprint,*,progress=None,history=None):
    """Return label bindings; mutate only the private selected replay."""
    history=history or capture(database.connection,selected,config,scope,namespace,fingerprint)
    if not history.groups:
        return {}
    grouping=config.data['clustering']
    groups=_groups(replay,scope,namespace,grouping)
    # Both frozen catalogues fit the same configured reference budget.
    usage=ReferenceUsage(grouping['max_reference_bytes'],progress)
    usage.begin('selected_and_historical')
    for group in list(history.groups)+groups:
        usage.add(group.samples,people=1,kind='Strict')
    for view in history.views:
        usage.add(view.examples,kind='Historical view')
    if len(history.groups)+len(groups)>grouping['max_people_per_lookup']:
        raise ValueError('Selected and historical identities exceed configured person budget.')
    proposals={}
    deferred=set()
    partial_sources=set()
    for group in groups:
        rows=replay.execute('SELECT id,geometry,dimensions,embedding FROM observations '
            'WHERE id IN (SELECT value FROM json_each(?)) ORDER BY id',(encoded(group.sample_ids),)).fetchall()
        decisions=[]
        potential=[]
        result=rank(rows,history,config,decisions=decisions,potential=potential) if rows else None
        if potential:
            partial_sources.add(group.identifier)
        # Existing selected labels can still be restored from ownership votes.
        # New local clusters with global rivals must not publish another person.
        owned=replay.execute('SELECT 1 FROM members m JOIN sample_keys k ON k.observation=m.observation '
            'JOIN origins o ON o.sample_key=k.sample_key JOIN old_people p ON p.id=o.person '
            'WHERE m.person=? AND p.retired=0 LIMIT 1',(group.identifier,)).fetchone()
        if result is None and decisions and not owned:
            deferred.add(group.identifier)
            replay.execute("UPDATE candidates SET person=NULL,status='withheld',payload=json_set(payload,"
                "'$.prior_reason',json_extract(payload,'$.reason'),'$.reason','historical_ambiguous_identity',"
                "'$.global_identity_decision',json(?)) WHERE person=?",
                (encoded(decisions[0]),group.identifier))
            replay.execute('DELETE FROM members WHERE person=?',(group.identifier,))
        if result is not None:
            if result[1]['mode']=='view':
                support=replay.execute('SELECT COUNT(*) FROM (SELECT o.scan,o.seconds FROM members m '
                    'JOIN observations o ON o.id=m.observation WHERE m.person=? GROUP BY o.scan,o.seconds)',
                    (group.identifier,)).fetchone()[0]
                if support<grouping['view_attach_min_support']:
                    continue
            proposals[group.identifier]=result
    groups=[group for group in groups if group.identifier not in deferred]
    # Two distinct co-present faces cannot both adopt the same historical label.
    blocked=set()
    by_target={}
    for source,(target,_) in proposals.items():
        by_target.setdefault(target,[]).append(source)
    for sources in by_target.values():
        for offset,left in enumerate(sources):
            for right in sources[offset+1:]:
                if _cooccurs(replay,left,right):
                    blocked.update((left,right))
    bindings={}
    for source,(target,proof) in proposals.items():
        if source in blocked:
            continue
        bindings[source]=target
        reason='historical_view_identity' if proof['mode']=='view' else 'historical_identity'
        replay.execute("UPDATE candidates SET payload=json_set(payload,'$.historical_reference',json(?),"
                       "'$.prior_reason',json_extract(payload,'$.reason'),'$.reason',CASE WHEN "
                       "json_extract(payload,'$.reason') IN ('candidate_group','known_identity','cross_scene_identity',"
                       "'cross_scene_provisional') THEN ? ELSE json_extract(payload,'$.reason') END) "
                       'WHERE person=?',(encoded(proof),reason,source))
    # Withheld observations can also match an existing global person directly.
    # All decisions use the original history, never any new selected membership.
    from .clustering import co_presence_allows
    from .anchor_views import _competing_anchor
    policy=matching_policy(config)
    mapped_groups=[Group(bindings.get(g.identifier,g.identifier),g.samples,True) for g in groups]
    reference_groups=list(history.groups)+[view.group for view in history.views]+mapped_groups
    pending=[]
    for item in selected:
        rows=replay.execute('SELECT o.*,c.id AS candidate FROM candidates c JOIN observations o '
            "ON o.id=json_extract(c.payload,'$.observation') WHERE c.scan=? AND c.policy=? "
            "AND c.status='withheld' AND o.rejected IS NULL AND " + anchor_sql('o',grouping) +
            ' ORDER BY o.cue,o.geometry,o.embedding,o.id',(item['scan'],policy)).fetchall()
        for row in rows:
            result=rank([row],history,config)
            if result is None:
                continue
            target,proof=result
            query=vector(row)
            if _competing_anchor(mapped_groups,target,query,proof['mean_score'],proof['ambiguity_margin']):
                continue
            if any(bindings.get(g.identifier,g.identifier)==target and replay.execute('SELECT 1 FROM members m JOIN observations o '
                    'ON o.id=m.observation WHERE m.person=? AND o.scan=? AND o.cue=? LIMIT 1',
                    (g.identifier,row['scan'],row['cue'])).fetchone() for g in groups):
                continue
            references=next((g for g in reference_groups if g.identifier==target and
                co_presence_allows(replay,row,g,reference_groups,query,config)),None)
            if references is None:
                continue
            pending.append((row,target,proof))
    # Frozen matching evaluates each view independently. Resolve all claims to
    # the same cue before accepting any, including disjoint historical views.
    # Exact duplicate detections still represent a single face.
    cue_claims={}
    blocked_cues=set()
    for row,target,_ in pending:
        key=(target,row['scan'],row['cue'])
        signature=(row['geometry'],row['embedding'])
        if key in cue_claims and cue_claims[key]!=signature:
            blocked_cues.add(key)
        else:
            cue_claims[key]=signature
    pending=[(row,target,proof) for row,target,proof in pending
             if (target,row['scan'],row['cue']) not in blocked_cues]
    support={}
    for row,target,proof in pending:
        key=(target,tuple(proof['medoids']))
        support.setdefault(key,set()).add((row['scan'],row['seconds']))
    for row,target,proof in pending:
        if proof['mode']=='view' and len(support[(target,tuple(proof['medoids']))])<grouping['view_attach_min_support']:
            continue
        replay.execute('INSERT OR IGNORE INTO people VALUES (?,?,?,?,?,?)',history.people[target])
        replay.execute('INSERT OR IGNORE INTO members VALUES (?,?)',(row['id'],target))
        reason='historical_view_identity' if proof['mode']=='view' else 'historical_identity'
        replay.execute("UPDATE candidates SET person=?,status='accepted',payload=json_set(payload,"
            "'$.reason',?,'$.historical_reference',json(?)) WHERE id=?",(target,reason,encoded(proof),row['candidate']))
        bindings[target]=target
    from .cohort_resolution import recover
    recover(replay,groups,history,config,bindings,source_ids=partial_sources)
    return bindings
