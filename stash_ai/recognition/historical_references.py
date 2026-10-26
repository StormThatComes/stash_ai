"""Read-only, frozen references outside the selected replay partition.

No historical observations or candidates are copied into replay. Frozen global
label/role repairs commit atomically with selected memberships. Strict checkpoints
remain independent of this late reconciliation stage.
"""
from dataclasses import dataclass,field
import hashlib

from .clustering import Group, _groups, matching_policy, vector
from .database import encoded
from .global_views import catalogue, rank_person, unambiguous, _cooccurs
from .pose_policy import anchor_sql,query_sql,restricted
from .reference_roles import strict_sql, verified_sql, attachment_sql
from .reference_compatibility import fallback_sql
from .reference_selection import select
from .reference_usage import ReferenceUsage
from .scanning import scan_identity
from .scoring import comparisons
from .view_prototypes import pose_view, pose_matches, query_pose_matches
from .rebuild_cache import _hash_row


@dataclass
class History:
    groups: tuple
    views: tuple
    people: dict
    context: str
    confirmations: tuple = ()
    unavailable: tuple = ()
    aliases: dict = field(default_factory=dict)
    actions: tuple = ()


def _source(config,compatibility,*,verified=False,gaps=False,attachment=False):
    from .reference_gaps import latest_clause
    from .scan_heads import predicate as current_scan
    latest=latest_clause() if gaps else current_scan()
    return ('FROM members m JOIN people p ON p.id=m.person JOIN observations o ON o.id=m.observation '
        'JOIN scans s ON s.id=o.scan WHERE p.scope=? AND p.namespace=? '
        'AND s.scene NOT IN (SELECT value FROM json_each(?)) AND (s.policy_sha=? OR '+compatibility+') '
        "AND s.scope=p.scope AND s.namespace=p.namespace AND s.status='complete' AND o.rejected IS NULL "
        'AND NOT EXISTS (SELECT 1 FROM identity_merges WHERE source=p.id) '
        'AND '+latest+' AND ' +
        (query_sql('o',config.data['clustering']) if verified or attachment else anchor_sql('o',config.data['clustering'])) + ' AND ' +
        (attachment_sql() if attachment else
         verified_sql() if verified else strict_sql()))


def parameters(selected,config,scope,namespace):
    return (scope,namespace,encoded([item['scene'] for item in selected]),scan_identity(config,namespace))


def context(db,selected,config,scope,namespace,*,cache=False):
    if not cache:
        return _context(db,selected,config,scope,namespace)
    from .dependencies import cached
    from .config import digest
    key=digest([scope,namespace,selected])
    return cached(db,'historical-context',key,matching_policy(config),
                  lambda:_context(db,selected,config,scope,namespace))


def _context(db,selected,config,scope,namespace):
    from .scan_heads import predicate as current_scan
    hasher=hashlib.sha256(b'frozen-external-references-v5')
    from .pose_policy import anchor_visibility
    visibility_fields=",'$.visible_face_fraction','$.visibility.contract'" if anchor_visibility(config.data['clustering']) else ''
    compatibility=fallback_sql(db,config,namespace)
    rows=db.execute("SELECT p.id,p.created,o.id,o.scan,o.cue,o.seconds,o.geometry,o.dimensions,o.embedding,"
        "CASE WHEN json_valid(o.quality) THEN json_extract(o.quality,'$.pixels','$.sharpness',"
        "'$.pose_deviation','$.roll_deviation','$.alignment_valid','$.alignment_error','$.alignment_face_coverage','$.evidence_crop_sha256','$.evidence_pixel_sha256','$.evidence_thumbnail','$.evidence_aspect','$.evidence_thumbnail_id'"+visibility_fields+") END " + _source(config,compatibility) +
        ' ORDER BY p.id,CAST(s.scene AS INTEGER),s.scene,o.cue,o.id',parameters(selected,config,scope,namespace))
    for row in rows:
        _hash_row(hasher,row)
    for row in db.execute('SELECT s.id,s.scene,s.input_sha,s.policy_sha,s.cursor,s.status FROM scans s '
            'WHERE s.scope=? AND s.namespace=? AND s.scene NOT IN (SELECT value FROM json_each(?)) '
            'AND '+current_scan()+' ORDER BY s.scene',
            (scope,namespace,encoded([item['scene'] for item in selected]))):
        _hash_row(hasher,row)
    for row in db.execute('SELECT p.id,o.id,o.geometry,o.quality,o.dimensions,o.embedding '+
            _source(config,compatibility,gaps=True)+' ORDER BY p.id,o.id',parameters(selected,config,scope,namespace)):
        _hash_row(hasher,row)
    if config.data['clustering']['view_prototypes_enabled']:
        for row in db.execute('SELECT p.id,o.id,o.scan,o.cue,o.seconds,o.geometry,o.quality,o.dimensions,o.embedding,'
                '(SELECT json_extract(c.payload,\'$.strict_confirmation\') FROM candidates c '
                'WHERE CAST(CASE WHEN json_valid(c.payload) THEN json_extract(c.payload,\'$.observation\') END AS TEXT)=o.id ORDER BY c.rowid DESC LIMIT 1) '+
                _source(config,compatibility,verified=True)+' ORDER BY p.id,s.scene,o.cue,o.id',
                parameters(selected,config,scope,namespace)):
            _hash_row(hasher,row)
    from .unresolved_pool import hash_context
    if config.data['clustering']['corroborated_view_enrichment']:
        for row in db.execute("SELECT p.id,o.id,o.scan,o.seconds,o.geometry,o.dimensions,o.embedding,"
                "CASE WHEN json_valid(o.quality) THEN json_extract(o.quality,'$.pixels','$.sharpness',"
                "'$.pose_deviation','$.roll_deviation','$.alignment_valid','$.alignment_error','$.alignment_face_coverage','$.evidence_crop_sha256','$.evidence_pixel_sha256','$.evidence_thumbnail','$.evidence_aspect','$.evidence_thumbnail_id'"+visibility_fields+") END "+
                _source(config,compatibility,attachment=True)+' ORDER BY p.id,o.id',
                parameters(selected,config,scope,namespace)):
            _hash_row(hasher,row)
    hash_context(db,selected,config,scope,namespace,hasher)
    return hasher.hexdigest()


def capture(db,selected,config,scope,namespace,fingerprint,*,resolve_aliases=True):
    usage=ReferenceUsage(config.data['clustering']['max_reference_bytes'])
    usage.begin('historical_references')
    source=_source(config,fallback_sql(db,config,namespace))
    args=parameters(selected,config,scope,namespace)
    people={}
    groups=[]
    from itertools import groupby
    # Keep the original person and per-person sample ordering in one SQL stream.
    # groupby hands select() only the current person's bounded streaming rows.
    rows = db.execute('SELECT p.id AS owner,p.created AS owner_created,p.scope AS owner_scope,'
        'p.namespace AS owner_namespace,p.performer AS owner_performer,p.token AS owner_token,'
        'o.id,o.scan,o.seconds,o.geometry,o.dimensions,o.embedding,o.quality,s.scene '+source+
        ' ORDER BY p.created,p.id,CAST(s.scene AS INTEGER),s.scene,o.cue,o.seconds,o.geometry,o.embedding,o.id', args)
    for identifier, records in groupby(rows, key=lambda row: row['owner']):
        if len(groups)>=config.data['clustering']['max_people_per_lookup']:
            raise ValueError('Historical identity lookup exceeds configured person budget.')
        first = next(records)
        from itertools import chain
        selected_samples=select(chain((first,), records),config.data['clustering']['max_samples_per_person'])
        if selected_samples:
            ids,samples=zip(*selected_samples)
            usage.add(samples,people=1,kind='Historical strict')
            groups.append(Group(identifier,tuple(samples),True,tuple(ids)))
            people[identifier]=(identifier,first['owner_scope'],first['owner_namespace'],
                first['owner_created'],first['owner_performer'],first['owner_token'])
    views=confirmations=()
    grouping=config.data['clustering']
    if grouping['anchor_view_matching'] or (restricted(grouping) and grouping['auxiliary_matching']):
        confirmations=catalogue(db,config,scope,namespace,usage=usage,complete=True,minimum=1,
            excluded_scenes=[item['scene'] for item in selected],scan_policy=scan_identity(config,namespace))
        if grouping['anchor_view_matching']:
            views=tuple(v for v in confirmations if v.support_count>=grouping['view_min_support'])
    from .anchor_quarantine import views as quarantined_views
    views+=quarantined_views(db,groups,config,usage)
    from .reference_gaps import capture as gaps
    history=History(tuple(groups),views,people,fingerprint,confirmations,
        gaps(db,selected,config,scope,namespace,usage))
    if resolve_aliases:
        from .reference_registry import resolve
        resolve(db,history,config,usage)
        refresh_verified(db,selected,config,scope,namespace,history,usage)
    return history


def refresh_verified(db,selected,config,scope,namespace,history,usage):
    history.views=tuple(v for v in history.views if v.authority!='corroborated_view')
    if config.data['clustering']['view_prototypes_enabled'] and config.data['clustering']['auxiliary_matching']:
        from .verified_views import historical_views
        history.views=tuple(v for v in history.views if v.authority in ('strict','quarantined_core'))+historical_views(
            db,config,scope,namespace,selected,history.groups,usage,history.confirmations,history.aliases)
    from .corroborated_views import enrich
    history.views+=enrich(db,config,scope,namespace,selected,history,usage)


def _build_rank_bank(history, config, strict_appearance):
    grouping, settings = config.data['clustering'], config.data['recognition']
    views=[];by_group={}
    minimum=max(grouping['view_min_support'],settings['min_merge_support'])
    for view in history.views:
        if view.authority=='quarantined_core' and not strict_appearance:
            continue
        if strict_appearance:
            if (view.authority not in ('strict','quarantined_core') or view.support_count<minimum or
                    view.authority=='quarantined_core' and not grouping.get('anchor_outlier_quarantine')):
                continue
            group=Group(view.group.identifier,view.examples,True,view.support)
        else:
            # A supported mode is scored against its witnesses, not just its
            # medoid. A corroborated promotion is a specifically supported edge
            # representative: averaging its broader parent/query cohort erases
            # that edge. Raw strict rivals remain in their unchanged bank.
            examples = getattr(view, 'examples', ())
            group = (Group(view.group.identifier, tuple(examples), True, view.support)
                     if examples and view.support_count >= minimum and
                     view.authority != 'corroborated_view' else view.group)
        views.append(group);by_group[id(group)]=view
    return views, by_group


def _rank_bank(history, config, strict_appearance):
    from .run_cache import current, MISSING
    cache=current()
    if cache is None:return _build_rank_bank(history,config,strict_appearance)
    views=tuple(history.views)
    g=config.data['clustering'];r=config.data['recognition']
    key=('rank-bank',tuple(id(v) for v in views),strict_appearance,g['view_min_support'],
         r['min_merge_support'],bool(g.get('anchor_outlier_quarantine')))
    found=cache.get(key)
    if found is not MISSING and all(a is b for a,b in zip(found[1],views)):return found[0]
    result=_build_rank_bank(history,config,strict_appearance)
    cache.put(key,(result,views),4096+sum(512+sum(s.nbytes for s in group.samples) for group in result[0]))
    return result


def rank_many(rows, history, config, **options):
    """Batch numerical work; preserve the original scalar ranking and proof.

    All exact scores are returned (no worker-side loss of weaker rivals).
    The original coordinator screening still controls proof/rival visibility.
    """
    from .parallel_scores import scored_rows
    from stash_ai.runtime.cleanup import closing
    bank = _rank_bank(history, config, options.get('strict_appearance', False))
    groups = tuple(history.groups) + tuple(bank[0])
    settings, grouping = config.data['recognition'], config.data['clustering']
    margin = max(settings['ambiguity_margin'], grouping['view_ambiguity_margin'])
    floor = settings['match_threshold'] - margin
    cohesion = min(grouping['cohesion_threshold'], grouping['view_cohesion_threshold'])
    with closing(scored_rows(rows, groups, floor, cohesion), name='close late ranking scores') as stream:
        for row, frozen in stream:
            scores = ({id(groups[position]): (mean, low) for position, mean, low in frozen}
                      if frozen is not None else None)
            bounds = ({id(groups[position]): value for position,value in frozen.bounds.items()}
                      if frozen is not None and hasattr(frozen,'bounds') else None)
            yield row, rank([row], history, config, _bank=bank, _scores=scores, _bounds=bounds, **options)


def rank(rows,history,config,*,decisions=None,potential=None,relaxed_pose=False,strict_appearance=False,quarantine_only=False,_bank=None,_scores=None,_bounds=None):
    settings,grouping=config.data['recognition'],config.data['clustering']
    threshold=settings['match_threshold']
    view_threshold=max(threshold,grouping['view_match_threshold'])
    margin=max(settings['ambiguity_margin'],grouping['view_ambiguity_margin'])
    views,by_group = _bank if _bank is not None else _rank_bank(history, config, strict_appearance)
    effective_view_threshold=threshold if strict_appearance else view_threshold
    effective_view_cohesion=grouping['cohesion_threshold'] if strict_appearance else grouping['view_cohesion_threshold']
    from .clustering import _similarities
    from .run_cache import current, immutable_samples, MISSING
    def exact(group, query):
        if _scores is not None and id(group) in _scores:return _scores[id(group)]
        cache=current();samples=immutable_samples(group.samples) if cache is not None else None
        if samples is None:return _similarities(group,query)
        key=('query-score',samples,query.tobytes())
        value=cache.get(key)
        if value is MISSING:
            value=_similarities(group,query)
            cache.put(key,value,sum(len(s)+128 for s in samples)+query.nbytes+512)
        return value
    def compare(groups, query, floor, cohesion):
        if _scores is None or _bounds is None:
            yield from comparisons(groups,query,floor,cohesion,exact=exact)
            return
        import math
        for group in groups:
            bound = _bounds.get(id(group))
            # Small banks deliberately retain their old scalar visibility.
            if len(groups) >= 16 and bound is not None and all(math.isfinite(v) for v in bound) and (bound[0] < floor or bound[1] < cohesion):
                continue
            # A missing sparse value is uncomputed, never an invented score.
            yield group, *exact(group,query)
    aggregate={};strict_rows=[];mixed_rows=[]
    for row in rows:
        query=vector(row)
        choices={}
        for group,score,low in compare(history.groups,query,threshold-margin,grouping['cohesion_threshold']):
            if low>=grouping['cohesion_threshold']:
                if group.identifier not in choices or score>choices[group.identifier][0]:
                    choices[group.identifier]=(score,'strict',None,score>=threshold)
        strict_rows.append(dict(choices))
        for group,score,low in compare(views,query,effective_view_threshold-margin,effective_view_cohesion):
            view=by_group[id(group)]
            ordinary_pose = pose_matches(row['geometry'], view.pose)
            premium, required = 0., 2
            if not ordinary_pose:
                if not relaxed_pose:
                    continue
                from .view_prototypes import pose_fallback
                premium, required = pose_fallback(row['geometry'], view.pose)
                if view.support_count < required:
                    continue
            if low < effective_view_cohesion:
                continue
            previous=choices.get(group.identifier)
            if previous is not None and previous[3] and score<effective_view_threshold:
                continue
            if previous is not None and previous[1]=='strict' and previous[3] and previous[0]>=score:
                continue
            admissible=score>=effective_view_threshold+premium and (not quarantine_only or view.authority=='quarantined_core')
            if previous is None or score>previous[0] or score==previous[0] and admissible and not previous[3]:
                choices[group.identifier]=(score,'view',view.medoid,admissible,view)
        mixed_rows.append(choices)
        if potential is not None and not potential and any(value[3] for value in choices.values()):
            potential.append(True)
        for person,value in choices.items():
            aggregate.setdefault(person,[]).append(value)
    ranked=[]
    for person,values in aggregate.items():
        if len(values)==len(rows):
            rank_person(ranked,(sum(v[0] for v in values)/len(values),person,values))
    from .anchor_views import _competing_anchor
    strict_checked=False
    ranking_score=ranked[0][0] if ranked else None
    if ranked and any(v[1]=='view' for v in ranked[0][2]):
        person=ranked[0][1];direct=[choices.get(person) for choices in strict_rows]
        # A stronger own view improves ranking, but cannot revoke an already
        # independent strict proof. Check that proof against every mixed rival
        # and the original raw-anchor competition using its own score/margin.
        if all(value is not None and value[3] for value in direct) and all(
            all(other==person or value[0]-rival[0]>=settings['ambiguity_margin']
                for other,rival in choices.items()) for value,choices in zip(direct,mixed_rows)):
            strict_checked=not any(_competing_anchor(history.groups,person,vector(row),value[0],settings['ambiguity_margin'])
                                   for row,value in zip(rows,direct))
            if strict_checked:ranked[0]=(sum(v[0] for v in direct)/len(direct),person,direct)
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
    # A rival need not explain the whole cohort to make one of its queries
    # ambiguous. Preserve every mixed rival at its original observation.
    if any(any(other != person and value[0]-rival[0] < decision_margin
               for other,rival in choices.items())
           for value,choices in zip(values,mixed_rows)):
        if decisions is not None:
            decisions.append(dict(outcome='per_query_competing_identity',context=history.context,
                                  ambiguity_margin=decision_margin))
        return None
    if not strict_checked and any(_competing_anchor(history.groups,person,vector(row),value[0],decision_margin)
           for row,value in zip(rows,values)):
        if decisions is not None:
            decisions.append(dict(outcome='competing_anchor',context=history.context,
                rivals=[dict(identity=r[1],score=r[0]) for r in ranked],
                ambiguity_margin=decision_margin))
        return None
    from .view_prototypes import pose_fallback
    return person,dict(format=2,frozen=True,context=history.context,identity=person,
        view_evidence=[dict(medoid=v.medoid,authority=v.authority,support=v.support_count,
                           support_ids=getattr(v,'source_ids',v.support),
                           **({'promotion':v.promotion} if getattr(v,'promotion',None) else {}))
            for v in {id(value[4]):value[4] for value in values if value[1]=='view'}.values()],
        mode='view' if any(v[1]=='view' for v in values) else 'strict',mean_score=score,ranking_score=ranking_score,
        pose_query_support=max([2]+[pose_fallback(row['geometry'],value[4].pose)[1]
            for row,value in zip(rows,values) if value[1]=='view' and relaxed_pose]),
        strict_appearance=strict_appearance,threshold=effective_view_threshold,cohesion_threshold=effective_view_cohesion,
        rival_score=ranked[1][0] if len(ranked)>1 else None,
        medoids=sorted({v[2] for v in values if v[2] is not None}),ambiguity_margin=decision_margin)


def reconcile(database,replay,selected,config,scope,namespace,fingerprint,*,progress=None,history=None,early=(),excluded_views=()):
    """Freeze global proposals; durable changes wait for the final transaction."""
    history=history or capture(database.connection,selected,config,scope,namespace,fingerprint,resolve_aliases=False)
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
    from .reference_registry import resolve
    proposals=resolve(database.connection,history,config,usage,replay=replay,groups=groups,early=early,excluded_views=excluded_views)
    refresh_verified(database.connection,selected,config,scope,namespace,history,usage)
    deferred=set()
    partial_sources=set()
    for group in groups:
        if group.identifier in proposals:
            continue
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
        if result is None and group.identifier not in deferred:
            from .reference_gaps import defer
            if defer(replay,group,history,config):
                deferred.add(group.identifier)
        if result is None and not owned and group.identifier not in deferred:
            from .identity_birth import potential as plausible,defer as defer_birth
            proof=plausible(rows,history,config)
            if proof is not None:
                defer_birth(replay,group.identifier,proof);deferred.add(group.identifier)
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
                if _cooccurs(replay,left,right,config):
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
        for row,result in rank_many(rows,history,config):
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
    from .physical_faces import duplicate
    for row,target,_ in pending:
        key=(target,row['scan'],row['cue'])
        previous=cue_claims.setdefault(key,[])
        if any(not duplicate(row,other,grouping.get('deduplicate_detections',False)) for other in previous):
            blocked_cues.add(key)
        previous.append(row)
    pending=[(row,target,proof) for row,target,proof in pending
             if (target,row['scan'],row['cue']) not in blocked_cues]
    from .query_support import eligible
    claims=[dict(row,owner=target,mode=proof['mode'],medoids=encoded(proof['medoids'])) for row,target,proof in pending]
    claims.sort(key=lambda r:(r['embedding'],r['geometry'],r['id']))
    supported=eligible(claims,config,key=lambda r:(r['owner'],r['mode']),prototype=lambda r:r['medoids'],
                       minimum=lambda family:grouping['view_attach_min_support'] if family[1]=='view' else 1)
    for row,target,proof in pending:
        if row['id'] not in supported:
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
