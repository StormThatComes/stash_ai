"""Recover independently matched cohorts without publishing their outliers.

Every query sees the original frozen bank. A selected-fragment repair additionally
requires a strict majority of independent times, some unmatched evidence, and a
higher coverage than any target that is itself proposed for repair. Equal cycles
stay unresolved. No recovered query becomes another query's reference.
"""
from dataclasses import replace

from .clustering import Group,_groups,matching_policy,vector,co_presence_allows
from .database import encoded
from .decision_resolution import resolve_pending
from .global_views import _cooccurs
from .progress import tracking
from .scoring import scoped


@scoped
def recover(db,groups,history,config,bindings,*,selected_only=False,source_ids=None):
    from .historical_references import rank
    from .anchor_views import _competing_anchor
    policy=matching_policy(config)
    from .independent_evidence import count, grouped_counts
    minimum=config.data['clustering']['view_attach_min_support']
    db.execute('CREATE TEMP TABLE cohort_claims(observation TEXT PRIMARY KEY,candidate TEXT,source TEXT,target TEXT,mode TEXT,medoids TEXT,proof TEXT)')
    try:
        totals={}
        for group in groups:
            if group.identifier in bindings or (source_ids is not None and group.identifier not in source_ids):
                continue
            frozen=replace(history,groups=tuple(g for g in history.groups if g.identifier!=group.identifier),
                           views=tuple(v for v in history.views if v.group.identifier!=group.identifier))
            # Avoid scanning every member when none of its bounded representatives
            # can match. Accepted members are streamed only for plausible repairs.
            samples=db.execute('SELECT id,geometry,dimensions,embedding FROM observations WHERE id IN '
                '(SELECT value FROM json_each(?))',(encoded(group.sample_ids),)).fetchall()
            if not any(rank([row],frozen,config) is not None for row in samples):
                continue
            rivals=[Group(bindings.get(g.identifier,g.identifier),g.samples,True) for g in groups
                    if g.identifier!=group.identifier]
            references=list(frozen.groups)+[v.group for v in frozen.views]+rivals
            totals[group.identifier]=count(db.execute('SELECT o.scan,o.seconds,o.quality FROM members m '
                'JOIN observations o ON o.id=m.observation WHERE m.person=?', (group.identifier,)))
            rows=db.execute('SELECT o.*,c.id AS candidate FROM members m JOIN observations o ON o.id=m.observation '
                "JOIN candidates c ON json_extract(c.payload,'$.observation')=o.id AND c.policy=? "
                "WHERE m.person=? AND c.person=m.person AND c.status='accepted' ORDER BY o.scan,o.cue,o.id",
                (policy,group.identifier))
            with tracking(rows,'Resolve frozen identity cohorts',unit='observation') as tracked:
                for row in tracked:
                    result=rank([row],frozen,config)
                    if result is None:
                        continue
                    target,proof=result
                    query=vector(row)
                    if _competing_anchor(rivals,target,query,proof['mean_score'],proof['ambiguity_margin']):
                        continue
                    if any(other!=group.identifier and mapped==target and _cooccurs(db,group.identifier,other,config)
                           for other,mapped in bindings.items()):
                        continue
                    target_refs=[v.group for v in frozen.views if v.medoid in proof['medoids']]
                    target_refs.extend(g for g in frozen.groups if g.identifier==target)
                    if not any(co_presence_allows(db,row,ref,references,query,config) for ref in target_refs):
                        continue
                    db.execute('INSERT INTO cohort_claims VALUES (?,?,?,?,?,?,?)',
                        (row['id'],row['candidate'],group.identifier,target,proof['mode'],encoded(proof['medoids']),encoded(proof)))
        def insufficient(pending=False):
            state=" AND c.status='pending' AND c.person=q.target" if pending else ''
            from .query_support import eligible
            query=('SELECT o.id,o.scan,o.seconds,o.quality,o.dimensions,o.embedding,q.source,q.target,q.mode,q.medoids FROM cohort_claims q '
                'JOIN observations o ON o.id=q.observation JOIN candidates c ON c.id=q.candidate WHERE 1=1'+state+
                ' ORDER BY o.embedding,o.geometry,o.id')
            supported=eligible(lambda:db.execute(query),config,key=lambda r:(r['source'],r['target'],r['mode']),prototype=lambda r:r['medoids'],
                minimum=lambda family:minimum if family[2]=='view' else 1)
            return [r['id'] for r in db.execute(query) if r['id'] not in supported]
        for observation in insufficient():
            db.execute('DELETE FROM cohort_claims WHERE observation=?',(observation,))
        counts=grouped_counts(db.execute('SELECT q.source,o.scan,o.seconds,o.quality FROM cohort_claims q '
            'JOIN observations o ON o.id=q.observation'), 'source')
        coverage={source:count/totals[source] for source,count in counts.items() if totals[source]}
        if selected_only:
            candidates={source for source,ratio in coverage.items() if .5<ratio<1.}
            for source in counts:
                if source not in candidates:
                    db.execute('DELETE FROM cohort_claims WHERE source=?',(source,))
            for source,target in db.execute('SELECT DISTINCT source,target FROM cohort_claims').fetchall():
                if target in candidates and coverage[target]>=coverage[source]:
                    db.execute('DELETE FROM cohort_claims WHERE source=? AND target=?',(source,target))
        if selected_only:
            active=[r[0] for r in db.execute('SELECT DISTINCT source FROM cohort_claims')]
            db.execute('DELETE FROM cohort_claims WHERE target IN (SELECT value FROM json_each(?))',
                       (encoded(active),))
        # Resolve competing selected fragments before replacing any memberships.
        targets={}
        for source,target in db.execute('SELECT DISTINCT source,target FROM cohort_claims'):
            targets.setdefault(target,[]).append(source)
        for target,sources in targets.items():
            blocked=set()
            for i,left in enumerate(sources):
                for right in sources[i+1:]:
                    if _cooccurs(db,left,right,config):
                        blocked.update((left,right))
            for source in blocked:
                db.execute('DELETE FROM cohort_claims WHERE source=? AND target=?',(source,target))
        if selected_only:
            # Filtering a conflicting target cannot leave a minority repair.
            for source,total in grouped_counts(db.execute('SELECT q.source,o.scan,o.seconds,o.quality '
                    'FROM cohort_claims q JOIN observations o ON o.id=q.observation'), 'source').items():
                if total<=totals[source]/2:
                    db.execute('DELETE FROM cohort_claims WHERE source=?',(source,))
        counts=grouped_counts(db.execute('SELECT q.source,o.scan,o.seconds,o.quality FROM cohort_claims q '
            'JOIN observations o ON o.id=q.observation'), 'source')
        sources=list(counts)
        proofs={source:dict(source=source,proposed_independent_support=counts[source],
            total_support=totals[source],frozen=True,selected_only=selected_only) for source in sources}
        for source in sources:
            db.execute("UPDATE candidates SET person=NULL,status='withheld',payload=json_set(payload,"
                "'$.prior_reason',json_extract(payload,'$.reason'),'$.reason','cohort_unresolved',"
                "'$.cohort_resolution',json(?)) WHERE policy=? AND person=?",(encoded(proofs[source]),policy,source))
            db.execute('DELETE FROM members WHERE person=?',(source,))
        reason='selected_cohort_identity' if selected_only else 'historical_cohort_identity'
        field='shared_reference' if selected_only else 'historical_reference'
        for row in db.execute('SELECT * FROM cohort_claims ORDER BY source,observation'):
            target=row['target']
            db.execute('INSERT OR IGNORE INTO people VALUES (?,?,?,?,?,?)',history.people[target])
            proof=proofs[row['source']]
            db.execute("UPDATE candidates SET person=?,status='pending',payload=json_set(payload,'$.reason',?,"
                "'$."+field+"',json(?),'$.cohort_resolution',json(?)) WHERE id=?",
                (target,reason,row['proof'],encoded(proof),row['candidate']))
        resolve_pending(db,policy,config)
        for observation in insufficient(pending=True):
            db.execute("UPDATE candidates SET person=NULL,status='withheld',payload=json_set(payload,"
                "'$.reason','cohort_insufficient_support') WHERE status='pending' AND id IN "
                '(SELECT candidate FROM cohort_claims WHERE observation=?)',
                (observation,))
        db.execute("UPDATE candidates SET status='accepted' WHERE policy=? AND status='pending' "
                   "AND json_extract(payload,'$.reason')=?",(policy,reason))
        db.execute("INSERT OR IGNORE INTO members SELECT json_extract(payload,'$.observation'),person "
            "FROM candidates WHERE policy=? AND status='accepted' AND json_extract(payload,'$.reason')=?",(policy,reason))
        for target, in db.execute("SELECT DISTINCT person FROM candidates WHERE policy=? AND status='accepted' "
                "AND json_extract(payload,'$.reason')=?",(policy,reason)):
            bindings[target]=target
    finally:
        db.execute('DROP TABLE cohort_claims')


def selected(database,config,early,global_state):
    from .historical_references import History
    from .reference_usage import ReferenceUsage
    db=database.connection
    first=db.execute('SELECT scope,namespace FROM scans LIMIT 1').fetchone()
    if first is None or not config.data['clustering']['anchor_view_matching']:
        return
    usage=ReferenceUsage(config.data['clustering']['max_reference_bytes'])
    groups=_groups(db,*first,config.data['clustering'],usage=usage)
    views,neighbours=global_state or ((),set())
    for view in views:
        usage.add(view.examples,kind='Selected strict cohort')
    if config.data['clustering']['strict_appearance_matching']:
        attached=whole_views(db,config,groups,views,strict_appearance=True)
        if attached:
            views=tuple(v for v in views if v.group.identifier not in attached)
            groups=_groups(db,*first,config.data['clustering'],usage=usage)
    owned={v.group.identifier for v in views}
    # A supported local core may be too small to train a view of its own.
    # Resolve it before the partial-cohort rule requiring an unmatched remainder.
    from .historical_references import rank
    from .anchor_views import _competing_anchor
    frozen=History((),tuple(views)+tuple(early),{},'frozen-selected-whole-views')
    proposals=[]
    for group in groups:
        if group.identifier in owned:
            continue
        rows=db.execute('SELECT id,geometry,dimensions,embedding FROM observations WHERE id IN '
            '(SELECT value FROM json_each(?))',(encoded(group.sample_ids),)).fetchall()
        result=rank(rows,replace(frozen,views=tuple(v for v in frozen.views
            if v.group.identifier!=group.identifier)),config) if rows else None
        if result is None:
            continue
        target,proof=result
        support=db.execute('SELECT COUNT(*) FROM (SELECT o.scan,o.seconds FROM members m '
            'JOIN observations o ON o.id=m.observation WHERE m.person=? GROUP BY o.scan,o.seconds)',
            (group.identifier,)).fetchone()[0]
        rivals=[g for g in groups if g.identifier!=group.identifier]
        if (support<config.data['clustering']['view_attach_min_support'] or _cooccurs(db,group.identifier,target,config)
            or any(_competing_anchor(rivals,target,vector(row),proof['mean_score'],proof['ambiguity_margin']) for row in rows)):
            continue
        proposals.append((group.identifier,target,proof))
    blocked=set()
    for i,(left,target,_) in enumerate(proposals):
        for right,other,_ in proposals[i+1:]:
            if target==other and _cooccurs(db,left,right,config):
                blocked.update((left,right))
    sources={source for source,_,_ in proposals}
    for source,target,proof in proposals:
        if source in blocked or target in sources:
            continue
        db.execute('UPDATE members SET person=? WHERE person=?',(target,source))
        db.execute("UPDATE candidates SET person=?,payload=json_set(payload,'$.prior_reason',"
            "json_extract(payload,'$.reason'),'$.reason','selected_view_identity','$.shared_reference',json(?)) "
            'WHERE person=?',(target,encoded(proof),source))
    if proposals:
        groups=_groups(db,*first,config.data['clustering'],usage=usage)
    source_ids={g.identifier for g in groups if g.identifier in neighbours or
        (g.identifier not in owned and len(g.samples)>max(config.data['clustering']['view_attach_min_support'],1))}
    for view in early:
        usage.add(view.group.samples,kind='Selected verified cohort')
    people={row['id']:tuple(row) for row in db.execute('SELECT * FROM people')}
    recover(db,groups,History(tuple(groups),tuple(views)+tuple(early),people,'frozen-selected-cohorts'),
            config,{},selected_only=True,source_ids=source_ids)


def whole_views(db,config,groups,views,*,strict_appearance=False):
    """A wholly contained query stays attachment-only in a joint replay too.

    Reciprocal proposals and competing/co-present sources stay unresolved here;
    the independently frozen merger plan handles compatible strict components.
    """
    from .historical_references import History,rank
    from .anchor_views import _competing_anchor
    proposals=[]
    for group in groups:
        frozen=History((),tuple(v for v in views if v.group.identifier!=group.identifier),{},'frozen-selected-whole-views')
        samples=db.execute('SELECT id,geometry,dimensions,embedding FROM observations WHERE id IN '
            '(SELECT value FROM json_each(?))',(encoded(group.sample_ids),)).fetchall()
        match=rank(samples,frozen,config,strict_appearance=strict_appearance) if samples else None
        if match is None:continue
        target,proof=match
        if _cooccurs(db,group.identifier,target,config):continue
        support=db.execute('SELECT COUNT(*) FROM (SELECT o.scan,o.seconds FROM members m JOIN observations o ON o.id=m.observation '
            'WHERE m.person=? GROUP BY o.scan,o.seconds)',(group.identifier,)).fetchone()[0]
        if support<config.data['clustering']['view_attach_min_support']:continue
        rivals=[g for g in groups if g.identifier!=group.identifier]
        valid=True
        for row in db.execute('SELECT o.* FROM members m JOIN observations o ON o.id=m.observation WHERE m.person=? ORDER BY o.id',
                              (group.identifier,)):
            result=rank([row],frozen,config,strict_appearance=strict_appearance)
            if result is None or result[0]!=target or _competing_anchor(rivals,target,vector(row),
                result[1]['mean_score'],result[1]['ambiguity_margin']):
                valid=False;break
        if valid:proposals.append((group.identifier,target,proof))
    sources={s for s,_,_ in proposals};blocked=set()
    for i,(left,target,_) in enumerate(proposals):
        for right,other,_ in proposals[i+1:]:
            if target==other and _cooccurs(db,left,right,config):blocked.update((left,right))
    attached=set()
    for source,target,proof in proposals:
        if source in blocked or target in sources:continue
        db.execute('UPDATE members SET person=? WHERE person=?',(target,source))
        db.execute("UPDATE candidates SET person=?,payload=json_set(payload,'$.prior_reason',json_extract(payload,'$.reason'),"
            "'$.reason','selected_view_identity','$.shared_reference',json(?)) WHERE person=?",
            (target,encoded(proof),source));attached.add(source)
    return attached
