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
            totals[group.identifier]=db.execute('SELECT COUNT(*) FROM (SELECT o.scan,o.seconds FROM members m '
                'JOIN observations o ON o.id=m.observation WHERE m.person=? GROUP BY o.scan,o.seconds)',
                (group.identifier,)).fetchone()[0]
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
                    if any(other!=group.identifier and mapped==target and _cooccurs(db,group.identifier,other)
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
            return db.execute('SELECT source,target,mode,medoids,COUNT(*) FROM ('
                'SELECT q.source,q.target,q.mode,q.medoids,o.scan,o.seconds FROM cohort_claims q '
                'JOIN observations o ON o.id=q.observation JOIN candidates c ON c.id=q.candidate WHERE 1=1'+state+
                ' GROUP BY q.source,q.target,q.mode,q.medoids,o.scan,o.seconds) GROUP BY source,target,mode,medoids '
                "HAVING mode='view' AND COUNT(*)<?",(minimum,)).fetchall()
        for source,target,mode,medoids,_ in insufficient():
            db.execute('DELETE FROM cohort_claims WHERE source=? AND target=? AND mode=? AND medoids=?',
                       (source,target,mode,medoids))
        counts=dict(db.execute('SELECT source,COUNT(*) FROM (SELECT q.source,o.scan,o.seconds FROM cohort_claims q '
            'JOIN observations o ON o.id=q.observation GROUP BY q.source,o.scan,o.seconds) GROUP BY source'))
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
                    if _cooccurs(db,left,right):
                        blocked.update((left,right))
            for source in blocked:
                db.execute('DELETE FROM cohort_claims WHERE source=? AND target=?',(source,target))
        if selected_only:
            # Filtering a conflicting target cannot leave a minority repair.
            for source,total in db.execute('SELECT source,COUNT(*) FROM (SELECT q.source,o.scan,o.seconds '
                    'FROM cohort_claims q JOIN observations o ON o.id=q.observation '
                    'GROUP BY q.source,o.scan,o.seconds) GROUP BY source').fetchall():
                if total<=totals[source]/2:
                    db.execute('DELETE FROM cohort_claims WHERE source=?',(source,))
        counts=dict(db.execute('SELECT source,COUNT(*) FROM (SELECT q.source,o.scan,o.seconds FROM cohort_claims q '
            'JOIN observations o ON o.id=q.observation GROUP BY q.source,o.scan,o.seconds) GROUP BY source'))
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
        for source,target,mode,medoids,_ in insufficient(pending=True):
            db.execute("UPDATE candidates SET person=NULL,status='withheld',payload=json_set(payload,"
                "'$.reason','cohort_insufficient_support') WHERE status='pending' AND id IN "
                '(SELECT candidate FROM cohort_claims WHERE source=? AND target=? AND mode=? AND medoids=?)',
                (source,target,mode,medoids))
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
    owned={v.group.identifier for v in views}
    source_ids={g.identifier for g in groups if g.identifier in neighbours or
        (g.identifier not in owned and len(g.samples)>max(config.data['clustering']['view_attach_min_support'],1))}
    for view in early:
        usage.add(view.group.samples,kind='Selected verified cohort')
    people={row['id']:tuple(row) for row in db.execute('SELECT * FROM people')}
    recover(db,groups,History(tuple(groups),tuple(views)+tuple(early),people,'frozen-selected-cohorts'),
            config,{},selected_only=True,source_ids=source_ids)
