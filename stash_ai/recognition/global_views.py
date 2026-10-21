"""Frozen strict appearance cohorts shared across selected scenes.

View-only members never train the catalogue. Identity consolidation requires
complete frozen ambiguity components from independently supported cohorts; the
frozen nomination graph is not recomputed after a merge, preventing chaining.
"""
from dataclasses import dataclass, replace
import time

from .clustering import Group, _groups, co_presence_allows, matching_policy, vector
from .database import encoded
from .identity_merges import canonical
from .pose_policy import anchor_sql
from .reference_roles import strict_sql
from .reference_usage import ReferenceUsage
from .replay_addresses import address
from .progress import tracking
from .scoring import comparisons
from .view_prototypes import _Cluster, pose_view, pose_matches


@dataclass(frozen=True)
class IdentityView:
    group: Group
    pose: str
    medoid: str
    support: tuple
    support_count: int
    examples: tuple
    authority: str = 'strict'


def catalogue(db, config, scope, namespace, *, usage, policy=None, excluded_scenes=(), scan_policy=None):
    grouping = config.data['clustering']
    condition = '' if policy is None else 'AND c.policy=? AND c.status=\'accepted\' AND c.person=m.person '
    from .reference_compatibility import fallback_sql
    scan_condition = ('' if scan_policy is None else
        'AND (s.policy_sha=? OR '+fallback_sql(db,config,namespace)+') ')
    join = '' if policy is None else f'JOIN candidates c ON {address(db, "c")}=o.id '
    parameters = [scope, namespace, encoded(list(excluded_scenes))]
    if policy is not None:
        parameters.append(policy)
    if scan_policy is not None:
        parameters.append(scan_policy)
    rows = db.execute(
        'SELECT m.person,o.id,o.scan,o.seconds,o.geometry,o.dimensions,o.embedding,o.quality '
        'FROM members m JOIN people p ON p.id=m.person JOIN observations o ON o.id=m.observation '
        'JOIN scans s ON s.id=o.scan ' + join +
        "WHERE p.scope=? AND p.namespace=? AND s.scene NOT IN (SELECT value FROM json_each(?)) "
        "AND s.scope=p.scope AND s.namespace=p.namespace AND s.status='complete' AND o.rejected IS NULL "
        'AND NOT EXISTS (SELECT 1 FROM identity_merges WHERE source=p.id) '
        'AND NOT EXISTS (SELECT 1 FROM scans newer WHERE newer.scope=s.scope AND newer.namespace=s.namespace '
        'AND newer.scene=s.scene AND newer.rowid>s.rowid) AND ' + anchor_sql('o', grouping) +
        ' AND ' + strict_sql() + ' ' + condition + scan_condition +
        'ORDER BY m.person,CAST(s.scene AS INTEGER),s.scene,o.cue,o.seconds,o.geometry,o.embedding,o.id', parameters)
    result, owner, slots = [], None, []

    def emit():
        if owner is None:
            return
        eligible = [slot for slot in slots if len(slot.cues) >= grouping['view_min_support']]
        eligible.sort(key=lambda slot: (-len(slot.cues), slot.pose, slot.first))
        for slot in eligible[:grouping['max_view_prototypes_per_person']]:
            sample, medoid = slot.medoid()
            usage.add(slot.examples, kind='Global view evidence')
            result.append(IdentityView(Group(owner, (sample,), True, (medoid,)), slot.pose,
                medoid, tuple(slot.observations), len(slot.cues), tuple(slot.examples)))
    with tracking(rows, 'Build global strict identity views', unit='observation') as tracked:
        for row in tracked:
            if owner != row['person']:
                emit()
                owner, slots = row['person'], []
            sample, pose = vector(row), pose_view(row['geometry'])
            slot = next((slot for slot in slots if slot.pose == pose and
                         slot.accepts(sample, grouping['view_cohesion_threshold'])), None)
            if slot is None:
                if len(slots) >= 32:
                    continue
                slot = _Cluster(pose, row['id'])
                slots.append(slot)
            slot.add(row, sample)
    emit()
    return tuple(result)


def rank_person(ranked, entry):
    previous = next((item for item in ranked if item[1] == entry[1]), None)
    if previous is not None:
        if previous[0] >= entry[0]:
            return
        ranked.remove(previous)
    ranked.append(entry)
    ranked.sort(key=lambda item: (-item[0], item[1]))
    del ranked[2:]


def unambiguous(ranked, threshold, margin):
    return bool(ranked and ranked[0][0] >= threshold and
                (len(ranked) == 1 or ranked[0][0]-ranked[1][0] >= margin))


def _cooccurs(db, left, right):
    return db.execute('SELECT 1 FROM members a JOIN observations x ON x.id=a.observation '
        'JOIN observations y ON y.scan=x.scan AND y.cue=x.cue '
        'JOIN members b ON b.observation=y.id WHERE a.person=? AND b.person=? '
        'AND x.id<>y.id AND x.rejected IS NULL AND y.rejected IS NULL LIMIT 1',
        (left, right)).fetchone() is not None


def cohort_evidence(db, left, right, config):
    """The same frozen cohort checks serve consolidation and read-only reports."""
    import numpy as np
    grouping,settings=config.data['clustering'],config.data['recognition']
    threshold=max(settings['match_threshold'],grouping['view_match_threshold'])
    score=float(np.dot(left.group.samples[0],right.group.samples[0]))
    cross=[float(np.dot(a,b)) for a in left.examples for b in right.examples]
    low,mean=min(cross),sum(cross)/len(cross)
    copresent=_cooccurs(db,left.group.identifier,right.group.identifier)
    blockers=[]
    if score<threshold:
        blockers.append('view_medoid_below_match_threshold')
    if min(left.support_count,right.support_count)<settings['min_merge_support']:
        blockers.append('view_independent_merge_support_insufficient')
    if low<grouping['view_cohesion_threshold']:
        blockers.append('view_cross_sample_below_cohesion_threshold')
    if mean<threshold:
        blockers.append('view_cross_mean_below_match_threshold')
    if copresent and low<max(threshold,settings['cooccurrence_merge_threshold']):
        blockers.append('view_copresence_requires_reflection_strength')
    return dict(cooccurring=copresent,medoids=[left.medoid,right.medoid],
        support=[left.support_count,right.support_count],support_ids=[left.support,right.support],
        minimum_cross_similarity=low,mean_cross_similarity=mean,match_score=score,
        threshold=threshold,blockers=blockers)


def merge_views(database, views, config, policy):
    """Consolidate complete frozen components, never neighbour chains.

    Weak neighbours join the ambiguity component too. Every identity pair must
    have independently supported strong cohort evidence before any member moves.
    Union-find and strong degrees use O(people) memory, without an edge table.
    """
    import numpy as np
    from .config import digest
    db=database.connection
    grouping,settings=config.data['clustering'],config.data['recognition']
    threshold=max(settings['match_threshold'],grouping['view_match_threshold'])
    margin=max(settings['ambiguity_margin'],grouping['view_ambiguity_margin'])
    groups=[v.group for v in views]
    by_group={id(v.group):v for v in views}
    owned={}
    for view in views:
        owned.setdefault(view.group.identifier,[]).append(view)
    parents={person:person for person in owned}
    degrees={person:0 for person in owned}
    def root(person):
        while parents[person]!=person:
            parents[person]=parents[parents[person]]
            person=parents[person]
        return person
    def join(left,right):
        a,b=root(left),root(right)
        if a!=b:
            parents[max(a,b)]=min(a,b)
    def witness(left,right,score):
        if score<threshold or min(left.support_count,right.support_count)<settings['min_merge_support']:
            return None
        proof=cohort_evidence(db,left,right,config)
        return proof if not proof['blockers'] else None
    with tracking(sorted(owned),'Validate frozen identity components',unit='person') as tracked:
        for person in tracked:
            candidates={}
            for view in owned[person]:
                for other,score,_ in comparisons(groups,view.group.samples[0],threshold-margin):
                    if other.identifier<=person:
                        continue
                    # One bounded row per other identity, irrespective of views.
                    if other.identifier not in candidates:
                        candidates[other.identifier]=None
                    if candidates[other.identifier] is None:
                        candidates[other.identifier]=witness(view,by_group[id(other)],score)
            for other,proof in candidates.items():
                join(person,other)
                if proof is not None:
                    degrees[person]+=1;degrees[other]+=1
    components={}
    for person in owned:
        components.setdefault(root(person),[]).append(person)
    order={r['id']:(r['created'],r['id']) for r in db.execute('SELECT id,created FROM people')}
    for people in components.values():
        if len(people)<2 or any(degrees[p]!=len(people)-1 for p in people):
            continue
        target,*sources=sorted(people,key=lambda p:order[p])
        context=digest(sorted(people))
        # Obtain all proofs from the original catalogue before membership moves.
        proofs=[]
        for source in sources:
            proof=next((proof for left in owned[target] for right in owned[source]
                if (proof:=witness(left,right,float(np.dot(left.group.samples[0],right.group.samples[0])))) is not None),None)
            if proof is None:
                break
            proofs.append((source,dict(proof,format=2,frozen=True,reciprocal=True,complete_component=True,
                component_size=len(people),component_context=context,scan=None,policy=policy,
                threshold=threshold,ambiguity_margin=margin)))
        if len(proofs)!=len(sources):
            continue
        for source,proof in proofs:
            db.execute('INSERT INTO identity_merges VALUES (?,?,?,?,?)',(source,target,time.time(),encoded(proof),'local'))
            db.execute('UPDATE members SET person=? WHERE person=?',(target,source))
            db.execute("UPDATE candidates SET payload=json_set(payload,'$.merged_from',person,"
                "'$.global_view_merge',json(?)),person=? WHERE person=?",(encoded(proof),target,source))
    return tuple(replace(view,group=Group(canonical(db,view.group.identifier),view.group.samples,True,
        view.group.sample_ids)) for view in views)


def recover(database, selected, config, views, anchors, policy):
    from .anchor_views import _competing_anchor
    db = database.connection
    grouping, settings = config.data['clustering'], config.data['recognition']
    threshold = max(settings['match_threshold'],grouping['view_match_threshold'])
    margin = max(settings['ambiguity_margin'],grouping['view_ambiguity_margin'])
    groups = [view.group for view in views]
    by_group = {id(view.group):view for view in views}
    for item in selected:
        rows = db.execute('SELECT o.*,c.id AS candidate FROM candidates c JOIN observations o '
            f'ON o.id={address(db,"c")} WHERE c.scan=? AND c.policy=? AND c.status=\'withheld\' '
            "AND json_extract(c.payload,'$.reason') IN ('candidate_group','insufficient_match_or_cohesion',"
            "'anchor_view_insufficient_support') AND o.rejected IS NULL AND " + anchor_sql('o',grouping) +
            ' ORDER BY o.cue,o.geometry,o.embedding,o.id',(item['scan'],policy))
        with tracking(rows,'Recover global anchor views',unit='observation') as tracked:
            for row in tracked:
                query, ranked = vector(row), []
                for group, score, low in comparisons(groups,query,threshold-margin):
                    view = by_group[id(group)]
                    if low < grouping['view_cohesion_threshold'] or not pose_matches(row['geometry'],view.pose):
                        continue
                    rank_person(ranked,(score,group.identifier,view))
                if not unambiguous(ranked,threshold,margin):
                    continue
                score, person, view = ranked[0]
                if _competing_anchor(anchors,person,query,score,margin):
                    continue
                if db.execute('SELECT 1 FROM members m JOIN observations o ON o.id=m.observation '
                    'WHERE m.person=? AND o.scan=? AND o.cue=? AND o.id<>? LIMIT 1',
                    (person,row['scan'],row['cue'],row['id'])).fetchone():
                    continue
                if not co_presence_allows(db,row,view.group,groups,query,config,
                        peer_threshold=threshold,cohesion_threshold=grouping['view_cohesion_threshold']):
                    continue
                proof=dict(format=1,frozen=True,medoid=view.medoid,support=view.support_count,
                           support_ids=view.support,score=score,threshold=threshold,ambiguity_margin=margin)
                db.execute("UPDATE candidates SET person=?,status='pending',payload=json_set(payload,"
                    "'$.reason','global_view_pending','$.global_view',json(?)) WHERE id=?",
                    (person,encoded(proof),row['candidate']))
    from .decision_resolution import resolve_pending
    resolve_pending(db, policy, config)
    votes=db.execute('SELECT person,medoid,COUNT(*) FROM (SELECT c.person,'
        "json_extract(c.payload,'$.global_view.medoid') AS medoid,o.scan,o.seconds "
        f'FROM candidates c JOIN observations o ON o.id={address(db,"c")} '
        "WHERE c.policy=? AND c.status='pending' AND json_extract(c.payload,'$.reason')='global_view_pending' "
        'GROUP BY c.person,medoid,o.scan,o.seconds) GROUP BY person,medoid',(policy,)).fetchall()
    for person,medoid,total in votes:
        if total >= grouping['view_attach_min_support']:
            db.execute("UPDATE candidates SET status='accepted',payload=json_set(payload,'$.reason','global_view_identity') "
                "WHERE policy=? AND person=? AND status='pending' AND json_extract(payload,'$.reason')='global_view_pending' "
                "AND json_extract(payload,'$.global_view.medoid')=?",(policy,person,medoid))
    db.execute(f'INSERT OR IGNORE INTO members SELECT {address(db,"c")},c.person FROM candidates c '
        "WHERE c.policy=? AND c.status='accepted' AND json_extract(c.payload,'$.reason')='global_view_identity'",(policy,))
    db.execute("UPDATE candidates SET person=NULL,status='withheld',payload=json_set(payload,"
        "'$.reason','global_view_insufficient_support') WHERE policy=? AND status='pending' "
        "AND json_extract(payload,'$.reason')='global_view_pending'",(policy,))


def apply_global_views(database, selected, config, *, progress=None):
    if not config.data['clustering']['anchor_view_matching'] or not selected:
        return
    db,policy = database.connection,matching_policy(config)
    first = db.execute('SELECT scope,namespace FROM scans WHERE id=?',(selected[0]['scan'],)).fetchone()
    usage = ReferenceUsage(config.data['clustering']['max_reference_bytes'],progress)
    anchors = _groups(db,first['scope'],first['namespace'],config.data['clustering'],usage=usage,stage='global_views')
    views = catalogue(db,config,first['scope'],first['namespace'],usage=usage,policy=policy)
    if not views:
        return
    views = merge_views(database,views,config,policy)
    anchors = [Group(canonical(db,group.identifier),group.samples,True,group.sample_ids) for group in anchors]
    recover(database,selected,config,views,anchors,policy)
    usage.emit(force=True)
