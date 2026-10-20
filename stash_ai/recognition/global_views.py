"""Frozen strict appearance cohorts shared across selected scenes.

View-only members never train the catalogue. Identity consolidation requires
reciprocal, unambiguous nominations from independently supported cohorts; the
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
from .view_prototypes import _Cluster, pose_view


@dataclass(frozen=True)
class IdentityView:
    group: Group
    pose: str
    medoid: str
    support: tuple
    support_count: int
    examples: tuple


def catalogue(db, config, scope, namespace, *, usage, policy=None, excluded_scenes=()):
    grouping = config.data['clustering']
    condition = '' if policy is None else 'AND c.policy=? AND c.status=\'accepted\' AND c.person=m.person '
    join = '' if policy is None else f'JOIN candidates c ON {address(db, "c")}=o.id '
    parameters = [scope, namespace, encoded(list(excluded_scenes))]
    if policy is not None:
        parameters.append(policy)
    rows = db.execute(
        'SELECT m.person,o.id,o.scan,o.seconds,o.geometry,o.dimensions,o.embedding '
        'FROM members m JOIN people p ON p.id=m.person JOIN observations o ON o.id=m.observation '
        'JOIN scans s ON s.id=o.scan ' + join +
        "WHERE p.scope=? AND p.namespace=? AND s.scene NOT IN (SELECT value FROM json_each(?)) "
        "AND s.scope=p.scope AND s.namespace=p.namespace AND s.status='complete' AND o.rejected IS NULL "
        'AND NOT EXISTS (SELECT 1 FROM identity_merges WHERE source=p.id) '
        'AND NOT EXISTS (SELECT 1 FROM scans newer WHERE newer.scope=s.scope AND newer.namespace=s.namespace '
        'AND newer.scene=s.scene AND newer.rowid>s.rowid) AND ' + anchor_sql('o', grouping) +
        ' AND ' + strict_sql() + ' ' + condition +
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


def merge_views(database, views, config, policy):
    """Only mutual best pairs from the original catalogue can consolidate."""
    import numpy as np
    db = database.connection
    grouping, settings = config.data['clustering'], config.data['recognition']
    threshold = max(settings['match_threshold'], grouping['view_match_threshold'])
    margin = max(settings['ambiguity_margin'], grouping['view_ambiguity_margin'])
    groups = [view.group for view in views]
    by_group = {id(view.group): view for view in views}
    nominations = {}
    with tracking(views, 'Nominate global identity view merges', unit='view') as tracked:
        for view in tracked:
            ranked = nominations.setdefault(view.group.identifier, [])
            for other, score, _ in comparisons(groups, view.group.samples[0], threshold-margin):
                if other.identifier == view.group.identifier:
                    continue
                rank_person(ranked, (score, other.identifier, view, by_group[id(other)]))
    order = {row['id']: (row['created'],row['id']) for row in db.execute('SELECT id,created FROM people')}
    consumed = set()
    for person in sorted(nominations, key=lambda person: order[person]):
        ranked = nominations[person]
        if person in consumed or not unambiguous(ranked, threshold, margin):
            continue
        score, other, left, right = ranked[0]
        rival = nominations.get(other, [])
        if (other in consumed or not unambiguous(rival, threshold, margin) or rival[0][1] != person):
            continue
        if min(left.support_count, right.support_count) < settings['min_merge_support']:
            continue
        cross = [float(np.dot(a,b)) for a in left.examples for b in right.examples]
        if min(cross) < grouping['view_cohesion_threshold'] or sum(cross)/len(cross) < threshold:
            continue
        cooccurring = _cooccurs(db, person, other)
        if cooccurring and min(cross) < max(threshold, settings['cooccurrence_merge_threshold']):
            continue
        target, source = sorted((person,other), key=lambda person: order[person])
        proof = dict(format=1, frozen=True, reciprocal=True, scan=None, policy=policy,
            cooccurring=cooccurring, medoids=[left.medoid,right.medoid],
            support=[left.support_count,right.support_count], support_ids=[left.support,right.support],
            minimum_cross_similarity=min(cross), mean_cross_similarity=sum(cross)/len(cross),
            threshold=threshold, ambiguity_margin=margin)
        db.execute('INSERT INTO identity_merges VALUES (?,?,?,?,?)',
                   (source,target,time.time(),encoded(proof),'local'))
        db.execute('UPDATE members SET person=? WHERE person=?',(target,source))
        db.execute("UPDATE candidates SET payload=json_set(payload,'$.merged_from',person,"
                   "'$.global_view_merge',json(?)),person=? WHERE person=?",(encoded(proof),target,source))
        consumed.update((person,other))
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
                    if low < grouping['view_cohesion_threshold'] or view.pose != pose_view(row['geometry']):
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
