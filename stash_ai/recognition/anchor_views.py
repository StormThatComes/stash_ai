"""Conservative same-scene anchor subclusters for withheld *anchor* faces.

Strict core members seed immutable local appearance medoids. A second pass may
attach previously withheld anchors only; it never creates or merges identities
or trains on its own attachments. Time never limits matches within a scene.
"""
from dataclasses import dataclass

from .clustering import Group, _groups, co_presence_allows, matching_policy, vector
from .database import encoded
from .evidence_cache import current as evidence_for
from .parallel_scores import contributions, scored_rows
from .pose_policy import anchor_sql
from .progress import tracking
from .reference_usage import ReferenceUsage
from .replay_addresses import address
from .view_prototypes import pose_view, pose_matches, add_cohort, coverage_threshold, supported_cohorts


@dataclass(frozen=True)
class AnchorView:
    scene_scan: str
    group: Group
    pose: str
    medoid: str
    support_ids: tuple
    support_count: int


def _confirmed(db, policy, grouping, usage):
    """Build local complete-link subclusters from *strict accepted* anchors.

    One scene/person at a time bounds scratch memory. Broad and narrow cohorts
    share the global bank's admission and priority rules. All supported reserve
    centres remain usable within the explicit memory budget. Duplicate cue times
    never vote twice; contextual attachments cannot seed strict views.
    """
    anchor = anchor_sql('o', grouping)
    rows = db.execute(
        'SELECT c.person,o.id,o.scan,o.seconds,o.geometry,o.dimensions,o.embedding,o.quality '
        f'FROM candidates c JOIN observations o ON o.id={address(db, "c")} '
        'JOIN members m ON m.observation=o.id AND m.person=c.person '
        'JOIN scans s ON s.id=o.scan '
        "WHERE c.policy=? AND c.status='accepted' AND "
        "json_extract(c.payload,'$.reason') IN ('known_identity','candidate_group') "
        f'AND c.person IS NOT NULL AND o.rejected IS NULL AND {anchor} '
        'ORDER BY CAST(s.scene AS INTEGER),s.scene,c.person,o.cue,o.seconds,o.geometry,o.embedding,o.id',
        (policy,))
    result, key, slots, cover, scratch = [], None, [], [], 0
    minimum = max(grouping['view_min_support'], grouping['same_scene_min_support'])
    cohesion = grouping['view_cohesion_threshold']

    def emit():
        if key is None:
            return
        for slot in supported_cohorts(slots,cover,minimum,grouping['max_view_prototypes_per_person'],complete=True):
            sample, medoid = slot.medoid()
            usage.add((sample,), kind='Anchor view')
            result.append(AnchorView(
                key[0], Group(key[1], (sample,), True, (medoid,)), slot.pose,
                medoid, tuple(slot.observations), len(slot.cues)))
    with tracking(rows, 'Build anchor view subclusters', unit='observation') as tracked:
        for row in tracked:
            rowkey = row['scan'], row['person']
            if rowkey != key:
                emit()
                key, slots, cover, scratch = rowkey, [], [], 0
            query = vector(row)
            scratch+=add_cohort(slots,row,query,cohesion,complete=True)
            scratch+=add_cohort(cover,row,query,coverage_threshold(grouping),complete=True)
            if usage.bytes+scratch>usage.limit:
                raise ValueError('Local anchor view evidence exceeds clustering.max_reference_bytes.')
    emit()
    if grouping['context_view_enrichment']:
        _enrich_from_validated_context(db, policy, grouping, usage, result)
    return result


def _enrich_from_validated_context(db, policy, grouping, usage, confirmed):
    """Frozen strict medoids validate contextual cues; only corroborated cohorts
    can add *additional* view medoids. Context samples never seed one another.
    """
    import numpy as np
    strict = {}
    for view in tuple(confirmed):
        strict.setdefault((view.scene_scan, view.group.identifier, view.pose), []).append(view)
    if not strict:
        return
    accepted = db.execute(
        'SELECT c.person,o.* FROM candidates c '
        f'JOIN observations o ON o.id={address(db, "c")} '
        'JOIN members m ON m.observation=o.id AND m.person=c.person '
        "WHERE c.policy=? AND c.status='accepted' "
        "AND json_extract(c.payload,'$.reason')='same_scene_identity' "
        'AND o.rejected IS NULL '
        'ORDER BY o.scan,c.person,o.cue,o.seconds,o.id', (policy,))
    additions = {}
    for row in accepted:
        pose = pose_view(row['geometry'])
        key = (row['scan'], row['person'], pose)
        if key not in strict:
            continue
        sample = vector(row)
        for core in strict[key]:
            if float(np.dot(sample, core.group.samples[0])) < grouping['view_match_threshold']:
                continue
            bucket = additions.setdefault((key, core.medoid), [])
            if (row['scan'], row['seconds']) in {(r['scan'], r['seconds']) for r,_ in bucket}:
                continue
            if all(float(np.dot(sample, other)) >= grouping['view_cohesion_threshold']
                   for _,other in bucket):
                bucket.append((row, sample))
            break
    # Two separately vetted contextual cues corroborate an already confirmed
    # strict core (whose own view_min_support was applied earlier).
    minimum = 2
    for (key, parent_medoid), bucket in sorted(additions.items(), key=lambda item: item[0]):
        scene, person, pose = key
        if len(bucket) < minimum:
            continue
        # Context medoids require at least two independently accepted cues;
        # their only authority is the immutable, strictly confirmed core.
        scores = [sum(float(np.dot(a,b)) for _,b in bucket) for _,a in bucket]
        selected = max(range(len(bucket)), key=lambda i:scores[i])
        row, medoid = bucket[selected]
        origin = next(v for v in strict[key] if v.medoid==parent_medoid)
        usage.add((medoid,), kind='Context corroborated view')
        confirmed.append(AnchorView(scene,Group(person,(medoid,),True),pose,
                         row['id'],tuple(r['id'] for r,_ in bucket),
                         origin.support_count + len(bucket)))


def _different_accepted_face(db, row, person, policy):
    """A second detection already assigned to the destination needs review."""
    return db.execute(
        'SELECT 1 FROM candidates c JOIN observations o '
        f'ON o.id={address(db, "c")} '
        "WHERE c.scan=? AND c.policy=? AND c.status='accepted' "
        'AND c.person=? AND o.cue=? AND o.id<>? LIMIT 1',
        (row['scan'], policy, person, row['cue'], row['id'])).fetchone() is not None


def _competing_anchor(groups, person, query, score, margin):
    """A sparse rival with no promoted view must not be invisible to ranking.

    Use the *maximum* scalar cosine against every retained rival anchor from
    this scene, not its group mean/minimum: a single convincing competitor is
    sufficient to veto an otherwise apparently unambiguous view attachment.
    """
    import numpy as np
    boundary = score - margin
    for rival in groups:
        if rival.identifier != person and any(
                float(np.dot(sample, query)) >= boundary
                for sample in rival.samples):
            return True
    return False


def _probe(db, candidate, reason):
    db.execute(
        "UPDATE candidates SET payload=json_set(payload,'$.anchor_view_probe',?) "
        'WHERE id=?', (reason, candidate))


def apply_anchor_views(database, selected, config, *, progress=None):
    grouping, settings = config.data['clustering'], config.data['recognition']
    if not grouping['anchor_view_matching'] or not selected:
        return
    db, policy = database.connection, matching_policy(config)
    cache = evidence_for(db)
    first = db.execute('SELECT scope,namespace FROM scans WHERE id=?',
                       (selected[0]['scan'],)).fetchone()
    usage = ReferenceUsage(grouping['max_reference_bytes'], progress)
    # Account existing strict references before admitting view medoids.
    anchors = _groups(db, first['scope'], first['namespace'], grouping,
                      usage=usage, stage='anchor_views')
    anchor_by_id = {group.identifier: group for group in anchors}
    views = _confirmed(db, policy, grouping, usage)
    # Per-medoid provenance is stored with its own accepted observation;
    # the read-only report can count *actual* promoted local prototypes.
    for view in views:
        db.execute(
            "UPDATE candidates SET payload=json_set(payload,"
            "'$.anchor_view_seeded',coalesce(json_extract(payload,'$.anchor_view_seeded'),0)+1) "
            f'WHERE policy=? AND {address(db, "candidates")}=?',
            (policy, view.medoid))
    # Record why this conservative stage could not recover a withheld anchor.
    db.execute(
        "UPDATE candidates SET payload=json_set(payload,'$.anchor_view_probe',"
        "'no_confirmed_prototype') WHERE policy=? AND status='withheld' "
        "AND json_extract(payload,'$.reason') IN ('insufficient_match_or_cohesion','prior_attachment_query')",
        (policy,))
    if not views:
        return
    views_by_scan = {}
    for v in views:
        views_by_scan.setdefault(v.scene_scan, []).append(v)
    threshold = max(settings['match_threshold'], grouping['view_match_threshold'])
    margin = max(settings['ambiguity_margin'], grouping['view_ambiguity_margin'])
    # Freeze views for the entire matching pass. Attachments never train views.
    for item in selected:
        scan = item['scan']
        local = views_by_scan.get(scan, [])
        if not local:
            db.execute(
                "UPDATE candidates SET payload=json_set(payload,'$.anchor_view_probe','no_scene_prototype') "
                "WHERE scan=? AND policy=? AND status='withheld' "
                "AND json_extract(payload,'$.reason') IN ('insufficient_match_or_cohesion','prior_attachment_query')",
                (scan,policy))
            continue
        groups = [v.group for v in local]
        # All local accepted identity rivals participate in the final margin
        # check, even when they have too few cues to promote any own view.
        local_people = {r[0] for r in db.execute(
            "SELECT DISTINCT person FROM candidates WHERE scan=? AND policy=? "
            "AND status='accepted' AND person IS NOT NULL", (scan, policy))}
        rival_anchors = [anchor_by_id[p] for p in sorted(local_people)
                         if p in anchor_by_id]
        by_group = {id(v.group): v for v in local}
        source = (f'FROM candidates c JOIN observations o ON o.id={address(db, "c")} '
                  "WHERE c.scan=? AND c.policy=? AND c.status='withheld' "
                  "AND json_extract(c.payload,'$.reason') IN ('insufficient_match_or_cohesion','prior_attachment_query') "
                  f'AND o.rejected IS NULL AND {anchor_sql("o", grouping)}')
        rows = db.execute('SELECT o.*,c.id AS candidate ' + source +
                          ' ORDER BY o.cue,o.geometry,o.embedding,o.id', (scan, policy))
        total = db.execute('SELECT COUNT(*) ' + source, (scan, policy)).fetchone()[0]
        with tracking(scored_rows(rows, groups, threshold-margin),
                      'Match withheld anchor views', unit='observation', total=total) as tracked:
            for row, frozen in tracked:
                _probe(db, row['candidate'], 'below_match_or_cohesion')
                query = cache.vector(row) if cache is not None else vector(row)
                by_person = {}
                saw_cohesion, saw_pose = False, False
                for group, score, lowest in contributions(
                        groups, query, frozen, len(groups), threshold-margin,
                        exact=(lambda g, q: cache.scores(g, q, row['id'])) if cache is not None else None):
                    if lowest < grouping['view_cohesion_threshold']:
                        saw_cohesion = True
                        continue
                    view = by_group[id(group)]
                    if not pose_matches(row['geometry'],view.pose):
                        saw_pose = True
                        continue
                    old = by_person.get(group.identifier)
                    if old is None or score > old[0]:
                        by_person[group.identifier] = (score, group, view)
                ranked = sorted(by_person.items(), key=lambda item: (-item[1][0], item[0]))
                if not ranked:
                    if saw_pose:
                        _probe(db, row['candidate'], 'pose_mismatch')
                    elif saw_cohesion:
                        _probe(db, row['candidate'], 'below_cohesion_threshold')
                    continue
                if ranked[0][1][0] < threshold:
                    _probe(db, row['candidate'], 'below_match_threshold')
                    continue
                _probe(db, row['candidate'], 'ambiguous_identity')
                if len(ranked)>1 and ranked[0][1][0]-ranked[1][1][0] < margin:
                    continue
                person, (score, group, view) = ranked[0]
                if _competing_anchor(rival_anchors, person, query, score, margin):
                    _probe(db, row['candidate'], 'competing_identity')
                    continue
                if _different_accepted_face(db, row, person, policy):
                    _probe(db, row['candidate'], 'co_present_identity')
                    continue
                if not co_presence_allows(
                        db, row, group, groups, query, config,
                        peer_threshold=threshold,
                        cohesion_threshold=grouping['view_cohesion_threshold']):
                    _probe(db, row['candidate'], 'co_presence')
                    continue
                proof = {
                    'format': 1, 'scene_scan': scan, 'medoid': view.medoid,
                    'anchor_support': view.support_count, 'support_ids': view.support_ids,
                    'match_score': score, 'threshold': threshold,
                    'ambiguity_margin': margin}
                db.execute("UPDATE candidates SET person=?,status='pending',"
                           "payload=json_set(payload,'$.reason','anchor_view_pending',"
                           "'$.anchor_view',json(?),'$.anchor_view_probe','pending_support') WHERE id=?",
                           (person, encoded(proof), row['candidate']))
    from .decision_resolution import resolve_pending
    resolve_pending(db, policy, config)
    anchors.clear()
    from .query_support import eligible
    source = (f'FROM candidates c JOIN observations o ON o.id={address(db, "c")} '
              "WHERE c.policy=? AND c.status='pending' "
              "AND json_extract(c.payload,'$.reason')='anchor_view_pending'")
    rows=db.execute("SELECT o.*,c.person,json_extract(c.payload,'$.anchor_view.medoid') AS medoid "+source+
                    ' ORDER BY o.embedding,o.geometry,o.id',(policy,))
    supported=eligible(rows,config,key=lambda r:r['person'],prototype=lambda r:r['medoid'])
    with db:
        db.execute("UPDATE candidates SET status='accepted',payload=json_set(payload,'$.reason','anchor_view_identity') "
            "WHERE policy=? AND status='pending' AND json_extract(payload,'$.reason')='anchor_view_pending' "
            f"AND {address(db, 'candidates')} IN (SELECT value FROM json_each(?))",(policy,encoded(sorted(supported))))
        db.execute(
            "UPDATE candidates SET person=NULL,status='withheld',"
            "payload=json_set(payload,'$.reason','anchor_view_insufficient_support') "
            "WHERE policy=? AND status='pending' "
            "AND json_extract(payload,'$.reason')='anchor_view_pending'", (policy,))
        db.execute(
            "UPDATE candidates SET payload=json_set(payload,'$.anchor_view_probe','accepted') "
            "WHERE policy=? AND json_extract(payload,'$.reason')='anchor_view_identity'",
            (policy,))
        db.execute(
            "UPDATE candidates SET payload=json_set(payload,'$.anchor_view_probe','insufficient_support') "
            "WHERE policy=? AND json_extract(payload,'$.reason')='anchor_view_insufficient_support'",
            (policy,))
        db.execute(f'INSERT OR IGNORE INTO members(observation,person) '
                   f'SELECT {address(db, "c")},c.person FROM candidates c '
                   "WHERE c.policy=? AND c.status='accepted' "
                   "AND json_extract(c.payload,'$.reason')='anchor_view_identity'", (policy,))
