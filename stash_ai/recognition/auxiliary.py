"""Attach pose-limited faces only to previously established anchor identities.

Attachments are not reference samples, new identities, or merge votes.
"""
import uuid

from .clustering import _groups, vector, co_presence_allows, matching_policy
from .database import encoded
from .evidence_cache import current as evidence_for
from .parallel_scores import scored_rows, contributions
from .pose_policy import anchor_sql,restricted,visibility_allows
from .progress import tracking
from .ranking import best_two
from .reference_usage import ReferenceUsage
from .view_prototypes import confirmed_views, pose_view, pose_matches
from .replay_addresses import address


def apply_auxiliary(database, selected, config, *, progress=None, reference_groups=None, reference_views=(),
                    reference_confirmations=()):
    grouping, settings = config.data['clustering'], config.data['recognition']
    if not restricted(grouping) or not selected:
        return
    db, policy = database.connection, matching_policy(config)
    cache = evidence_for(db)
    first = db.execute('SELECT scope,namespace FROM scans WHERE id=?', (selected[0]['scan'],)).fetchone()
    usage = ReferenceUsage(grouping['max_reference_bytes'], progress)
    if reference_groups is None:
        groups = _groups(db, first['scope'], first['namespace'], grouping, usage=usage, stage='auxiliary')
    else:
        groups = reference_groups
        usage.begin('auxiliary')
        for group in groups:
            usage.add(group.samples,people=1,kind='Shared strict')
    for view in reference_views:
        usage.add(view.group.samples,kind='Shared view')
    if grouping['view_prototypes_enabled'] and grouping['auxiliary_matching']:
        from .global_views import catalogue
        from .clustering import Group
        cohorts=catalogue(db,config,*first,usage=usage,policy=policy,minimum=1,complete=True)
        for view in reference_confirmations:
            usage.add(view.examples,kind='Historical confirmation evidence')
        groups=list(groups)+[Group(v.group.identifier,v.examples,True,v.support)
                            for v in tuple(cohorts)+tuple(reference_confirmations)]
    condition = 'NOT ('+anchor_sql('o',grouping)+')'
    for item in selected:
        scan = item['scan']
        rows = db.execute(
            f'SELECT o.* FROM observations o WHERE o.scan=? AND o.rejected IS NULL '
            f'AND {condition} ORDER BY o.cue,o.geometry,o.embedding,o.id', (scan,))
        total = db.execute(
            f'SELECT COUNT(*) FROM observations o WHERE o.scan=? AND o.rejected IS NULL '
            f'AND {condition}', (scan,)).fetchone()[0]
        with tracking(scored_rows(rows, groups, settings['match_threshold'] - settings['ambiguity_margin']),
                      'Attach auxiliary face observations', unit='observation', total=total) as tracked:
            for row, frozen in tracked:
                person, status, reason, scores = None, 'withheld', 'auxiliary_unmatched', []
                confirmation=None
                if not visibility_allows(row['quality'],grouping.get('min_auxiliary_visible_face_fraction')):
                    reason='auxiliary_visibility_insufficient'
                elif grouping['auxiliary_matching'] and groups:
                    query = cache.vector(row) if cache is not None else vector(row)
                    ranked = []
                    for group, mean, low in contributions(
                            groups, query, frozen, len(groups),
                            settings['match_threshold'] - settings['ambiguity_margin'],
                            exact=(lambda g, q: cache.scores(g, q, row['id'])) if cache is not None else None):
                        if low >= grouping['cohesion_threshold']:
                            from .global_views import rank_person
                            rank_person(ranked, (mean, group.identifier, group))
                    scores = [entry[0] for entry in ranked[:2]]
                    if ranked and ranked[0][0] >= settings['match_threshold']:
                        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < settings['ambiguity_margin']:
                            reason = 'auxiliary_ambiguous'
                        elif co_presence_allows(db, row, ranked[0][2], groups, query, config):
                            from .anchor_views import _competing_anchor
                            if not _competing_anchor(groups,ranked[0][1],query,ranked[0][0],settings['ambiguity_margin']):
                                person, status, reason = ranked[0][1], 'pending', 'auxiliary_identity'
                                confirmation=dict(format=1,frozen=True,identity=person,
                                    support_ids=ranked[0][2].sample_ids,score=ranked[0][0],
                                    match_threshold=settings['match_threshold'],cohesion_threshold=grouping['cohesion_threshold'])
                            else:
                                reason='auxiliary_ambiguous'
                        else:
                            reason = 'auxiliary_co_presence'
                elif not grouping['auxiliary_matching']:
                    reason = 'auxiliary_disabled'
                candidate = str(uuid.uuid5(uuid.UUID(row['id']), policy))
                db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?)',
                           (candidate, scan, policy, person, status,
                            encoded({'observation': row['id'], 'reason': reason, 'scores': scores,
                                     'strict_confirmation':confirmation,
                                     'policy': policy, 'representative_limit': grouping['max_samples_per_person']})))
    from .decision_resolution import resolve_pending
    resolve_pending(db, policy, config)
    # Duplicate detections at the same scene/time are only one support vote.
    source = (f'FROM candidates c JOIN observations o ON o.id={address(db, "c")} '
              "WHERE c.policy=? AND c.status='pending' "
              "AND json_extract(c.payload,'$.reason')='auxiliary_identity'")
    supporters = db.execute(
        f'SELECT person,COUNT(*) FROM (SELECT c.person AS person,o.scan,o.seconds {source} '
        'GROUP BY c.person,o.scan,o.seconds) GROUP BY person', (policy,)).fetchall()
    eligible = {person for person, count in supporters if count >= grouping['auxiliary_min_support']}
    with db:
        for person, _ in supporters:
            if person in eligible:
                db.execute("UPDATE candidates SET status='accepted' WHERE person=? AND policy=? "
                           "AND json_extract(payload,'$.reason')='auxiliary_identity'", (person, policy))
            else:
                db.execute("UPDATE candidates SET person=NULL,status='withheld',"
                           "payload=json_set(payload,'$.reason','auxiliary_insufficient_support') "
                           "WHERE person=? AND policy=? AND json_extract(payload,'$.reason')='auxiliary_identity'",
                           (person, policy))
        db.execute(f'INSERT OR IGNORE INTO members(observation,person) '
                   f'SELECT {address(db, "c")},c.person FROM candidates c '
                   "WHERE c.policy=? AND c.status='accepted' "
                   "AND json_extract(c.payload,'$.reason')='auxiliary_identity'", (policy,))
    if (grouping['view_prototypes_enabled'] or reference_views) and grouping['auxiliary_matching']:
        _attach_confirmed_views(db, selected, config, policy, groups, usage, cache,reference_views)


def _view_probe(db, candidate, reason):
    db.execute(
        "UPDATE candidates SET payload=json_set(payload,'$.auxiliary_view_probe',?) "
        'WHERE id=?', (reason, candidate))


def _attach_confirmed_views(db, selected, config, policy, anchors, usage, cache, extra_views=()):
    """A second, frozen pass: no new view evidence can promote another view.

    Only independently supported *anchor-matched* profiles are prototypes.
    Late matches are never used to train these prototypes in this replay.
    """
    grouping, settings = config.data['clustering'], config.data['recognition']
    views = (confirmed_views(db, policy, grouping,
                            remaining_bytes=grouping['max_reference_bytes'] - usage.bytes,complete=True)
             if grouping['view_prototypes_enabled'] else ())
    views=tuple({(v.group.identifier,v.pose,v.medoid):v for v in tuple(views)+tuple(extra_views)}.values())
    for view in views:
        db.execute(
            "UPDATE candidates SET payload=json_set(payload,"
            "'$.auxiliary_view_seeded',coalesce(json_extract(payload,'$.auxiliary_view_seeded'),0)+1) "
            f'WHERE policy=? AND {address(db, "candidates")}=?',
            (policy, view.medoid))
    db.execute(
        "UPDATE candidates SET payload=json_set(payload,'$.auxiliary_view_probe',"
        "'no_confirmed_prototype') WHERE policy=? AND status='withheld' "
        "AND json_extract(payload,'$.reason')='auxiliary_unmatched'",
        (policy,))
    if not views:
        return
    groups = [view.group for view in views]
    by_group = {id(view.group): view for view in views}
    threshold = max(settings['match_threshold'], grouping['view_match_threshold'])
    margin = max(settings['ambiguity_margin'], grouping['view_ambiguity_margin'])
    for item in selected:
        scan = item['scan']
        source = (f'FROM candidates c JOIN observations o ON o.id={address(db, "c")} '
                  "WHERE c.scan=? AND c.policy=? AND c.status='withheld' "
                  "AND json_extract(c.payload,'$.reason') IN ('auxiliary_unmatched','prior_attachment_query','insufficient_match_or_cohesion',"
                  "'candidate_group','anchor_view_insufficient_support','global_view_insufficient_support')")
        rows = db.execute('SELECT o.*,c.id AS candidate ' + source +
                          ' ORDER BY o.cue,o.geometry,o.embedding,o.id', (scan, policy))
        total = db.execute('SELECT COUNT(*) ' + source, (scan, policy)).fetchone()[0]
        with tracking(scored_rows(rows, groups, threshold-margin),
                      'Match confirmed identity views', unit='observation', total=total) as tracked:
            for row, frozen in tracked:
                _view_probe(db, row['candidate'], 'below_match_or_cohesion')
                query = cache.vector(row) if cache is not None else vector(row)
                # Multiple prototypes of one person must *not* consume both
                # best/second-best slots and hide another person's rival view.
                by_person = {}
                saw_cohesion = False
                for group, mean, lowest in contributions(
                        groups, query, frozen, len(groups), threshold-margin,
                        exact=(lambda g, q: cache.scores(g, q, row['id'])) if cache is not None else None):
                    if lowest < grouping['view_cohesion_threshold']:
                        saw_cohesion = True
                        continue
                    view = by_group[id(group)]
                    if not pose_matches(row['geometry'],view.pose):
                        continue
                    previous = by_person.get(group.identifier)
                    if previous is None or (mean, view.identifier) > (previous[0], previous[2].identifier):
                        by_person[group.identifier] = (mean, group, view)
                ranked = sorted(by_person.items(), key=lambda item: (-item[1][0], item[0]))
                if not ranked:
                    if saw_cohesion:
                        _view_probe(db, row['candidate'], 'below_cohesion_threshold')
                    continue
                if ranked[0][1][0] < threshold:
                    _view_probe(db, row['candidate'], 'below_match_threshold')
                    continue
                _view_probe(db, row['candidate'], 'ambiguous_identity')
                if len(ranked) > 1 and ranked[0][1][0]-ranked[1][1][0] < margin:
                    continue
                person, (score, group, view) = ranked[0]
                if not pose_matches(row['geometry'],view.pose):
                    _view_probe(db, row['candidate'], 'pose_mismatch')
                    continue
                from .anchor_views import _competing_anchor
                if _competing_anchor(anchors,person,query,score,margin):
                    _view_probe(db,row['candidate'],'competing_identity')
                    continue
                if not co_presence_allows(db, row, group, groups, query, config,
                                          peer_threshold=threshold,
                                          cohesion_threshold=grouping['view_cohesion_threshold']):
                    _view_probe(db, row['candidate'], 'co_presence')
                    continue
                proof = {'pose': view.pose, 'medoid': view.medoid, 'authority': view.authority,
                         'anchor_verified_support': view.support_count,
                         'support_examples': list(view.support),
                         'match_score': score, 'threshold': threshold,
                         'ambiguity_margin': margin}
                db.execute("UPDATE candidates SET person=?,status='pending',"
                           "payload=json_set(payload,'$.reason','view_identity_pending',"
                           "'$.view_prototype',json(?),'$.scores',json(?),"
                           "'$.auxiliary_view_probe','pending_support') WHERE id=?",

                           (person, encoded(proof),
                            encoded([item[1][0] for item in ranked[:2]]), row['candidate']))
    from .decision_resolution import resolve_pending
    resolve_pending(db, policy, config)
    # A single false near-neighbour cannot establish a new view attachment.
    from .query_support import eligible
    source = (f'FROM candidates c JOIN observations o ON o.id={address(db, "c")} '
              "WHERE c.policy=? AND c.status='pending' "
              "AND json_extract(c.payload,'$.reason')='view_identity_pending'")
    rows=db.execute("SELECT o.*,c.person,json_extract(c.payload,'$.view_prototype.medoid') AS medoid "+source+
                    ' ORDER BY o.embedding,o.geometry,o.id',(policy,))
    supported=eligible(rows,config,key=lambda r:r['person'],prototype=lambda r:r['medoid'])
    with db:
        db.execute("UPDATE candidates SET status='accepted',payload=json_set(payload,'$.reason','view_identity') "
            "WHERE policy=? AND status='pending' AND json_extract(payload,'$.reason')='view_identity_pending' "
            f"AND {address(db, 'candidates')} IN (SELECT value FROM json_each(?))",(policy,encoded(sorted(supported))))
        db.execute("UPDATE candidates SET person=NULL,status='withheld',"
                   "payload=json_set(payload,'$.reason','view_insufficient_support') "
                   "WHERE policy=? AND status='pending' "
                   "AND json_extract(payload,'$.reason')='view_identity_pending'", (policy,))
        db.execute(
            "UPDATE candidates SET payload=json_set(payload,'$.auxiliary_view_probe','accepted') "
            "WHERE policy=? AND json_extract(payload,'$.reason')='view_identity'",
            (policy,))
        db.execute(
            "UPDATE candidates SET payload=json_set(payload,'$.auxiliary_view_probe','insufficient_support') "
            "WHERE policy=? AND json_extract(payload,'$.reason')='view_insufficient_support'",
            (policy,))
        db.execute(f'INSERT OR IGNORE INTO members(observation,person) '
                   f'SELECT {address(db, "c")},c.person FROM candidates c '
                   "WHERE c.policy=? AND c.status='accepted' "
                   "AND json_extract(c.payload,'$.reason')='view_identity'", (policy,))

