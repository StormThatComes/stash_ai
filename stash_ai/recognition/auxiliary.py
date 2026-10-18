"""Attach pose-limited faces only to previously established anchor identities.

Attachments are not reference samples, new identities, or merge votes.
"""
import uuid

from .clustering import _groups, vector, co_presence_allows, matching_policy
from .database import encoded
from .evidence_cache import current as evidence_for
from .parallel_scores import scored_rows, contributions
from .pose_policy import auxiliary_sql
from .progress import tracking
from .ranking import best_two
from .reference_usage import ReferenceUsage
from .replay_addresses import address


def apply_auxiliary(database, selected, config, *, progress=None):
    grouping, settings = config.data['clustering'], config.data['recognition']
    if grouping['max_anchor_pose_deviation'] is None or not selected:
        return
    db, policy = database.connection, matching_policy(config)
    cache = evidence_for(db)
    first = db.execute('SELECT scope,namespace FROM scans WHERE id=?', (selected[0]['scan'],)).fetchone()
    groups = _groups(db, first['scope'], first['namespace'], grouping,
                     usage=ReferenceUsage(grouping['max_reference_bytes'], progress),
                     stage='auxiliary')
    condition = auxiliary_sql('o', grouping)
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
                if grouping['auxiliary_matching'] and groups:
                    query = cache.vector(row) if cache is not None else vector(row)
                    ranked = []
                    for group, mean, low in contributions(
                            groups, query, frozen, len(groups),
                            settings['match_threshold'] - settings['ambiguity_margin'],
                            exact=(lambda g, q: cache.scores(g, q, row['id'])) if cache is not None else None):
                        if low >= grouping['cohesion_threshold']:
                            best_two(ranked, (mean, group.identifier, group))
                    scores = [entry[0] for entry in ranked[:2]]
                    if ranked and ranked[0][0] >= settings['match_threshold']:
                        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < settings['ambiguity_margin']:
                            reason = 'auxiliary_ambiguous'
                        elif co_presence_allows(db, row, ranked[0][2], groups, query, config):
                            person, status, reason = ranked[0][1], 'pending', 'auxiliary_identity'
                        else:
                            reason = 'auxiliary_co_presence'
                elif not grouping['auxiliary_matching']:
                    reason = 'auxiliary_disabled'
                candidate = str(uuid.uuid5(uuid.UUID(row['id']), policy))
                db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?)',
                           (candidate, scan, policy, person, status,
                            encoded({'observation': row['id'], 'reason': reason, 'scores': scores,
                                     'policy': policy, 'representative_limit': grouping['max_samples_per_person']})))
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
