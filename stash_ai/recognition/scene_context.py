"""Attach borderline observations to a strictly established same-scene core.

Run only after the entire cold replay has finished strict matching and merging.
The frozen references and strict local support never include these attachments.
"""
from .clustering import _groups, _similarities, co_presence_allows, matching_policy, reference_context, vector
from .database import encoded
from .progress import tracking, work
from .ranking import best_two
from .reference_usage import ReferenceUsage
from .scoring import comparisons
from .replay_addresses import address
from .evidence_cache import current as evidence_for
from .parallel_scores import scored_rows, contributions


def _rank(groups, query, config, *, cache=None, observation=None, frozen=None):
    ranked = []
    cohesion = config.data['clustering']['cohesion_threshold']
    for group, mean, minimum in contributions(groups, query, frozen, len(groups),
            float('-inf'), cohesion,
            exact=(lambda g,q: cache.scores(g,q,observation)) if cache is not None else None):
        if minimum >= cohesion:
            best_two(ranked, (mean, group.identifier, group))
    return ranked


def _unambiguous(ranked, margin):
    return bool(ranked and (len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= margin))


def _cores(db, scan, policy, groups, config):
    """Revalidate strong accepted members against the final strict references."""
    cache = evidence_for(db)
    settings = config.data['recognition']
    minimum = config.data['clustering']['same_scene_min_support']
    support = {}
    source = (f"FROM candidates c JOIN observations o ON o.id={address(db, 'c')} "
        "JOIN members m ON m.observation=o.id AND c.person=m.person "
        "WHERE c.policy=? AND c.scan=o.scan AND o.scan=? AND o.rejected IS NULL AND c.status='accepted'")
    rows = db.execute("SELECT o.id,o.seconds,o.dimensions,o.embedding,m.person " + source +
        " ORDER BY o.cue,o.geometry,o.embedding,o.id", (policy, scan))
    with tracking(scored_rows(rows, groups, float('-inf'), config.data['clustering']['cohesion_threshold']),
                  'Validate same-scene support', unit='observation',
                  total=lambda: db.execute('SELECT COUNT(*) ' + source, (policy, scan)).fetchone()[0]) as tracked:
        for row, frozen in tracked:
            cues = support.setdefault(row['person'], {})
            if len(cues) >= minimum or row['seconds'] in cues:
                continue
            query = cache.vector(row) if cache is not None else vector(row)
            ranked = _rank(groups, query, config, cache=cache, observation=row['id'], frozen=frozen)
            if (ranked and ranked[0][1] == row['person'] and ranked[0][0] >= settings['match_threshold']
                    and _unambiguous(ranked, settings['ambiguity_margin'])):
                cues[row['seconds']] = row['id']
    return {person: cues for person, cues in support.items() if len(cues) >= minimum}


def apply_scene_context(database, selected, config, *, progress=None):
    """One final cold-replay pass, before binding labels or copying assignments.

    This helper consumes a replay containing only the selected scans, not the
    historical database. Raw scores, ambiguity and cohesion stay independent.
    """
    grouping, settings = config.data['clustering'], config.data['recognition']
    bonus = grouping['same_scene_match_bonus']
    if not bonus or not selected:
        return
    db, policy = database.connection, matching_policy(config)
    cache = evidence_for(db)
    if not db.execute("SELECT 1 FROM candidates WHERE policy=? AND status='withheld' LIMIT 1", (policy,)).fetchone():
        return
    # Replay-only index: each selected scene must not rescan every candidate
    # in the library. No durable schema migration is needed for this phase.
    with work('Index same-scene candidates', 1, unit='query', position=0) as indexed:
        db.execute('CREATE INDEX scene_context_candidates ON candidates(scan,policy,status)')
        indexed.advance()
    first = db.execute('SELECT scope,namespace FROM scans WHERE id=?', (selected[0]['scan'],)).fetchone()
    usage = ReferenceUsage(grouping['max_reference_bytes'], progress)
    groups = _groups(db, first['scope'], first['namespace'], grouping, usage=usage, stage='scene_context')
    strict_context = reference_context(groups)
    threshold = max(grouping['cohesion_threshold'], settings['match_threshold'] - bonus)
    with tracking(selected, 'Apply same-scene context', unit='scene', position=0) as tracked:
        for item in tracked:
            scan = item['scan']
            if not db.execute("SELECT 1 FROM candidates WHERE scan=? AND policy=? AND status='withheld' LIMIT 1",
                              (scan, policy)).fetchone():
                continue
            cores = _cores(db, scan, policy, groups, config)
            if not cores:
                continue
            source = (f"FROM candidates c JOIN observations o ON o.id={address(db, 'c')} "
                "WHERE c.policy=? AND c.scan=o.scan AND o.scan=? AND o.rejected IS NULL AND c.status='withheld'")
            rows = db.execute("SELECT o.*,c.id AS candidate " + source +
                " ORDER BY o.cue,o.geometry,o.embedding,o.id", (policy, scan))
            with tracking(scored_rows(rows, groups, float('-inf'), grouping['cohesion_threshold']),
                          'Match same-scene observations', unit='observation',
                          total=lambda: db.execute('SELECT COUNT(*) ' + source, (policy, scan)).fetchone()[0]) as observations:
                for row, frozen in observations:
                    usage.emit()
                    query = cache.vector(row) if cache is not None else vector(row)
                    ranked = _rank(groups, query, config, cache=cache, observation=row['id'], frozen=frozen)
                    if (not ranked or ranked[0][0] < threshold or ranked[0][1] not in cores
                            or not _unambiguous(ranked, settings['ambiguity_margin'])):
                        continue
                    group = ranked[0][2]
                    # Include peers below the ordinary threshold: otherwise two
                    # borderline faces in one cue could bypass co-presence checks.
                    if not co_presence_allows(db, row, group, groups, query, config, peer_threshold=threshold):
                        continue
                    evidence = {'scene': item['scene'], 'match_bonus': bonus, 'threshold': threshold,
                                'reference_context': strict_context,
                                'support': [{'seconds': seconds, 'observation': observation}
                                            for seconds, observation in cores[group.identifier].items()]}
                    db.execute('INSERT INTO members VALUES (?,?)', (row['id'], group.identifier))
                    db.execute("UPDATE candidates SET person=?,status='accepted',payload=json_set(payload,"
                        "'$.reason','same_scene_identity','$.scores',json(?),'$.same_scene_context',json(?)) WHERE id=?",
                        (group.identifier, encoded([r[0] for r in ranked]), encoded(evidence), row['candidate']))
    groups.clear()
    usage.emit(force=True)
