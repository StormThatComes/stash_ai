"""Attach borderline observations to a strictly established same-scene core.

Run only after the entire cold replay has finished strict matching and merging.
The frozen references and strict local support never include these attachments.
"""
from dataclasses import replace

from .clustering import _groups, _similarities, co_presence_allows, matching_policy, reference_context, vector
from .config import plain
from .identity_merges import supported_merges
from .database import encoded
from .progress import tracking, work
from .ranking import best_two
from .reference_usage import ReferenceUsage
from .scoring import comparisons
from .replay_addresses import address
from .evidence_cache import current as evidence_for
from .parallel_scores import scored_rows, contributions
from .pose_policy import anchor_sql


def _rank(groups, query, config, *, cache=None, observation=None, frozen=None, cohesion=None):
    ranked = []
    cohesion = (config.data['clustering']['cohesion_threshold']
                if cohesion is None else cohesion)
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
    anchor = anchor_sql('o', config.data['clustering'])
    minimum = config.data['clustering']['same_scene_min_support']
    from .independent_evidence import Votes, representatives, grouped_counts
    support, evidence = {}, {}
    source = (f"FROM candidates c JOIN observations o ON o.id={address(db, 'c')} "
        "JOIN members m ON m.observation=o.id AND c.person=m.person "
        f"WHERE c.policy=? AND c.scan=o.scan AND o.scan=? AND o.rejected IS NULL AND c.status='accepted' AND {anchor}")
    rows = db.execute("SELECT o.id,o.scan,o.quality,o.seconds,o.dimensions,o.embedding,m.person " + source +
        " ORDER BY o.cue,o.geometry,o.embedding,o.id", (policy, scan))
    with tracking(scored_rows(rows, groups, float('-inf'), config.data['clustering']['cohesion_threshold']),
                  'Validate same-scene support', unit='observation',
                  total=lambda: db.execute('SELECT COUNT(*) ' + source, (policy, scan)).fetchone()[0]) as tracked:
        for row, frozen in tracked:
            cues = evidence.setdefault(row['person'], Votes())
            if len(cues) >= minimum:
                continue
            query = cache.vector(row) if cache is not None else vector(row)
            ranked = _rank(groups, query, config, cache=cache, observation=row['id'], frozen=frozen)
            if (ranked and ranked[0][1] == row['person'] and ranked[0][0] >= settings['match_threshold']
                    and _unambiguous(ranked, settings['ambiguity_margin'])):
                cues.add(row)
                support.setdefault(row['person'], []).append(row)
    return {person: {r['seconds']: r['id'] for r in representatives(rows)}
            for person, rows in support.items() if len(evidence[person]) >= minimum}


def apply_scene_context(database, selected, config, *, progress=None):
    """Resolve local borderline faces and compatible fragments after strict replay.

    Strict references stay frozen throughout the contextual phase. Local merges
    use the same complete-group/independent-cue protections as strict merges,
    but can never become seeds for the earlier cross-scene replay.
    """
    grouping, settings = config.data['clustering'], config.data['recognition']
    bonus = grouping['same_scene_match_bonus']
    anchor = anchor_sql('o', grouping)
    if not bonus or not selected:
        return
    db, policy = database.connection, matching_policy(config)
    cache = evidence_for(db)
    has_withheld = db.execute(
        "SELECT 1 FROM candidates WHERE policy=? AND status='withheld' LIMIT 1",
        (policy,)).fetchone() is not None
    with work('Index same-scene candidates', 1, unit='query', position=0) as indexed:
        db.execute('CREATE INDEX scene_context_candidates ON candidates(scan,policy,status)')
        indexed.advance()
    first = db.execute('SELECT scope,namespace FROM scans WHERE id=?', (selected[0]['scan'],)).fetchone()
    usage = ReferenceUsage(grouping['max_reference_bytes'], progress)
    groups = _groups(db, first['scope'], first['namespace'], grouping, usage=usage, stage='scene_context')
    strict_context = reference_context(groups)
    local_match = max(0., settings['match_threshold'] - bonus)
    local_cohesion = max(0., grouping['cohesion_threshold'] - bonus)

    # Existing withheld observations may attach to a strictly proven identity,
    # but do not become new references or support for the local merge pass.
    if has_withheld:
        with tracking(selected, 'Apply same-scene context', unit='scene', position=0) as tracked:
            for item in tracked:
                scan = item['scan']
                if not db.execute(
                    "SELECT 1 FROM candidates WHERE scan=? AND policy=? AND status='withheld' LIMIT 1",
                    (scan, policy)).fetchone():
                    continue
                cores = _cores(db, scan, policy, groups, config)
                if not cores:
                    continue
                source = (f"FROM candidates c JOIN observations o ON o.id={address(db, 'c')} "
                    f"WHERE c.policy=? AND c.scan=o.scan AND o.scan=? AND o.rejected IS NULL AND c.status='withheld' AND {anchor}")
                rows = db.execute("SELECT o.*,c.id AS candidate " + source +
                    " ORDER BY o.cue,o.geometry,o.embedding,o.id", (policy, scan))
                with tracking(scored_rows(rows, groups, float('-inf'), local_cohesion),
                              'Match same-scene observations', unit='observation',
                              total=lambda: db.execute('SELECT COUNT(*) ' + source, (policy, scan)).fetchone()[0]) as observations:
                    for row, frozen in observations:
                        usage.emit()
                        query = cache.vector(row) if cache is not None else vector(row)
                        ranked = _rank(groups, query, config, cache=cache,
                                       observation=row['id'], frozen=frozen, cohesion=local_cohesion)
                        if (not ranked or ranked[0][0] < local_match or ranked[0][1] not in cores
                                or not _unambiguous(ranked, settings['ambiguity_margin'])):
                            continue
                        group = ranked[0][2]
                        # Consider borderline cue peers as conflicts too; the
                        # reflection requirement remains at its strict value.
                        if not co_presence_allows(db, row, group, groups, query, config,
                                                  peer_threshold=local_match,
                                                  cohesion_threshold=local_cohesion):
                            continue
                        proof = {'scene': item['scene'], 'match_bonus': bonus,
                                 'threshold': local_match, 'cohesion_threshold': local_cohesion,
                                 'reference_context': strict_context,
                                 'support': [{'seconds': seconds, 'observation': observation}
                                             for seconds, observation in cores[group.identifier].items()]}
                        db.execute('INSERT INTO members VALUES (?,?)', (row['id'], group.identifier))
                        db.execute("UPDATE candidates SET person=?,status='accepted',payload=json_set(payload,"
                            "'$.reason','same_scene_identity','$.scores',json(?),'$.same_scene_context',json(?)) WHERE id=?",
                            (group.identifier, encoded([r[0] for r in ranked]), encoded(proof), row['candidate']))

    # Never reuse the looser policy for the earlier strict replay. Only groups
    # already present in the current scene may consolidate, with a strictly
    # accepted local anchor and distinct-cue merge evidence.
    local_config = replace(config, data=plain(config.data))
    local_config.data['recognition']['match_threshold'] = local_match
    local_config.data['clustering']['cohesion_threshold'] = local_cohesion
    local_state = {}
    with tracking(selected, 'Merge same-scene identities', unit='scene', position=0) as tracked:
        for item in tracked:
            scan = item['scan']
            from .independent_evidence import grouped_counts
            strict = grouped_counts(db.execute(
                f"SELECT c.person,o.scan,o.seconds,o.quality FROM candidates c "
                f"JOIN observations o ON o.id={address(db, 'c')} "
                "WHERE c.scan=? AND c.policy=? AND c.status='accepted' "
                "AND json_extract(c.payload,'$.reason')<>'same_scene_identity' "
                f"AND {anchor} ORDER BY c.person", (scan, policy)), 'person')
            if len(strict) < 2:
                continue
            cores = {person for person, count in strict.items() if count >= grouping['same_scene_min_support']}
            if not cores:
                continue
            supported_merges(database, scan, local_config, groups, policy,
                             local_state=local_state,
                             local_groups=set(strict),
                             local_cores=cores)
    groups.clear()
    usage.emit(force=True)
