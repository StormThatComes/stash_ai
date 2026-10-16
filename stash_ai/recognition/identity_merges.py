"""Evidence-backed, oldest-first aliases; never a transitive single-link union."""
import hashlib
import time

from .database import encoded
from .progress import tracking
from .profiling import measured, count
from .evidence_cache import current as evidence_for, latest_rows, MISSING
from .merge_cache import cached_cross, support_identity, incremental_votes, save_votes
from .scoring import comparisons


def canonical(db, person):
    seen = set()
    while person not in seen:
        seen.add(person)
        row = db.execute('SELECT target FROM identity_merges WHERE source=?', (person,)).fetchone()
        if row is None:
            return person
        person = row['target']
    raise ValueError('Recognition identity aliases contain a cycle.')


def _cross(first, second, config):
    from .clustering import _similarities
    total = 0.
    minimum = 1.
    for sample in first.samples:
        mean, low = _similarities(second, sample)
        minimum = min(minimum, low)
        if minimum < config.data['clustering']['cohesion_threshold']:
            return None
        total += mean
    mean = total / len(first.samples)
    if mean < config.data['recognition']['match_threshold']:
        return None
    return {'mean': mean, 'minimum': minimum}


@measured('merge_compatibility')
def _compatible(first, second, config):
    # Keep the original bounded references for every constituent until this
    # pass finishes. Capping a union early could hide an incompatible endpoint.
    worst_mean = worst_minimum = 1.
    for left in first:
        for right in second:
            scores = cached_cross(left, right, config, _cross)
            if scores is None:
                return None
            worst_mean = min(worst_mean, scores['mean'])
            worst_minimum = min(worst_minimum, scores['minimum'])
    return {'minimum_cross_mean': worst_mean, 'minimum_cross_similarity': worst_minimum}


def _matches(component, query, config, *, cache=None, observation=None):
    from .clustering import _similarities
    for group in component:
        mean, lowest = (cache.scores(group, query, observation) if cache is not None
                        else _similarities(group, query))
        if (mean < config.data['recognition']['match_threshold'] or
                lowest < config.data['clustering']['cohesion_threshold']):
            return False
    return True


@measured('merge_copresence')
def _cooccurs(db, first, second, components, selected, config):
    if db.execute('''SELECT 1 FROM members a JOIN observations x ON x.id=a.observation
      JOIN observations y ON y.scan=x.scan AND y.cue=x.cue
      JOIN members b ON b.observation=y.id WHERE a.person=? AND b.person=?
      AND x.id<>y.id AND x.rejected IS NULL AND y.rejected IS NULL LIMIT 1''', (first, second)).fetchone() is not None:
        return True
    # Membership is not a prerequisite for contextual evidence: the most
    # ambiguous observations are precisely those withheld from members.
    # Stream compatible, non-superseded cues, with one vector and at most four
    # small signatures in scratch. Exact duplicate detections are one sample.
    from .clustering import vector
    cue, left, right = None, [], []
    sql = '''SELECT o.id,s.id AS scan,s.scene,s.rowid AS scan_order,o.cue,o.dimensions,o.embedding,o.geometry
      FROM observations o JOIN scans s ON s.id=o.scan
      WHERE s.scope=? AND s.namespace=? AND s.policy_sha=? AND s.status='complete'
        AND o.rejected IS NULL
        {latest}
        AND EXISTS (SELECT 1 FROM observations y WHERE y.scan=o.scan AND y.cue=o.cue
                    AND y.id<>o.id AND y.rejected IS NULL)
      ORDER BY s.rowid,o.cue,o.id'''
    parameters = selected['scope'], selected['namespace'], selected['policy_sha']
    cache = evidence_for(db)
    rows = (cache.contextual(sql, parameters, selected) if cache is not None
            else latest_rows(db, sql, parameters, selected))
    for row in rows:
        key = row['scan_order'], row['cue']
        if key != cue:
            cue, left, right = key, [], []
        query = cache.vector(row) if cache is not None else vector(row)
        signature = (cache.signature(row) if cache is not None else
                     hashlib.sha256(row['geometry'].encode() + row['embedding']).digest())
        matches_left = _matches(components[first], query, config, cache=cache, observation=row['id'])
        matches_right = _matches(components[second], query, config, cache=cache, observation=row['id'])
        if (matches_left and any(other != signature for other in right) or
                matches_right and any(other != signature for other in left)):
            return True
        for matched, seen in ((matches_left, left), (matches_right, right)):
            if matched and signature not in seen and len(seen) < 2:
                seen.append(signature)
    return False


def _merge_support_policy(config):
    """Only thresholds that affect a nomination's evidence, not support counts.

    Changing the minimum support should reuse valid proofs; changes to match
    or cohesion thresholds must obtain new nominations from selected scenes.
    """
    from .context import policy_digest
    return policy_digest(config, 'merge-support-v1', lambda: {'format': 1,
                   'match_threshold': config.data['recognition']['match_threshold'],
                   'cohesion_threshold': config.data['clustering']['cohesion_threshold']})


@measured('merge_support')
def _supporting_cues(db, selected, policy, source, target, components, config, minimum, *, nomination=None):
    """Revalidate durable nominations against the current bounded references.

    One (scene, timestamp) is one vote, regardless of duplicated detections.
    A newer scan supersedes its earlier scene scan, even when its detection
    policy or media content changed; superseded observations cannot vote.
    """
    from .clustering import vector

    cues, seen = [], set()
    cache = evidence_for(db)
    identity = support_identity(cache, selected, policy, source, target, components, config, minimum) if nomination is not None else None
    reused = incremental_votes(cache, identity, selected, nomination, components, config, minimum, _matches)
    if reused is not MISSING:
        return reused
    votes = []
    for row in latest_rows(db, '''SELECT o.id,o.seconds,o.cue,o.dimensions,o.embedding,s.scene,s.id AS scan,s.rowid AS scan_order
      FROM merge_support m
      JOIN observations o ON o.id=m.observation
      JOIN scans s ON s.id=o.scan
      WHERE m.source=? AND m.target=? AND m.policy=?
        AND s.scope=? AND s.namespace=? AND s.policy_sha=?
        AND s.status='complete' AND o.rejected IS NULL
        {latest}
      ORDER BY s.rowid,o.cue,o.id''',
      (source, target, policy, selected['scope'], selected['namespace'],
       selected['policy_sha']), selected):
        key = (row['scene'], row['seconds'])
        if key in seen:
            continue
        query = cache.vector(row) if cache is not None else vector(row)
        if _matches(components[target], query, config, cache=cache, observation=row['id']) and _matches(components[source], query, config, cache=cache, observation=row['id']):
            seen.add(key)
            cues.append({'scene': row['scene'], 'seconds': row['seconds'], 'observation': row['id']})
            votes.append(dict(cues[-1], order=(row['scan_order'],row['cue'],row['id'])))
            if len(cues) >= minimum:
                break
    save_votes(cache, identity, votes)
    return cues


@measured('merging')
def supported_merges(database, scan, config, groups, policy, *, usage=None):
    """Evidence-backed consolidation with persisted cross-scene nominations.

    Only explicit selected completed scans can contribute; current sample
    comparisons, independent cue times, and oldest-first identity ownership
    still control every merge. No transitive single-link union is allowed.
    Caller owns the transaction including support and membership writes.
    """
    from .clustering import _similarities, vector, reference_context
    db = database.connection
    if len(groups) < 2:
        return 0
    if usage is not None:
        usage.begin('merging')
    selected = db.execute('SELECT id,scope,namespace,policy_sha,scene,status,rowid AS scan_order FROM scans WHERE id=?', (scan,)).fetchone()
    rows = {row['id']: row for row in db.execute(
        'SELECT p.id,p.created FROM people p JOIN scans s ON s.scope=p.scope AND s.namespace=p.namespace '
        'WHERE s.id=? AND NOT EXISTS (SELECT 1 FROM identity_merges WHERE source=p.id)', (scan,))}
    order = lambda person: (rows[person]['created'], person)
    roots = {g.identifier: g.identifier for g in groups}
    components = {g.identifier: [g] for g in groups}
    nominations = {}
    merged = 0
    settings = config.data['recognition']
    evidence_policy = _merge_support_policy(config)
    cache = evidence_for(db)
    with tracking(
        db.execute(
            'SELECT id,cue,seconds,dimensions,embedding FROM observations WHERE scan=? AND rejected IS '
            'NULL ORDER BY cue,id',
            (scan,),
        ),
        'Check identity merges',
        unit='observation',
        total=lambda: db.execute(
            'SELECT COUNT(*) FROM observations WHERE scan=? AND rejected IS NULL',
            (scan,),
        ).fetchone()[0],
    ) as tracked:
        for observation in tracked:
            if usage is not None:
                usage.emit()
            query = cache.vector(observation) if cache is not None else vector(observation)
            constituents = [group for component in components.values() for group in component]
            matched = {group.identifier for group, mean, low in comparisons(constituents, query,
                settings['match_threshold'], config.data['clustering']['cohesion_threshold'],
                exact=(lambda g,q: cache.scores(g,q,observation['id'])) if cache is not None else None)
                if mean >= settings['match_threshold'] and low >= config.data['clustering']['cohesion_threshold']}
            eligible = []
            for root, component in components.items():
                if all(group.identifier in matched for group in component):
                    eligible.append(root)
            eligible.sort(key=order)
            for index, source in enumerate(eligible[1:], 1):
                source = roots[source]
                for target in eligible[:index]:
                    target = roots[target]
                    if source == target:
                        continue
                    target, source = sorted((target, source), key=order)
                    count('merge_pairs')
                    scores = _compatible(components[target], components[source], config)
                    if scores is None:
                        continue
                    previous = nominations.get(source)
                    if previous is not None and roots[previous] != target:
                        if order(roots[previous]) < order(target):
                            break
                    nominations[source] = target
                    inserted = db.execute('INSERT OR IGNORE INTO merge_support(source,target,observation,policy) VALUES (?,?,?,?)',
                               (source, target, observation['id'], evidence_policy))
                    supporting = _supporting_cues(db, selected, evidence_policy, source, target,
                                                  components, config, settings['min_merge_support'],
                                                  nomination=(inserted.rowcount == 1, observation))
                    if len(supporting) >= settings['min_merge_support']:
                        cooccurring = _cooccurs(db, target, source, components, selected, config)
                        stronger = max(settings['match_threshold'], settings['cooccurrence_merge_threshold'])
                        if cooccurring and scores['minimum_cross_similarity'] < stronger:
                            # Reflections may co-occur, but need stronger proof.
                            db.execute("UPDATE candidates SET payload=json_set(payload,'$.merge_withheld_reason',?) "
                                       "WHERE scan=? AND policy=? AND status='withheld' AND json_extract(payload,'$.observation')=?",
                                       ('cooccurrence_identity_evidence_insufficient', scan, policy, observation['id']))
                            break
                        evidence = {'format': 3, 'cooccurring': cooccurring,
                            'cooccurrence_merge_threshold': stronger, 'scan': scan, 'policy': policy,
                            'support_cues': supporting,
                            'support_seconds': [item['seconds'] for item in supporting],
                            'scores': scores, 'representative_limit': config.data['clustering']['max_samples_per_person'],
                            'source_created': rows[source]['created'], 'target_created': rows[target]['created'],
                            'source_constituents': {'count': len(components[source]),
                                                    'references': reference_context(components[source])},
                            'target_constituents': {'count': len(components[target]),
                                                    'references': reference_context(components[target])}}
                        db.execute('INSERT INTO identity_merges VALUES (?,?,?,?,?)',
                                   (source, target, time.time(), encoded(evidence), 'pending'))
                        db.execute('UPDATE members SET person=? WHERE person=?', (target, source))
                        db.execute("UPDATE candidates SET payload=json_set(payload,'$.merged_from',person),person=? "
                                   'WHERE person=?', (target, source))
                        for group in components.pop(source):
                            roots[group.identifier] = target
                            components[target].append(group)
                        merged += 1
                    break
    query = None
    if cache is not None:
        cache.discard_vectors()
    if usage is not None:
        usage.begin('retained')
    return merged
