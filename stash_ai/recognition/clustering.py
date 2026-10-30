"""Scoped, conservative incremental matching against retained face identities."""
from dataclasses import dataclass
import hashlib
import json
import time
import uuid

from .config import digest
from .database import encoded
from .reference_usage import ReferenceUsage
from .reference_roles import strict_sql
from .identity_merges import supported_merges
from .progress import tracking
from .ranking import best_two
from .replay_addresses import address
from .rebuild_cache import resource_limits
from .profiling import measured, count
from .scoring import comparisons, scoped
from .evidence_cache import current as evidence, scoped as evidence_owned
from .parallel_scores import scored_rows, contributions

MATCHING_FORMAT = 56


def vector(row):
    import numpy as np
    from .run_cache import current, MISSING
    dimensions, blob = row['dimensions'], row['embedding']
    if (type(dimensions) is not int or not 1 <= dimensions <= 1024 or
            not isinstance(blob, bytes) or len(blob) != 4 * dimensions):
        raise ValueError('Invalid retained face embedding shape/length.')
    cache = current()
    key = 'vector', dimensions, blob
    found = cache.get(key) if cache is not None else MISSING
    if found is not MISSING:
        return found
    count('validated_vectors')
    result = np.frombuffer(blob, dtype='<f4')
    if not np.isfinite(result).all() or abs(float(np.linalg.norm(result)) - 1) > .001:
        raise ValueError('Invalid retained face embedding values/normalization.')
    if cache is not None:
        cache.put(key, result, len(blob) + 256)
    return result


@dataclass
class Group:
    identifier: str
    samples: list
    known: bool
    sample_ids: tuple = ()


def reference_admission(policy, *, scan_policy=None, label_scenes=(), compatibility=None):
    from .pose_policy import anchor_sql
    from .scan_heads import predicate as current_scan
    anchor = anchor_sql('o', policy)
    if scan_policy is not None:
        compatible = ('o.scan IN (SELECT current.id FROM scans current WHERE (current.policy_sha=' + repr(scan_policy) +
            (' OR '+compatibility if compatibility else '') + ')' +
            ' AND '+current_scan('current')+')')
        if label_scenes:
            literal = "'" + encoded(list(label_scenes)).replace("'", "''") + "'"
            compatible += ' OR o.scan IN (SELECT id FROM scans WHERE scene IN (SELECT value FROM json_each(' + literal + ')))'
        anchor += ' AND (' + compatible + ')'
    return anchor


@measured('references')
def _groups(db, scope, namespace, policy, *, usage=None, stage='loading', scan_policy=None, label_scenes=(), compatibility=None):
    anchor = reference_admission(policy,scan_policy=scan_policy,label_scenes=label_scenes,compatibility=compatibility) + ' AND o.rejected IS NULL'
    strict = strict_sql()
    usage = usage or ReferenceUsage(policy['max_reference_bytes'])
    cache = evidence(db)
    if cache is not None:
        cache.discard_vectors()
    from .reference_index import current
    index = current(db)
    usage.begin(stage, reset=True, report=index is None)
    if index is not None and scan_policy is None:
        result = index.snapshot(policy, usage)
        usage.emit(force=True)
        return result
    maximum = policy['max_people_per_lookup']
    rows = db.execute('SELECT id FROM people WHERE scope=? AND namespace=? '
                      'AND NOT EXISTS (SELECT 1 FROM identity_merges WHERE source=people.id) '
                      f'AND EXISTS (SELECT 1 FROM members m JOIN observations o ON o.id=m.observation '
                      f'WHERE m.person=people.id AND {anchor} AND {strict}) '
                      'ORDER BY created,id LIMIT ?',
                      (scope, namespace, maximum + 1)).fetchall()
    if len(rows) > maximum:
        raise ValueError('Face identity lookup exceeds configured person budget.')
    result = []
    from itertools import islice
    iterator = iter(rows)
    while batch := list(islice(iterator, 256)):
        identifiers = encoded([person['id'] for person in batch])
        invalid = db.execute('SELECT 1 FROM members m JOIN observations o ON o.id=m.observation '
            'JOIN scans s ON s.id=o.scan WHERE m.person IN (SELECT value FROM json_each(?)) AND '
            "(s.scope<>? OR s.namespace<>? OR s.status<>'complete') LIMIT 1",
            (identifiers, scope, namespace)).fetchone()
        if invalid:
            raise ValueError('Face identity has incompatible retained membership evidence.')
        from .reference_selection import select
        with tracking(batch, 'Load identity references', unit='person') as tracked:
            for person in tracked:
                rows = db.execute('SELECT o.id,o.dimensions,o.embedding,o.quality,o.geometry,o.seconds,s.scene FROM observations o '
                    'JOIN members m ON m.observation=o.id JOIN scans s ON s.id=o.scan '
                    f'WHERE m.person=? AND {anchor} AND {strict} '
                    'ORDER BY CAST(s.scene AS INTEGER),s.scene,o.cue,o.seconds,o.geometry,o.embedding,o.id',
                    (person['id'],))
                selected = select(rows, policy['max_samples_per_person'])
                if not selected:
                    continue
                ids, vectors = zip(*selected)
                usage.add(vectors, people=1, kind='Retained')
                result.append(Group(person['id'], tuple(vectors), True, tuple(ids)))

    usage.emit(force=True)
    return result


def _similarities(group, query):
    import numpy as np
    if any(sample.shape != query.shape for sample in group.samples):
        raise ValueError('Face namespace contains mixed embedding dimensions.')
    count('group_comparisons')
    count('sample_comparisons', len(group.samples))
    from .run_cache import current, immutable_samples, MISSING
    cache=current();samples=immutable_samples(group.samples) if cache is not None else None
    query_key=immutable_samples((query,)) if samples is not None else None
    key=('scalar-query',samples,query_key)
    result=cache.get(key) if query_key is not None else MISSING
    if result is MISSING:
        scores = np.asarray([float(np.dot(sample, query)) for sample in group.samples])
        result=float(scores.mean()), float(scores.min())
        if query_key is not None:cache.put(key,result,sum(len(v)+128 for v in samples+query_key))
    return result


@measured('cue_context')
def co_presence_allows(db, observation, group, groups, query, config, *, peer_threshold=None, cohesion_threshold=None, resolved_peers=()):
    """A distinct cue peer competing for this identity needs stronger evidence.

    Inspect the entire cue, including observations not assigned yet, so reader
    order cannot hide a second face. A peer clearly belonging to another group
    does not make an otherwise unambiguous identity unsafe. Stream vectors;
    exact duplicate detections remain one piece of evidence.
    """
    import numpy as np
    settings = config.data['recognition']
    peer_threshold = settings['match_threshold'] if peer_threshold is None else peer_threshold
    cohesion_threshold = (config.data['clustering']['cohesion_threshold']
                          if cohesion_threshold is None else cohesion_threshold)
    stronger = max(settings['match_threshold'], settings['cooccurrence_merge_threshold'])
    cache = evidence(db)
    peers = (cache.peers(observation['scan'], observation['cue']) if cache is not None else
        db.execute('SELECT id,geometry,dimensions,embedding FROM observations WHERE scan=? AND cue=? '
                   'AND rejected IS NULL ORDER BY id', (observation['scan'], observation['cue'])))
    for peer in peers:
        if peer['id'] == observation['id'] or peer['id'] in resolved_peers:
            continue
        from .physical_faces import duplicate
        if duplicate(peer,observation,config.data['clustering'].get('deduplicate_detections',False)):
            continue
        other = cache.vector(peer) if cache is not None else vector(peer)
        mean, minimum = cache.scores(group, other, peer['id']) if cache is not None else _similarities(group, other)
        if mean < peer_threshold or minimum < cohesion_threshold:
            continue
        best = mean
        for rival in groups:
            if rival.identifier == group.identifier:continue
            rival_mean, rival_minimum = (cache.scores(rival, other, peer['id']) if cache is not None
                                          else _similarities(rival, other))
            if rival_minimum >= cohesion_threshold:
                best = max(best, rival_mean)
        if best - mean >= settings['ambiguity_margin'] and best > mean:
            continue
        query_minimum = (cache.scores(group, query, observation['id'])[1] if cache is not None
                         else _similarities(group, query)[1])
        if min(query_minimum, minimum, float(np.dot(query, other))) < stronger:
            return False
    return True


def reference_context(groups):
    """Fingerprint exactly the bounded references used for identity decisions."""
    result = hashlib.sha256()
    for group in sorted(groups, key=lambda item: item.identifier):
        result.update(encoded([group.identifier, len(group.samples)]).encode('utf-8'))
        for sample in group.samples:
            result.update(encoded([len(sample)]).encode('ascii'))
            result.update(sample.astype('<f4', copy=False).tobytes())
    return result.hexdigest()


def _result(db, scan, policy):
    rows = db.execute('SELECT person,status,COUNT(*) AS observations FROM candidates '
                      'WHERE scan=? AND policy=? GROUP BY person,status ORDER BY person,status', (scan, policy))
    return [dict(row) for row in rows]


def complete_candidates(db, scan, policy, context, count, *, exact_memberships=False, admission=None):
    """Derived cache absence cannot mean that supported identities disappeared."""
    totals = db.execute("SELECT COUNT(*),COUNT(DISTINCT json_extract(payload,'$.observation')) "
                        'FROM candidates WHERE scan=? AND policy=?', (scan, policy)).fetchone()
    if tuple(totals) != (count, count):
        return False
    rows = db.execute("SELECT c.id,c.person,c.status,json_extract(c.payload,'$.observation') AS observation,"
        "json_extract(c.payload,'$.reference_context') AS context,json_extract(c.payload,'$.resource_limits') AS limits,"
        "o.scan,o.rejected,m.person AS member "
        "FROM candidates c LEFT JOIN observations o ON o.id=json_extract(c.payload,'$.observation') "
        'LEFT JOIN members m ON m.observation=o.id WHERE c.scan=? AND c.policy=?', (scan, policy))
    for row in rows:
        try:
            expected = str(uuid.uuid5(uuid.UUID(row['observation']), policy))
        except (ValueError, TypeError, AttributeError):
            return False
        if (row['id'] != expected or row['scan'] != scan or row['rejected'] is not None
                or row['context'] != context or row['status'] not in ('accepted', 'withheld')):
            return False
        if admission is not None:
            try:
                limits=json.loads(row['limits'])
            except (ValueError,TypeError):
                return False
            if not isinstance(limits,dict) or any(type(limits.get(k)) is not int or admission[k]<limits[k] for k in admission):
                return False
        if row['status'] == 'accepted':
            if row['person'] is None or row['member'] != row['person']:
                return False
        elif exact_memberships and row['member'] is not None:
            return False
    return True


def matching_policy(config):
    from .context import policy_digest
    from .person_signal import contract
    return policy_digest(config, f'matching-v{MATCHING_FORMAT}-assignment-guards-v1', lambda: {
        **({'assignment_guards': 4} if config.data['sapiensid']['enabled'] else {}),
        **({'hybrid': {'contract': contract(config),
                      'collection_quality':config.data['sapiensid']['collection'],
                      'matching': config.data['sapiensid']['matching']}} if config.data['sapiensid']['enabled'] else {}),
        **({'body_anchor_quality':{k:config.data['sapiensid']['identities'][k] for k in
              ('anchor_min_person_pixels','anchor_min_keypoints','anchor_min_sharpness')}}
           if config.data['sapiensid']['enabled'] else {}),
        **({'body': {'local':config.data['sapiensid']['identities'],'global':config.data['sapiensid']['global_matching']}} if config.data['sapiensid']['identities']['enabled'] else {}),
        'format': MATCHING_FORMAT, 'recognition': config.data['recognition'],
        'clustering': {key:value for key,value in config.data['clustering'].items()
                       if key not in ('max_reference_bytes','max_people_per_lookup')}})


@measured('scene')
@scoped
@evidence_owned
def cluster_scene(database, scan, config, *, progress=None, _cold_replay=False):
    """Selected evidence can merge compatible historical identities oldest-first.

    Cohesion uses a bounded representative sample, not an N-by-N matrix or a
    transitive single-link union. Approximation is explicit in policy provenance.
    """
    import numpy as np
    db = database.connection
    cache = evidence(db)
    record = db.execute('SELECT * FROM scans WHERE id=?', (scan,)).fetchone()
    if record is None or record['status'] != 'complete':
        raise ValueError('Face matching requires a complete selected-scene scan.')
    policy = matching_policy(config)
    from .pose_policy import anchor_sql
    anchor = anchor_sql('o', config.data['clustering'])
    count = db.execute(f'SELECT COUNT(*) FROM observations o WHERE scan=? AND rejected IS NULL AND {anchor}',
                       (scan,)).fetchone()[0]
    usage = ReferenceUsage(config.data['clustering']['max_reference_bytes'], progress)
    if not count:
        usage.begin('no_observations')
        return []
    settings, grouping = config.data['recognition'], config.data['clustering']
    independent = _cold_replay and grouping['anchor_view_matching']
    # Scene hypotheses must not borrow an earlier scene's provisional labels.
    # Global resolution sees every independently established local core later.
    groups = [] if independent else _groups(db, record['scope'], record['namespace'], grouping, usage=usage)
    existing = 0
    if not _cold_replay:
        context = reference_context(groups)
        existing = db.execute('SELECT COUNT(*) FROM candidates WHERE scan=? AND policy=? '
                              "AND json_extract(payload,'$.reference_context')=?", (scan, policy, context)).fetchone()[0]
    if existing == count and count and complete_candidates(db, scan, policy, context, count, admission=resource_limits(config)):
        with tracking(
            db.execute(
                f'SELECT dimensions,embedding FROM observations o WHERE scan=? AND rejected IS NULL AND {anchor}',
                (scan,),
            ),
            'Validate cached matching',
            unit='observation',
            total=count,
        ) as tracked:
            for row in tracked:
                vector(row)
        usage.begin('cached')
        return _result(db, scan, policy)
    usage.begin('matching')
    with db:
        fixed_count = len(groups)
        by_id = {g.identifier: g for g in groups}
        db.execute('DELETE FROM candidates WHERE scan=? AND policy=?', (scan, policy))
        from .local_cohorts import ordering
        query_order=ordering(db) if independent else 'o.cue'
        previous=("EXISTS (SELECT 1 FROM prior_attachment_queries q WHERE q.id=o.id)" if _cold_replay and
                  db.execute("SELECT 1 FROM sqlite_temp_master WHERE name='prior_attachment_queries'").fetchone() else '0')
        with tracking(
            scored_rows(db.execute(
                'SELECT o.*,m.person AS prior_person,'+previous+' AS prior_query FROM observations o LEFT JOIN members m '
                f'ON m.observation=o.id WHERE o.scan=? AND o.rejected IS NULL AND {anchor} '
                'ORDER BY '+query_order+',o.geometry,o.embedding,o.cue,o.id',
                (scan,),
            ), groups, settings['match_threshold'] - settings['ambiguity_margin']),
            'Match face observations',
            unit='observation',
            total=count,
        ) as tracked:
            for observation, frozen in tracked:
                query = cache.vector(observation) if cache is not None else vector(observation)
                prior_person = observation['prior_person']
                ranked = []
                near = False
                for group, score, cohesion in contributions(
                        groups, query, frozen, fixed_count, settings['match_threshold'] - settings['ambiguity_margin'],
                        exact=(lambda g,q: cache.scores(g,q,observation['id'])) if cache is not None else None):
                    near |= score >= settings['match_threshold'] - settings['ambiguity_margin']
                    if (score >= settings['match_threshold'] - settings['ambiguity_margin'] and
                            cohesion >= grouping['cohesion_threshold']):
                        best_two(ranked, (score, group.identifier, group))
                scores = [item[0] for item in ranked[:2]]
                eligible = bool(ranked and ranked[0][0] >= settings['match_threshold'])
                if observation['prior_query']:
                    person, status, reason = None, 'withheld', 'prior_attachment_query'
                elif eligible and len(ranked) > 1 and ranked[0][0] - ranked[1][0] < settings['ambiguity_margin']:
                    person, status, reason = None, 'withheld', 'ambiguous_identity'
                elif eligible:
                    group = ranked[0][2]
                    person, status, reason = group.identifier, 'pending', 'known_identity' if group.known else 'candidate_group'
                    if not co_presence_allows(db, observation, group, groups, query, config):
                        person, status, reason = None, 'withheld', 'co_present_identity_conflict'
                    elif prior_person is not None and prior_person != person:
                        person, status, reason = None, 'withheld', 'policy_reassignment_requires_review'
                    if person is not None and not group.known:
                        if grouping['strict_appearance_matching']:
                            from .reference_selection import refresh_local
                            refresh_local(db,group,observation,grouping['max_samples_per_person'],usage,policy=policy,config=config)
                        elif len(group.samples) < grouping['max_samples_per_person']:
                            usage.add([query], kind='Selected')
                            group.samples.append(query)
                elif near:
                    person, status, reason = None, 'withheld', 'insufficient_match_or_cohesion'
                elif prior_person is not None:
                    person, status, reason = None, 'withheld', 'policy_reassignment_requires_review'
                else:
                    if len(groups) >= grouping['max_people_per_lookup']:
                        raise ValueError('Selected-scene face groups exceed the configured budget.')
                    person = str(uuid.uuid5(uuid.UUID(database.installation),
                                           encoded([record['scope'], record['namespace'], observation['id']])))
                    usage.add([query], people=1, kind='Selected')
                    group = Group(person, [query], False, (observation['id'],))
                    groups.append(group)
                    by_id[person] = group
                    status, reason = 'pending', 'candidate_group'
                candidate = str(uuid.uuid5(uuid.UUID(observation['id']), policy))
                db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?)',
                           (candidate, scan, policy, person, status,
                            encoded({'observation': observation['id'], 'reason': reason, 'scores': scores,
                                     'prior_attachment_query':bool(observation['prior_query']),
                                     'policy': policy, 'representative_limit': grouping['max_samples_per_person'],
                                     'resource_limits': resource_limits(config)})))
        if grouping['strict_appearance_matching']:
            from .reference_selection import refresh_local
            for group in groups[fixed_count:]:
                refresh_local(db,group,None,grouping['max_samples_per_person'],usage,policy=policy,config=config)
        usage.emit(force=True)
        from .decision_resolution import resolve_pending
        resolve_pending(db, policy, config, scans=(scan,))
        people = [row[0] for row in db.execute('SELECT DISTINCT person FROM candidates '
                                                "WHERE scan=? AND policy=? AND status='pending'", (scan, policy))]
        from .independent_evidence import Votes
        support_votes = {}
        for row in db.execute('SELECT c.person,o.scan,o.seconds,o.quality FROM candidates c '
            f"JOIN observations o ON o.id={address(db, 'c')} "
            'WHERE c.scan=? AND c.policy=? ORDER BY o.cue,o.id', (scan, policy)):
            support_votes.setdefault(row['person'], Votes()).add(row)
        support_counts = {person:len(votes) for person,votes in support_votes.items()}
        with tracking(
            people,
            'Resolve identity support',
            unit='person',
        ) as tracked:
            for person in tracked:
                group = by_id[person]
                support = support_counts.get(person, 0)
                accepted = group.known or support >= settings['min_identity_support']
                status = 'accepted' if accepted else 'withheld'
                db.execute('UPDATE candidates SET status=? WHERE scan=? AND policy=? AND person=?',
                           (status, scan, policy, person))
                if accepted:
                    token = f'stash-ai-recognition:{database.installation}:{person}'
                    db.execute('INSERT OR IGNORE INTO people VALUES (?,?,?,?,?,?)',
                               (person, record['scope'], record['namespace'],
                                getattr(database, 'identity_clock', time.time)(), None, token))
                    db.execute('INSERT OR IGNORE INTO members SELECT '
                               f"{address(db)},person FROM candidates "
                               'WHERE scan=? AND policy=? AND person=?', (scan, policy, person))
        # Include the members accepted above, so a stable next invocation does
        # not invalidate its own cache merely by having established a person.
        # Release every owner of the old reference set before resetting its
        # accounting and allocating the database replacement. Query scratch is
        # one bounded vector, distinct from retained representative storage.
        from .replay_memory import release_vectors
        release_vectors()
        groups.clear()
        by_id.clear()
        ranked.clear()
        group = query = observation = None
        retained = _groups(db, record['scope'], record['namespace'], grouping, usage=usage, stage='retained')
        if not independent and supported_merges(database, scan, config, retained, policy, usage=usage):
            retained.clear()
            retained = _groups(db, record['scope'], record['namespace'], grouping, usage=usage, stage='retained')
            _recheck_after_merge(db, scan, policy, retained, config)
            retained.clear()
            retained = _groups(db, record['scope'], record['namespace'], grouping, usage=usage, stage='retained')
        if not _cold_replay:
            context = reference_context(retained)
            db.execute("UPDATE candidates SET payload=json_set(payload,'$.reference_context',?) "
                       'WHERE scan=? AND policy=?', (context, scan, policy))
    return _result(db, scan, policy)


@measured('recheck')
def _recheck_after_merge(db, scan, policy, groups, config):
    """Recheck strict anchor observations only; auxiliary views are deferred."""
    from .pose_policy import anchor_sql
    anchor = anchor_sql('o', config.data['clustering'])
    settings = config.data['recognition']
    cache = evidence(db)
    with tracking(
        scored_rows(db.execute(
            'SELECT o.*,m.person AS prior_person FROM observations o LEFT JOIN members m '
            f'ON m.observation=o.id WHERE o.scan=? AND o.rejected IS NULL AND {anchor} '
            'ORDER BY o.cue,o.geometry,o.embedding,o.id',
            (scan,),
        ), groups, settings['match_threshold'] - settings['ambiguity_margin'],
            config.data['clustering']['cohesion_threshold']),
        'Recheck merged observations',
        unit='observation',
        total=lambda: db.execute(
            f'SELECT COUNT(*) FROM observations o WHERE scan=? AND rejected IS NULL AND {anchor}',
            (scan,),
        ).fetchone()[0],
    ) as tracked:
        for row, frozen in tracked:
            query = cache.vector(row) if cache is not None else vector(row)
            ranked = []
            for group, mean, minimum in contributions(groups, query, frozen, len(groups),
                    settings['match_threshold'] - settings['ambiguity_margin'],
                    config.data['clustering']['cohesion_threshold'],
                    exact=(lambda g,q: cache.scores(g,q,row['id'])) if cache is not None else None):
                if (mean >= settings['match_threshold'] - settings['ambiguity_margin'] and
                        minimum >= config.data['clustering']['cohesion_threshold']):
                    best_two(ranked, (mean, group.identifier, group))
            person, status, reason = None, 'withheld', 'insufficient_match_or_cohesion'
            if ranked and ranked[0][0] >= settings['match_threshold']:
                if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < settings['ambiguity_margin']:
                    reason = 'ambiguous_identity'
                else:
                    person, status, reason = ranked[0][1], 'accepted', 'known_identity'
                    if not co_presence_allows(db, row, ranked[0][2], groups, query, config):
                        person, status, reason = None, 'withheld', 'co_present_identity_conflict'
                    elif row['prior_person'] is not None and row['prior_person'] != person:
                        person, status, reason = None, 'withheld', 'policy_reassignment_requires_review'
                    else:
                        db.execute('INSERT OR IGNORE INTO members VALUES (?,?)', (row['id'], person))
            if person is None:
                db.execute('DELETE FROM members WHERE observation=?', (row['id'],))
            candidate = str(uuid.uuid5(uuid.UUID(row['id']), policy))
            db.execute("UPDATE candidates SET person=?,status=?,payload=json_set(payload,'$.reason',?,'$.scores',json(?)) "
                       'WHERE id=?', (person, status, reason, encoded([r[0] for r in ranked[:2]]), candidate))
