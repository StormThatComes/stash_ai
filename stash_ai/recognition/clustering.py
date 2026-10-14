"""Scoped, conservative incremental matching against retained face identities."""
from dataclasses import dataclass
import hashlib
import json
import time
import uuid

from .config import digest
from .database import encoded
from .reference_usage import ReferenceUsage
from .identity_merges import supported_merges
from .progress import tracking


def vector(row):
    import numpy as np
    dimensions, blob = row['dimensions'], row['embedding']
    if (type(dimensions) is not int or not 1 <= dimensions <= 1024 or
            not isinstance(blob, bytes) or len(blob) != 4 * dimensions):
        raise ValueError('Invalid retained face embedding shape/length.')
    result = np.frombuffer(blob, dtype='<f4')
    if not np.isfinite(result).all() or abs(float(np.linalg.norm(result)) - 1) > .001:
        raise ValueError('Invalid retained face embedding values/normalization.')
    return result


@dataclass
class Group:
    identifier: str
    samples: list
    known: bool


def _groups(db, scope, namespace, policy, *, usage=None, stage='loading'):
    usage = usage or ReferenceUsage(policy['max_reference_bytes'])
    usage.begin(stage, reset=True)
    maximum = policy['max_people_per_lookup']
    rows = db.execute('SELECT id FROM people WHERE scope=? AND namespace=? '
                      'AND NOT EXISTS (SELECT 1 FROM identity_merges WHERE source=people.id) '
                      'AND EXISTS (SELECT 1 FROM members WHERE person=people.id) '
                      'ORDER BY created,id LIMIT ?',
                      (scope, namespace, maximum + 1)).fetchall()
    if len(rows) > maximum:
        raise ValueError('Face identity lookup exceeds configured person budget.')
    result = []
    with tracking(
        rows,
        'Load identity references',
        unit='person',
    ) as tracked:
        for person in tracked:
            samples = db.execute('SELECT o.dimensions,o.embedding FROM observations o JOIN members m '
                                 'ON m.observation=o.id WHERE m.person=? ORDER BY o.id LIMIT ?',
                                 (person['id'], policy['max_samples_per_person'])).fetchall()
            if not samples:
                raise ValueError('Accepted face identity has no retained supporting embeddings.')
            vectors = [vector(row) for row in samples]
            usage.add(vectors, people=1, kind='Retained')
            result.append(Group(person['id'], vectors, True))
    usage.emit(force=True)
    return result


def _similarities(group, query):
    import numpy as np
    if any(sample.shape != query.shape for sample in group.samples):
        raise ValueError('Face namespace contains mixed embedding dimensions.')
    scores = np.asarray([float(np.dot(sample, query)) for sample in group.samples])
    return float(scores.mean()), float(scores.min())


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


def matching_policy(config):
    return digest({'format': 4, 'recognition': config.data['recognition'],
                   'clustering': config.data['clustering']})


def cluster_scene(database, scan, config, *, progress=None):
    """Selected evidence can merge compatible historical identities oldest-first.

    Cohesion uses a bounded representative sample, not an N-by-N matrix or a
    transitive single-link union. Approximation is explicit in policy provenance.
    """
    import numpy as np
    db = database.connection
    record = db.execute('SELECT * FROM scans WHERE id=?', (scan,)).fetchone()
    if record is None or record['status'] != 'complete':
        raise ValueError('Face matching requires a complete selected-scene scan.')
    policy = matching_policy(config)
    count = db.execute('SELECT COUNT(*) FROM observations WHERE scan=? AND rejected IS NULL', (scan,)).fetchone()[0]
    usage = ReferenceUsage(config.data['clustering']['max_reference_bytes'], progress)
    if not count:
        usage.begin('no_observations')
        return []
    settings, grouping = config.data['recognition'], config.data['clustering']
    groups = _groups(db, record['scope'], record['namespace'], grouping, usage=usage)
    context = reference_context(groups)
    existing = db.execute('SELECT COUNT(*) FROM candidates WHERE scan=? AND policy=? '
                          "AND json_extract(payload,'$.reference_context')=?", (scan, policy, context)).fetchone()[0]
    if existing == count and count:
        with tracking(
            db.execute(
                'SELECT dimensions,embedding FROM observations WHERE scan=? AND rejected IS NULL',
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
        by_id = {g.identifier: g for g in groups}
        db.execute('DELETE FROM candidates WHERE scan=? AND policy=?', (scan, policy))
        with tracking(
            db.execute(
                'SELECT * FROM observations WHERE scan=? AND rejected IS NULL ORDER BY cue,id',
                (scan,),
            ),
            'Match face observations',
            unit='observation',
            total=count,
        ) as tracked:
            for observation in tracked:
                query = vector(observation)
                previous = db.execute("SELECT person FROM members WHERE observation=?", (observation["id"],)).fetchone()
                prior_person = previous[0] if previous is not None else None
                ranked = []
                near = False
                for group in groups:
                    score, cohesion = _similarities(group, query)
                    near |= score >= settings['match_threshold'] - settings['ambiguity_margin']
                    if (score >= settings['match_threshold'] - settings['ambiguity_margin'] and
                            cohesion >= grouping['cohesion_threshold']):
                        ranked.append((score, group.identifier, group))
                ranked.sort(key=lambda item: (-item[0], item[1]))
                scores = [item[0] for item in ranked[:2]]
                eligible = bool(ranked and ranked[0][0] >= settings['match_threshold'])
                if eligible and len(ranked) > 1 and ranked[0][0] - ranked[1][0] < settings['ambiguity_margin']:
                    person, status, reason = None, 'withheld', 'ambiguous_identity'
                elif eligible:
                    group = ranked[0][2]
                    person, status, reason = group.identifier, 'pending', 'known_identity' if group.known else 'candidate_group'
                    if prior_person is not None and prior_person != person:
                        person, status, reason = None, 'withheld', 'policy_reassignment_requires_review'
                    if person is not None and not group.known and len(group.samples) < grouping['max_samples_per_person']:
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
                    group = Group(person, [query], False)
                    groups.append(group)
                    by_id[person] = group
                    status, reason = 'pending', 'candidate_group'
                candidate = str(uuid.uuid5(uuid.UUID(observation['id']), policy))
                db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?)',
                           (candidate, scan, policy, person, status,
                            encoded({'observation': observation['id'], 'reason': reason, 'scores': scores,
                                     'policy': policy, 'representative_limit': grouping['max_samples_per_person']})))
        usage.emit(force=True)
        people = [row[0] for row in db.execute('SELECT DISTINCT person FROM candidates '
                                                "WHERE scan=? AND policy=? AND status='pending'", (scan, policy))]
        with tracking(
            people,
            'Resolve identity support',
            unit='person',
        ) as tracked:
            for person in tracked:
                group = by_id[person]
                support = db.execute('SELECT COUNT(DISTINCT o.seconds) FROM candidates c JOIN observations o '
                                     "ON o.id=json_extract(c.payload,'$.observation') "
                                     'WHERE c.scan=? AND c.policy=? AND c.person=?', (scan, policy, person)).fetchone()[0]
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
                               "json_extract(payload,'$.observation'),person FROM candidates "
                               'WHERE scan=? AND policy=? AND person=?', (scan, policy, person))
        # Include the members accepted above, so a stable next invocation does
        # not invalidate its own cache merely by having established a person.
        # Release every owner of the old reference set before resetting its
        # accounting and allocating the database replacement. Query scratch is
        # one bounded vector, distinct from retained representative storage.
        groups.clear()
        by_id.clear()
        ranked.clear()
        group = query = observation = None
        retained = _groups(db, record['scope'], record['namespace'], grouping, usage=usage, stage='retained')
        if supported_merges(database, scan, config, retained, policy, usage=usage):
            retained.clear()
            retained = _groups(db, record['scope'], record['namespace'], grouping, usage=usage, stage='retained')
            _recheck_after_merge(db, scan, policy, retained, config)
            retained.clear()
            retained = _groups(db, record['scope'], record['namespace'], grouping, usage=usage, stage='retained')
        context = reference_context(retained)
        db.execute("UPDATE candidates SET payload=json_set(payload,'$.reference_context',?) "
                   'WHERE scan=? AND policy=?', (context, scan, policy))
    return _result(db, scan, policy)


def _recheck_after_merge(db, scan, policy, groups, config):
    """Resolve ambiguity against the surviving groups, preserving raw evidence."""
    settings = config.data['recognition']
    with tracking(
        db.execute(
            'SELECT * FROM observations WHERE scan=? AND rejected IS NULL ORDER BY cue,id',
            (scan,),
        ),
        'Recheck merged observations',
        unit='observation',
        total=lambda: db.execute(
            'SELECT COUNT(*) FROM observations WHERE scan=? AND rejected IS NULL',
            (scan,),
        ).fetchone()[0],
    ) as tracked:
        for row in tracked:
            query = vector(row)
            ranked = []
            for group in groups:
                mean, minimum = _similarities(group, query)
                if (mean >= settings['match_threshold'] - settings['ambiguity_margin'] and
                        minimum >= config.data['clustering']['cohesion_threshold']):
                    ranked.append((mean, group.identifier))
            ranked.sort(key=lambda item: (-item[0], item[1]))
            person, status, reason = None, 'withheld', 'insufficient_match_or_cohesion'
            if ranked and ranked[0][0] >= settings['match_threshold']:
                if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < settings['ambiguity_margin']:
                    reason = 'ambiguous_identity'
                else:
                    person, status, reason = ranked[0][1], 'accepted', 'known_identity'
                    previous = db.execute('SELECT person FROM members WHERE observation=?', (row['id'],)).fetchone()
                    if previous is not None and previous[0] != person:
                        person, status, reason = None, 'withheld', 'policy_reassignment_requires_review'
                    else:
                        db.execute('INSERT OR IGNORE INTO members VALUES (?,?)', (row['id'], person))
            candidate = str(uuid.uuid5(uuid.UUID(row['id']), policy))
            db.execute("UPDATE candidates SET person=?,status=?,payload=json_set(payload,'$.reason',?,'$.scores',json(?)) "
                       'WHERE id=?', (person, status, reason, encoded([r[0] for r in ranked[:2]]), candidate))
