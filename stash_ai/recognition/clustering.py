"""Scoped, conservative incremental matching against retained face identities."""
from dataclasses import dataclass
import hashlib
import json
import time
import uuid

from .config import digest
from .database import encoded


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


def _groups(db, scope, namespace, policy):
    maximum = policy['max_people_per_lookup']
    rows = db.execute('SELECT id FROM people WHERE scope=? AND namespace=? ORDER BY id LIMIT ?',
                      (scope, namespace, maximum + 1)).fetchall()
    if len(rows) > maximum:
        raise ValueError('Face identity lookup exceeds configured person budget.')
    result = []
    reference_bytes = 0
    for person in rows:
        samples = db.execute('SELECT o.dimensions,o.embedding FROM observations o JOIN members m '
                             'ON m.observation=o.id WHERE m.person=? ORDER BY o.id LIMIT ?',
                             (person['id'], policy['max_samples_per_person'])).fetchall()
        if not samples:
            raise ValueError('Accepted face identity has no retained supporting embeddings.')
        vectors = [vector(row) for row in samples]
        reference_bytes += sum(item.nbytes for item in vectors)
        if reference_bytes > policy['max_reference_bytes']:
            raise ValueError('Retained face references exceed the configured RAM budget.')
        result.append(Group(person['id'], vectors, True))
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


def cluster_scene(database, scan, config):
    """Only selected observations change. Existing identities are read references.

    Cohesion uses a bounded representative sample, not an N-by-N matrix or a
    transitive single-link union. Approximation is explicit in policy provenance.
    """
    import numpy as np
    db = database.connection
    record = db.execute('SELECT * FROM scans WHERE id=?', (scan,)).fetchone()
    if record is None or record['status'] != 'complete':
        raise ValueError('Face matching requires a complete selected-scene scan.')
    policy = digest({'format': 1, 'recognition': config.data['recognition'],
                     'clustering': config.data['clustering']})
    count = db.execute('SELECT COUNT(*) FROM observations WHERE scan=? AND rejected IS NULL', (scan,)).fetchone()[0]
    if not count:
        return []
    settings, grouping = config.data['recognition'], config.data['clustering']
    groups = _groups(db, record['scope'], record['namespace'], grouping)
    context = reference_context(groups)
    existing = db.execute('SELECT COUNT(*) FROM candidates WHERE scan=? AND policy=? '
                          "AND json_extract(payload,'$.reference_context')=?", (scan, policy, context)).fetchone()[0]
    if existing == count and count:
        for row in db.execute('SELECT dimensions,embedding FROM observations WHERE scan=? AND rejected IS NULL', (scan,)):
            vector(row)
        return _result(db, scan, policy)
    reference_bytes = sum(v.nbytes for g in groups for v in g.samples)
    by_id = {g.identifier: g for g in groups}
    with db:
        db.execute('DELETE FROM candidates WHERE scan=? AND policy=?', (scan, policy))
        for observation in db.execute('SELECT * FROM observations WHERE scan=? AND rejected IS NULL '
                                       'ORDER BY cue,id', (scan,)):
            query = vector(observation)
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
                if not group.known and len(group.samples) < grouping['max_samples_per_person']:
                    reference_bytes += query.nbytes
                    if reference_bytes > grouping['max_reference_bytes']:
                        raise ValueError('Selected face references exceed the configured RAM budget.')
                    group.samples.append(query.copy())
            elif near:
                person, status, reason = None, 'withheld', 'insufficient_match_or_cohesion'
            else:
                if len(groups) >= grouping['max_people_per_lookup']:
                    raise ValueError('Selected-scene face groups exceed the configured budget.')
                person = str(uuid.uuid5(uuid.UUID(database.installation),
                                       encoded([record['scope'], record['namespace'], observation['id']])))
                group = Group(person, [query.copy()], False)
                reference_bytes += query.nbytes
                if reference_bytes > grouping['max_reference_bytes']:
                    raise ValueError('Selected face references exceed the configured RAM budget.')
                groups.append(group)
                by_id[person] = group
                status, reason = 'pending', 'candidate_group'
            candidate = str(uuid.uuid5(uuid.UUID(observation['id']), policy))
            db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?)',
                       (candidate, scan, policy, person, status,
                        encoded({'observation': observation['id'], 'reason': reason, 'scores': scores,
                                 'policy': policy, 'representative_limit': grouping['max_samples_per_person']})))
        people = [row[0] for row in db.execute('SELECT DISTINCT person FROM candidates '
                                                "WHERE scan=? AND policy=? AND status='pending'", (scan, policy))]
        for person in people:
            group = by_id[person]
            support = db.execute('SELECT COUNT(DISTINCT o.seconds) FROM candidates c JOIN observations o '
                                 "ON o.id=json_extract(c.payload,'$.observation') "
                                 'WHERE c.scan=? AND c.policy=? AND c.person=?', (scan, policy, person)).fetchone()[0]
            accepted = group.known or support >= settings['min_support']
            status = 'accepted' if accepted else 'withheld'
            db.execute('UPDATE candidates SET status=? WHERE scan=? AND policy=? AND person=?',
                       (status, scan, policy, person))
            if accepted:
                token = f'stash-ai-recognition:{database.installation}:{person}'
                db.execute('INSERT OR IGNORE INTO people VALUES (?,?,?,?,?,?)',
                           (person, record['scope'], record['namespace'], time.time(), None, token))
                db.execute('INSERT OR IGNORE INTO members SELECT '
                           "json_extract(payload,'$.observation'),person FROM candidates "
                           'WHERE scan=? AND policy=? AND person=?', (scan, policy, person))
        # Include the members accepted above, so a stable next invocation does
        # not invalidate its own cache merely by having established a person.
        context = reference_context(_groups(db, record['scope'], record['namespace'], grouping))
        db.execute("UPDATE candidates SET payload=json_set(payload,'$.reference_context',?) "
                   'WHERE scan=? AND policy=?', (context, scan, policy))
    return _result(db, scan, policy)
