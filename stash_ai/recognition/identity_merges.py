"""Evidence-backed, oldest-first aliases; never a transitive single-link union."""
import time

from .database import encoded


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


def _compatible(first, second, config):
    # Keep the original bounded references for every constituent until this
    # pass finishes. Capping a union early could hide an incompatible endpoint.
    worst_mean = worst_minimum = 1.
    for left in first:
        for right in second:
            scores = _cross(left, right, config)
            if scores is None:
                return None
            worst_mean = min(worst_mean, scores['mean'])
            worst_minimum = min(worst_minimum, scores['minimum'])
    return {'minimum_cross_mean': worst_mean, 'minimum_cross_similarity': worst_minimum}


def _cooccurs(db, first, second):
    return db.execute('''SELECT 1 FROM members a JOIN observations x ON x.id=a.observation
      JOIN observations y ON y.scan=x.scan AND y.cue=x.cue
      JOIN members b ON b.observation=y.id WHERE a.person=? AND b.person=?
      AND x.id<>y.id AND x.rejected IS NULL AND y.rejected IS NULL LIMIT 1''', (first, second)).fetchone() is not None


def supported_merges(database, scan, config, groups, policy, *, usage=None):
    """One nomination per identity, bounded references, distinct selected times.

    Only a selected observation matching both endpoints can support a merge.
    Every constituent pair must pass mean match and minimum cohesion; a weak
    bridge between incompatible identities stays ambiguous. No all-person
    distance matrix or pair-support dictionary is built.
    Caller owns the transaction, including member/candidate transfers.
    """
    from .clustering import _similarities, vector, reference_context
    db = database.connection
    if len(groups) < 2:
        return 0
    if usage is not None:
        usage.begin('merging')
    rows = {row['id']: row for row in db.execute(
        'SELECT p.id,p.created FROM people p JOIN scans s ON s.scope=p.scope AND s.namespace=p.namespace '
        'WHERE s.id=? AND NOT EXISTS (SELECT 1 FROM identity_merges WHERE source=p.id)', (scan,))}
    order = lambda person: (rows[person]['created'], person)
    roots = {g.identifier: g.identifier for g in groups}
    components = {g.identifier: [g] for g in groups}
    nominations = {}
    merged = 0
    settings = config.data['recognition']
    for observation in db.execute('SELECT id,seconds,dimensions,embedding FROM observations '
                                   'WHERE scan=? AND rejected IS NULL ORDER BY cue,id', (scan,)):
        if usage is not None:
            usage.emit()
        query = vector(observation)
        eligible = []
        for root, component in components.items():
            if all((scores := _similarities(group, query))[0] >= settings['match_threshold']
                   and scores[1] >= config.data['clustering']['cohesion_threshold'] for group in component):
                eligible.append(root)
        eligible.sort(key=order)
        for index, source in enumerate(eligible[1:], 1):
            source = roots[source]
            for target in eligible[:index]:
                target = roots[target]
                if source == target:
                    continue
                target, source = sorted((target, source), key=order)
                scores = _compatible(components[target], components[source], config)
                if scores is None:
                    continue
                previous = nominations.get(source)
                if previous and roots[previous['target']] != target:
                    if order(roots[previous['target']]) < order(target):
                        break
                    previous = None
                nomination = previous or {'target': target, 'times': set()}
                if len(nomination['times']) < settings['min_support']:
                    nomination['times'].add(observation['seconds'])
                nominations[source] = nomination
                if len(nomination['times']) >= settings['min_support']:
                    if _cooccurs(db, target, source):
                        # Distinct detected faces in one cue are negative
                        # evidence, even when their embeddings look similar.
                        break
                    evidence = {'format': 1, 'scan': scan, 'policy': policy,
                        'support_seconds': sorted(nomination['times']), 'scores': scores,
                        'representative_limit': config.data['clustering']['max_samples_per_person'],
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
    if usage is not None:
        usage.begin('retained')
    return merged
