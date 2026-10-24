"""Isolated evaluation inputs; retained history never includes future test scenes."""
import random

from .database import encoded


def ordered(scans, order, seed):
    result = list(scans)
    if order == 'reverse':
        result.reverse()
    elif order == 'shuffle':
        random.Random(seed).shuffle(result)
    elif order != 'numeric':
        raise ValueError('Unknown evaluation scene order.')
    return result


def copy_history(source, target, selected, scope, namespace):
    """Copy authoritative external evidence only, without images or run caches.

    Keep scan insertion order, including partial/newer scans that supersede old
    evidence. Never admit any version of a selected scene before its batch.
    """
    excluded = encoded([s['scene'] for s in selected])
    records = source.execute('SELECT * FROM scans WHERE scope=? AND namespace=? '
        'AND scene NOT IN (SELECT value FROM json_each(?)) ORDER BY rowid',
        (scope, namespace, excluded))
    for row in records:
        target.execute('INSERT INTO scans VALUES (?,?,?,?,?,?,?,?,?,?,?)', tuple(row))
        target.executemany('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
            source.execute('SELECT id,scan,cue,seconds,source,rect,geometry,quality,rejected,'
                           'dimensions,embedding,NULL FROM observations WHERE scan=? ORDER BY rowid', (row['id'],)))
    target.executemany('INSERT INTO people VALUES (?,?,?,?,?,?)',
        source.execute('SELECT * FROM people WHERE scope=? AND namespace=? ORDER BY rowid', (scope, namespace)))
    scans = {r[0] for r in target.execute('SELECT id FROM scans')}
    people = {r[0] for r in target.execute('SELECT id FROM people')}
    observations = {r[0] for r in target.execute('SELECT id FROM observations')}
    rules = {
        'members': lambda r: r['observation'] in observations and r['person'] in people,
        'candidates': lambda r: r['scan'] in scans,
        'identity_origins': lambda r: r['observation'] in observations and r['person'] in people,
        'identity_merges': lambda r: r['source'] in people and r['target'] in people,
        'merge_support': lambda r: r['observation'] in observations and r['source'] in people and r['target'] in people,
    }
    for table, allowed in rules.items():
        columns = len(target.execute(f'PRAGMA table_info({table})').fetchall())
        target.executemany(f'INSERT INTO {table} VALUES (' + ','.join('?' for _ in range(columns)) + ')',
            (tuple(r) for r in source.execute(f'SELECT * FROM {table} ORDER BY rowid') if allowed(r)))
    target.commit()
    return dict(scenes=len({r[0] for r in target.execute('SELECT scene FROM scans')}),
                observations=len(observations), people=len(people))


def snapshot(target, config, labels):
    from collections import Counter
    import json
    from .clustering import matching_policy
    from .decision_evidence import recorded, explanation
    members = dict(target.execute('SELECT observation,person FROM members'))
    predictions = {identifier: members.get(identifier) for identifier in labels}
    decisions, reasons = {}, Counter()
    for row in target.execute('SELECT * FROM candidates WHERE policy=? ORDER BY rowid', (matching_policy(config),)):
        payload = json.loads(row['payload'])
        identifier = payload.get('observation')
        if identifier not in labels:
            continue
        reasons[payload.get('reason', 'unknown')] += 1
        decisions[identifier] = dict(status=row['status'], identity=row['person'],
            evidence=recorded(payload), explanation=explanation(payload))
    for identifier in labels.keys() - decisions.keys():
        rejected = target.execute('SELECT rejected FROM observations WHERE id=?', (identifier,)).fetchone()[0]
        decisions[identifier] = dict(status='rejected' if rejected else 'withheld', identity=None,
            evidence=dict(observation=identifier, reason=rejected or 'no_assignment_decision'),
            explanation='No accepted production decision for this retained observation.')
    return predictions, decisions, dict(sorted(reasons.items()))
