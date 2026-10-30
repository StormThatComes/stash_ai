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
        from .person_signal import copy_evidence
        copy_evidence(source,target,row['id'])
        from .pixel_storage import copy_rows
        copy_rows(source,target,source.execute('SELECT quality FROM observations WHERE scan=?',(row['id'],)))
    from .scan_heads import copy_heads
    copy_heads(source, target)
    target.executemany('INSERT INTO people VALUES (?,?,?,?,?,?)',
        source.execute('SELECT * FROM people WHERE scope=? AND namespace=? ORDER BY rowid', (scope, namespace)))
    scan_query = ('SELECT id FROM scans WHERE scope=? AND namespace=? '
                  'AND scene NOT IN (SELECT value FROM json_each(?))')
    people_query = 'SELECT id FROM people WHERE scope=? AND namespace=?'
    observation_query = f'SELECT id FROM observations WHERE scan IN ({scan_query})'
    body_query = f'SELECT id FROM body_observations WHERE scan IN ({scan_query})'
    scan_args, people_args = (scope, namespace, excluded), (scope, namespace)
    rules = {
        'members': (f'observation IN ({observation_query}) AND person IN ({people_query})', scan_args+people_args),
        'candidates': (f'scan IN ({scan_query})', scan_args),
        'identity_origins': (f'observation IN ({observation_query}) AND person IN ({people_query})', scan_args+people_args),
        'identity_merges': (f'source IN ({people_query}) AND target IN ({people_query})', people_args*2),
        'merge_support': (f'observation IN ({observation_query}) AND source IN ({people_query}) AND target IN ({people_query})', scan_args+people_args*2),
    }
    from .body_identity import available as body_available
    if body_available(source):
        rules.update(body_members=(f'observation IN ({body_query}) AND (person IS NULL OR person IN ({people_query}))', scan_args+people_args),
                     body_origins=(f'observation IN ({body_query}) AND person IN ({people_query})', scan_args+people_args))
    for table, (where, args) in rules.items():
        columns = len(target.execute(f'PRAGMA table_info({table})').fetchall())
        target.executemany(f'INSERT INTO {table} VALUES (' + ','.join('?' for _ in range(columns)) + ')',
            source.execute(f'SELECT * FROM {table} WHERE {where} ORDER BY rowid', args))
    target.commit()
    return dict(scenes=target.execute('SELECT COUNT(DISTINCT scene) FROM scans').fetchone()[0],
                observations=target.execute('SELECT COUNT(*) FROM observations').fetchone()[0],
                people=target.execute('SELECT COUNT(*) FROM people').fetchone()[0])


def snapshot(target, config, labels):
    from collections import Counter
    import json
    from .clustering import matching_policy
    from .decision_evidence import recorded, explanation
    members = dict(target.execute('SELECT observation,person FROM members'))
    from .body_reports import members as body_members
    members.update(body_members(target,config))
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
    from .body_identity import available,policy
    if available(target):
        for row in target.execute('SELECT * FROM body_members WHERE policy=?',(policy(config),)):
            if row['observation'] not in labels:continue
            proof=json.loads(row['proof']);reasons[proof.get('reason','unknown')]+=1
            decisions[row['observation']]=dict(status=row['status'],identity=row['person'],evidence=proof,
                evidence_kind='person',explanation='Recorded independent person decision.')
    for identifier in labels.keys() - decisions.keys():
        row=target.execute('SELECT rejected FROM observations WHERE id=?',(identifier,)).fetchone()
        if row is None and available(target):row=target.execute('SELECT rejected FROM body_observations WHERE id=?',(identifier,)).fetchone()
        rejected=row[0] if row is not None else 'missing_observation'
        decisions[identifier] = dict(status='rejected' if rejected else 'withheld', identity=None,
            evidence=dict(observation=identifier, reason=rejected or 'no_assignment_decision'),
            explanation='No accepted production decision for this retained observation.')
    return predictions, decisions, dict(sorted(reasons.items()))
