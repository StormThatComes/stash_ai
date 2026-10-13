"""Reconcile owned Stash performer merges from durable SQLite identity aliases."""
import json

from .publication import operation_id, perform, PendingRecognition
from .stash import checked_id, owns

COMPLETE = ('confirmed', 'observed', 'local')


def aliases(db, person):
    return db.execute('''WITH RECURSIVE aliases(id) AS (
      SELECT source FROM identity_merges WHERE target=?
      UNION SELECT m.source FROM identity_merges m JOIN aliases a ON m.target=a.id)
      SELECT p.*,m.status AS merge_status FROM people p JOIN aliases a ON p.id=a.id
      JOIN identity_merges m ON m.source=p.id ORDER BY p.created DESC,p.id DESC''', (person,)).fetchall()


def scene_merge_roots(db, scope, scan):
    """Pending aliases supported by this selected scan, including ambiguous cues."""
    from .identity_merges import canonical
    selected = {row[0] for row in db.execute('SELECT DISTINCT person FROM members m '
                'JOIN observations o ON o.id=m.observation WHERE o.scan=?', (scan,))}
    result = set()
    for row in db.execute('SELECT m.target,m.evidence FROM identity_merges m '
                          'JOIN people p ON p.id=m.source JOIN scans s ON s.namespace=p.namespace '
                          'WHERE p.scope=? AND s.id=? AND s.scope=p.scope '
                          "AND m.status NOT IN ('confirmed','observed','local')", (scope, scan)):
        root = canonical(db, row['target'])
        if root in selected or json.loads(row['evidence'])['scan'] == scan:
            result.add(root)
    return result


def _complete(database, source, status):
    with database.connection:
        database.connection.execute('UPDATE identity_merges SET status=? WHERE source=?', (status, source))


def _reconcile(database, client, previous):
    """A missing source alone cannot prove a merge; require our carried marker."""
    payload = json.loads(previous['payload'])
    expected = json.loads(previous['precondition'])
    matches = client.find_token(payload['marker'])
    if (len(matches) != 1 or not owns(matches[0], expected['source_token'])
            or not owns(matches[0], expected['destination_token'])
            or client.lookup_performer(payload['source']) is not None):
        raise PendingRecognition('Earlier performer merge is unresolved; no blind merge retry.')
    with database.connection:
        database.connection.execute("UPDATE publications SET status='observed',error=NULL WHERE id=?",
                                    (previous['id'],))


def sync_merges(database, client, report, scope, person, performer, *, counts=None):
    db = database.connection
    winner = db.execute('SELECT * FROM people WHERE id=? AND scope=?', (person, scope)).fetchone()
    counts = {} if counts is None else counts
    for source in aliases(db, person):
        if source['scope'] != scope or source['namespace'] != winner['namespace']:
            raise ValueError('Recognition merge crosses an instance or embedding namespace.')
        if (winner['created'], person) >= (source['created'], source['id']):
            raise ValueError('Recognition merge destination is not the oldest identity.')
        if source['merge_status'] in COMPLETE:
            continue
        identifier = operation_id(scope, '', source['id'], 'merge')
        previous = db.execute('SELECT * FROM publications WHERE id=?', (identifier,)).fetchone()
        if previous and previous['status'] in ('pending', 'unknown'):
            _reconcile(database, client, previous)
            _complete(database, source['id'], 'observed')
            counts['identity_merges_observed'] = counts.get('identity_merges_observed', 0) + 1
            continue
        if previous and previous['status'] in ('confirmed', 'observed'):
            _complete(database, source['id'], previous['status'])
            counts['identity_merges_observed'] = counts.get('identity_merges_observed', 0) + 1
            continue
        matches = client.find_token(source['token'])
        if len(matches) > 1:
            raise PendingRecognition('Merge source ownership token resolves to multiple Stash performers.')
        if not matches:
            creation = db.execute('SELECT status FROM publications WHERE id=?',
                                 (operation_id(scope, '', source['id'], 'create'),)).fetchone()
            if source['performer'] or creation and creation['status'] != 'failed':
                raise PendingRecognition('Merge source profile/creation is unresolved; preserved for reconciliation.')
            _complete(database, source['id'], 'local')
            continue
        live_source = checked_id(matches[0])
        if source['performer'] and live_source['id'] != source['performer']:
            raise PendingRecognition('Merge source performer binding changed.')
        if live_source['id'] == performer:
            raise PendingRecognition('Source token already occurs on destination without a recorded merge.')
        # Fresh reads of both owned profiles precede this destructive operation.
        live_source = checked_id(client.performer(live_source['id']), live_source['id'])
        live_winner = checked_id(client.performer(performer), performer)
        if not owns(live_source, source['token']) or not owns(live_winner, winner['token']):
            raise PendingRecognition('Performer ownership changed before identity merge.')
        marker = f'stash-ai-recognition-merge:{database.installation}:{source["id"]}:{person}'
        # Preserve notes and all inherited ownership/merge tokens. Scalar fields
        # (including name/image) stay those of the oldest destination profile.
        details = '\n\n'.join([live_winner.get('details') or '', live_source.get('details') or '', marker])
        def merge():
            checked_id(client.merge_person(live_source['id'], performer, details), performer)
            confirmed = checked_id(client.performer(performer), performer)
            if (not owns(confirmed, winner['token']) or not owns(confirmed, source['token'])
                    or not owns(confirmed, marker) or client.lookup_performer(live_source['id']) is not None):
                raise PendingRecognition('Performer merge acknowledgement was not confirmed by fresh reads.')
        perform(database, client, report, scope=scope, scene='', person=source['id'], kind='merge', target=performer,
                payload={'source': live_source['id'], 'destination': performer, 'marker': marker},
                precondition={'source_token': source['token'], 'destination_token': winner['token'],
                              'source_details': live_source.get('details'), 'destination_details': live_winner.get('details')},
                callback=merge)
        _complete(database, source['id'], 'confirmed')
        counts['identities_merged_in_stash'] = counts.get('identities_merged_in_stash', 0) + 1
    return counts


def canonical_targets(database, ready):
    """Remap earlier scenes in a multi-scene run before names/portraits execute."""
    from .identity_merges import canonical
    db, result = database.connection, {}
    for target in ready.values():
        person = canonical(db, target['person'])
        row = db.execute('SELECT performer FROM people WHERE id=?', (person,)).fetchone()
        if row is None or not row['performer'] or any(r['merge_status'] not in COMPLETE for r in aliases(db, person)):
            continue
        current = result.setdefault(person, {'person': person, 'performer': row['performer'], 'scans': []})
        for scan in target['scans']:
            if scan not in current['scans']:
                current['scans'].append(scan)
    return [result[key] for key in sorted(result)]
