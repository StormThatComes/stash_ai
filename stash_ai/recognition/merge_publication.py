"""Reconcile owned Stash performer merges from durable SQLite identity aliases."""
import json

from .publication import operation_id, perform, PendingRecognition
from .stash import checked_id, checked_token, owns
from .progress import tracking

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
    checked_token(database, previous['person'], expected['source_token'])
    destination = database.connection.execute('SELECT id FROM people WHERE scope=? AND token=?',
                                               (previous['scope'], expected['destination_token'])).fetchone()
    if destination is None:
        raise PendingRecognition('Earlier performer merge destination ownership is unavailable.')
    checked_token(database, destination['id'], expected['destination_token'])
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
    checked_token(database, person, winner['token'])
    counts = {} if counts is None else counts
    for source in aliases(db, person):
        if source['scope'] != scope or source['namespace'] != winner['namespace']:
            raise ValueError('Recognition merge crosses an instance or embedding namespace.')
        checked_token(database, source['id'], source['token'])
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


def _target_context(db, identifiers):
    """Read only the alias forest reachable from the requested targets."""
    from .database import encoded
    identifiers = list(identifiers)
    edges, roots = {}, {}
    for start in range(0, len(identifiers), 256):
        selected = encoded(identifiers[start:start + 256])
        for row in db.execute('''WITH RECURSIVE ancestors(id) AS (
          SELECT value FROM json_each(?)
          UNION SELECT m.target FROM identity_merges m JOIN ancestors a ON m.source=a.id)
          SELECT m.source,m.target FROM identity_merges m JOIN ancestors a ON m.source=a.id''', (selected,)):
            edges[row['source']] = row['target']
    for identifier in identifiers:
        person, path, seen = identifier, [], set()
        while person in edges and person not in roots:
            if person in seen:
                raise ValueError('Recognition identity aliases contain a cycle.')
            seen.add(person)
            path.append(person)
            person = edges[person]
        person = roots.get(person, person)
        roots[identifier] = person
        for item in path:
            roots[item] = person
    people, blocked = {}, set()
    unique = sorted(set(roots.values()))
    for start in range(0, len(unique), 256):
        selected = encoded(unique[start:start + 256])
        people.update((row['id'], row['performer']) for row in db.execute(
            'SELECT id,performer FROM people WHERE id IN (SELECT value FROM json_each(?))', (selected,)))
        blocked.update(row[0] for row in db.execute('''WITH RECURSIVE descendants(root,id) AS (
          SELECT value,value FROM json_each(?)
          UNION SELECT a.root,m.source FROM identity_merges m JOIN descendants a ON m.target=a.id)
          SELECT DISTINCT a.root FROM descendants a JOIN identity_merges m ON m.source=a.id
          WHERE m.status NOT IN ('confirmed','observed','local')''', (selected,)))
    return roots, people, blocked


def canonical_targets(database, ready, *, on_blocked=None):
    """Remap earlier scenes in a multi-scene run before names/portraits execute."""
    db, result = database.connection, {}
    roots, performers, blocked = _target_context(db, (target['person'] for target in ready.values()))
    for person in sorted(blocked.intersection(roots.values())):
        error = PendingRecognition('Recognition performer merge remains unresolved; metadata withheld for identity ' + person + '.')
        if on_blocked is None:
            raise error
        on_blocked(person, error)
    seen = {}
    with tracking(
        ready.values(),
        'Resolve metadata targets',
        unit='person',
        position=0,
    ) as tracked:
        for target in tracked:
            person = roots[target['person']]
            performer = performers.get(person)
            if not performer or person in blocked:
                continue
            current = result.setdefault(person, {'person': person, 'performer': performer, 'scans': []})
            known = seen.setdefault(person, set())
            if target.get('matching'):
                current.setdefault('matching', {}).update(target['matching'])
            for scan in target['scans']:
                key = scan['scene'], scan['scan']
                if key not in known:
                    known.add(key)
                    current['scans'].append(scan)
    return [result[key] for key in sorted(result)]
