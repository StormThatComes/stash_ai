"""SQLite intent-before-dispatch and reconciliation without uncertain retries."""
import uuid

from stash_ai.runtime.mutations import write
from stash_ai.stash_api.transport import safe_error
from .database import encoded
from .stash import owns, checked_id


class PendingRecognition(RuntimeError):
    pass


def operation_id(scope, scene, person, kind):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, encoded([scope, scene, person, kind])))


def save_operation(database, *, scope, scene, person, kind, target, payload, precondition,
                   status='pending', error=None):
    identifier = operation_id(scope, scene, person, kind)
    with database.connection:
        database.connection.execute('INSERT OR REPLACE INTO publications VALUES (?,?,?,?,?,?,?,?,?,?)',
                                    (identifier, scope, scene, person, kind, target,
                                     encoded(payload), encoded(precondition), status, error))
    return identifier


def perform(database, client, report, *, scope, scene, person, kind, target, payload, precondition, callback):
    identifier = save_operation(database, scope=scope, scene=scene, person=person, kind=kind,
                                target=target, payload=payload, precondition=precondition)
    try:
        result = write(report, 'recognition_' + kind, callback)
    except BaseException as error:
        status = 'failed' if getattr(error, 'write_outcome', None) == 'failed' else 'unknown'
        # Preserve the primary exception even if recovery metadata cannot commit.
        from stash_ai.runtime.cleanup import cleanup
        def record():
            with database.connection:
                database.connection.execute('UPDATE publications SET status=?,error=? WHERE id=?',
                    (status, safe_error(error, getattr(client, 'api_key', ''))[:1000], identifier))
        cleanup([('record recognition mutation outcome', record)])
        raise
    with database.connection:
        database.connection.execute("UPDATE publications SET status='confirmed',error=NULL WHERE id=?", (identifier,))
    return result


def ensure_person(database, client, report, scope, scene, person):
    row = database.connection.execute('SELECT * FROM people WHERE id=? AND scope=?', (person, scope)).fetchone()
    if row is None:
        raise ValueError('Accepted recognition identity is missing or belongs to another Stash instance.')
    if row['performer']:
        live = client.performer(row['performer'])
        if not owns(live, row['token']):
            raise PendingRecognition('Stored performer ownership changed; no duplicate profile was created.')
        return checked_id(live, row['performer'])['id']
    matches = client.find_token(row['token'])
    if len(matches) > 1:
        raise PendingRecognition('Multiple Stash performers have the recognition ownership token.')
    if matches:
        value = matches[0]
    else:
        # Creation identity is global to the person, not a particular scene:
        # another selected scene must not retry an earlier uncertain create.
        identifier = operation_id(scope, '', person, 'create')
        pending = database.connection.execute('SELECT status FROM publications WHERE id=?', (identifier,)).fetchone()
        if pending and pending['status'] in ('pending', 'unknown', 'confirmed'):
            raise PendingRecognition('Earlier performer creation is unresolved; no blind create retry.')
        name = 'AI Person ' + person.replace('-', '')[:12]
        def create():
            acknowledgement = checked_id(client.create_person(name, row['token']))
            confirmed = checked_id(client.performer(acknowledgement['id']), acknowledgement['id'])
            if not owns(confirmed, row['token']):
                raise PendingRecognition('Created performer ownership was not confirmed by a fresh read.')
            return confirmed
        value = perform(database, client, report, scope=scope, scene='', person=person, kind='create',
                        target=None, payload={'name': name, 'token': row['token']}, precondition={},
                        callback=create)
    checked_id(value)
    if not owns(value, row['token']):
        raise PendingRecognition('Resolved performer ownership was not confirmed.')
    with database.connection:
        database.connection.execute('UPDATE people SET performer=? WHERE id=?', (value['id'], person))
        identifier = operation_id(scope, '', person, 'create')
        database.connection.execute("UPDATE publications SET status='confirmed',target=?,error=NULL WHERE id=?",
                                    (value['id'], identifier))
    return value['id']


def publish_person(database, client, report, scope, scene, person):
    performer = ensure_person(database, client, report, scope, scene, person)
    live = client.scene(scene)
    existing = {checked_id(item)['id'] for item in live['performers']}
    identifier = operation_id(scope, scene, person, 'link')
    if performer in existing:
        save_operation(database, scope=scope, scene=scene, person=person, kind='link', target=performer,
                       payload={'performer': performer}, precondition={}, status='observed')
        return performer
    pending = database.connection.execute('SELECT status FROM publications WHERE id=?', (identifier,)).fetchone()
    if pending and pending['status'] in ('pending', 'unknown'):
        raise PendingRecognition('Earlier additive scene link is unresolved; no blind retry.')
    perform(database, client, report, scope=scope, scene=scene, person=person, kind='link', target=performer,
            payload={'performer': performer}, precondition={'performers': sorted(existing)},
            callback=lambda: client.add_scene_person(scene, performer))
    # Refresh after acknowledgement; unrelated concurrently added performers
    # are preserved by ADD, not rewritten from a stale full list.
    confirmed = client.scene(scene)
    if performer not in {checked_id(item)['id'] for item in confirmed['performers']}:
        raise PendingRecognition('Scene link disappeared after acknowledgement; rerun to reconcile.')
    return performer
