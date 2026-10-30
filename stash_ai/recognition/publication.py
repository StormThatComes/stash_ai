"""SQLite intent-before-dispatch and reconciliation without uncertain retries."""
import json
import uuid
from contextlib import contextmanager

from stash_ai.runtime.mutations import write
from .diagnostics import message as safe_error
from .database import encoded
from .stash import owns, checked_id, checked_token, RecognitionStash


LINK_INPUT_CONTRACT = 'bulk-scene-update-ids-v1'


def legacy_rejected_link(row, client, performer):
    """Only the old malformed BulkUpdateIds.values request plus recorded 400.

    That legacy request cannot execute on the supported Stash schema. New link
    intents carry an input contract so a later unrelated 400 cannot qualify.
    Other uncertain operations still require observation, never a blind retry.
    """
    import json
    from stash_ai.stash_api.transport import absolute_stash_url
    try:
        payload = json.loads(row['payload'])
    except (TypeError, ValueError):
        return False
    return (isinstance(client, RecognitionStash) and row['status'] == 'unknown'
            and row['target'] == performer
            and payload == {'performer': performer}
            and row['error'] == '400 Client Error: Bad Request for url: '
            + absolute_stash_url('graphql', client.url))


class PendingRecognition(RuntimeError):
    pass


@contextmanager
def before_dispatch():
    """A failed read/precondition before HTTP cannot leave an uncertain write."""
    try:yield
    except BaseException as error:
        error.write_outcome='failed'
        raise


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


def ensure_person(database, client, report, scope, scene, person,*,guard=None):
    if guard:guard()
    from .identity_merges import canonical
    if canonical(database.connection, person) != person:
        raise PendingRecognition('Merged identity cannot recreate its former Stash profile.')
    row = database.connection.execute('SELECT * FROM people WHERE id=? AND scope=?', (person, scope)).fetchone()
    if row is None:
        raise ValueError('Accepted recognition identity is missing or belongs to another Stash instance.')
    checked_token(database, person, row['token'])
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
            with before_dispatch():
                if guard:guard()
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


def publish_person(database, client, report, scope, scene, person, *, merge_counts=None, reconcile_merges=True,guard=None):
    performer = ensure_person(database, client, report, scope, scene, person,guard=guard)
    if reconcile_merges:
        from .merge_publication import sync_merges
        sync_merges(database, client, report, scope, person, performer, counts=merge_counts)
    live = client.scene(scene)
    existing = {checked_id(item)['id'] for item in live['performers']}
    identifier = operation_id(scope, scene, person, 'link')
    pending = database.connection.execute('SELECT * FROM publications WHERE id=?', (identifier,)).fetchone()
    if performer in existing:
        # A pre-existing manual link is not adopted. Legacy recorded links retain
        # their historical managed status; future observed reruns keep provenance.
        prior = json.loads(pending['payload']) if pending else {}
        managed = bool(pending and pending['status'] in ('pending', 'unknown', 'confirmed', 'observed')
                       and prior.get('managed', True) and prior.get('active', True))
        save_operation(database, scope=scope, scene=scene, person=person, kind='link', target=performer,
                       payload={'performer': performer, 'input_contract': LINK_INPUT_CONTRACT, 'managed': managed, 'active': True},
                       precondition=json.loads(pending['precondition']) if pending else {}, status='observed')
        return performer
    precondition = {'performers': sorted(existing)}
    if pending and pending['status'] in ('pending', 'unknown'):
        if not legacy_rejected_link(pending, client, performer):
            raise PendingRecognition('Earlier additive scene link is unresolved; no blind retry.')
        precondition['legacy_validation_failure'] = 'BulkUpdateIds.values rejected with HTTP 400'
    def link():
        with before_dispatch():
            if guard:guard()
        return client.add_scene_person(scene,performer)
    perform(database, client, report, scope=scope, scene=scene, person=person, kind='link', target=performer,
            payload={'performer': performer, 'input_contract': LINK_INPUT_CONTRACT, 'managed': True, 'active': True}, precondition=precondition,
            callback=link)
    # Refresh after acknowledgement; unrelated concurrently added performers
    # are preserved by ADD, not rewritten from a stale full list.
    confirmed = client.scene(scene)
    if performer not in {checked_id(item)['id'] for item in confirmed['performers']}:
        raise PendingRecognition('Scene link disappeared after acknowledgement; rerun to reconcile.')
    return performer
