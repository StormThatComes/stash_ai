"""Optional immutable, server-scoped operator names; no public sync command."""
from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
import unicodedata

from .config import parse_yaml
from .database import encoded
from .publication import operation_id, perform, PendingRecognition
from .stash import checked_id, owns
from .progress import tracking


@dataclass(frozen=True)
class Names:
    raw: bytes
    values: object

    def snapshot(self):
        return {'source': self.raw.decode('utf-8-sig'), 'sha256': hashlib.sha256(self.raw).hexdigest(),
                'performers': dict(self.values)}


def load_names(config, scope):
    if config.names_path is None:
        return Names(b'', MappingProxyType({}))
    with config.names_path.open('rb') as stream:
        raw = stream.read(131073)
    data = parse_yaml(raw)
    if not isinstance(data, dict) or set(data) != {'version', 'stash_scope', 'performers'}:
        raise ValueError('performer_names.yml requires only version, stash_scope and performers.')
    if type(data['version']) is not int or data['version'] != 1:
        raise ValueError('performer_names.yml version must be 1.')
    supplied = data['stash_scope']
    if not isinstance(supplied, str):
        raise ValueError('Name mapping stash_scope must be the Stash base URL or its recognition scope digest.')
    if supplied != scope:
        from .run import instance_scope
        if instance_scope(supplied) != scope:
            raise ValueError('Name mapping belongs to a different Stash instance.')
    values = data['performers']
    if not isinstance(values, dict) or len(values) > 1000:
        raise ValueError('Name mapping must contain at most 1000 quoted performer IDs.')
    for identifier, name in values.items():
        if (not isinstance(identifier, str) or not identifier.isascii() or not identifier.isdecimal()
                or int(identifier) <= 0 or str(int(identifier)) != identifier):
            raise ValueError('Quote each canonical positive Stash performer ID, for example "123".')
        if (not isinstance(name, str) or not name.strip() or len(name) > 200
                or name != name.strip() or any(unicodedata.category(c) in ('Cc', 'Cs') for c in name)):
            raise ValueError('Each performer name must be a nonempty string without controls (maximum 200 characters).')
    return Names(raw, MappingProxyType(values.copy()))


def plan_names(database, client, report, config, names, scope, targets):
    plan = {}
    # Validate the entire selected intersection before any name mutations.
    with tracking(
        targets,
        'Validate selected names',
        unit='person',
    ) as tracked:
        for target in tracked:
            identifier = target['performer']
            if identifier not in names.values:
                continue
            live = checked_id(client.performer(identifier), identifier)
            row = database.connection.execute('SELECT token FROM people WHERE id=? AND scope=?',
                                             (target['person'], scope)).fetchone()
            if row is None or not owns(live, row['token']) or not isinstance(live.get('name'), str):
                raise PendingRecognition('Selected name mapping target identity/name was not confirmed.')
            plan[identifier] = {'before': live['name'], 'after': names.values[identifier], 'token': row['token']}
    return plan


def sync_name(database, client, report, scope, target, plan):
    performer, person = target['performer'], target['person']
    if performer not in plan:
        return {}
    expected = plan[performer]
    identifier = operation_id(scope, '', person, 'name')
    previous = database.connection.execute('SELECT * FROM publications WHERE id=?', (identifier,)).fetchone()
    live = checked_id(client.performer(performer), performer)
    if not owns(live, expected['token']):
        raise PendingRecognition('Performer ownership changed before name synchronization.')
    if previous and previous['status'] in ('pending', 'unknown'):
        intent, precondition = json.loads(previous['payload']), json.loads(previous['precondition'])
        if (previous['target'] != performer or precondition.get('token') != expected['token']
                or live['name'] != intent.get('name')):
            raise PendingRecognition('Earlier name mutation is unresolved; original intent was not observed.')
        with database.connection:
            database.connection.execute("UPDATE publications SET status='observed',error=NULL WHERE id=?", (identifier,))
    if live['name'] == expected['after']:
        return {'names_unchanged': 1}
    if live['name'] != expected['before']:
        raise PendingRecognition('Concurrent name edit preserved; selected name plan is stale.')
    def update():
        value = checked_id(client.update_person(performer, name=expected['after']), performer)
        if value.get('name') != expected['after'] or not owns(value, expected['token']):
            raise PendingRecognition('Name update result was not confirmed.')
        return value
    perform(database, client, report, scope=scope, scene='', person=person, kind='name', target=performer,
            payload={'name': expected['after']}, precondition={'name': expected['before'], 'token': expected['token']},
            callback=update)
    return {'names_updated': 1}


def starter(client, targets, scope):
    import yaml
    values = {}
    with tracking(
        targets[:50],
        'Read names starter',
        unit='person',
        position=0,
    ) as tracked:
        for target in tracked:
            identifier = target['performer']
            values[identifier] = checked_id(client.performer(identifier), identifier)['name']
    return yaml.safe_dump({'version': 1, 'stash_scope': scope, 'performers': values},
                          allow_unicode=True, sort_keys=False)
