"""Converge an owned Stash portrait to absence after a complete policy rejection."""
import json

from .database import encoded
from .media import safe_url
from .publication import PendingRecognition, perform
from .stash import checked_id, owns


def retry_precondition(previous):
    if previous is None or previous['status'] not in ('pending', 'unknown'):
        return {}
    # One bounded predecessor snapshot, including set and removal intents.
    return {'retry_of': {'id': previous['id'], 'status': previous['status'],
                        'payload': json.loads(previous['payload']),
                        'image_sha256': json.loads(previous['precondition']).get('image_sha256'),
                        'error': previous['error']}}


def selection_incomplete(database, scope, target, policy_sha):
    scans = [item['scan'] for item in target['scans']]
    row = database.connection.execute('''SELECT 1 FROM portraits WHERE scope=? AND person=?
      AND json_extract(metadata,'$.policy_sha')=?
      AND json_extract(metadata,'$.scan') IN (SELECT value FROM json_each(?))
      AND json_extract(metadata,'$.reason') IN
        ('vtt_source_unavailable_or_changed','video_unavailable_or_budget_exceeded',
         'portrait_identity_unverified','face_visibility_unavailable','face_expression_unavailable') LIMIT 1''',
      (scope, target['person'], policy_sha, encoded(scans))).fetchone()
    return row is not None


def remove_owned(database, client, report, config, scope, target, token, baseline, previous, *, reason):
    from .portraits import image_digest, missing_image, policy_identity
    performer, person = target['performer'], target['person']
    live = checked_id(client.performer(performer), performer)
    if not owns(live, token):
        raise PendingRecognition('Portrait target ownership changed before removal.')
    if missing_image(live, client, performer):
        return {'portraits_absent': 1}
    if image_digest(client, live, performer, config) != baseline:
        return {'portraits_preserved': 1}
    intent = {'action': 'remove', 'portrait': json.loads(previous['payload'])['portrait'],
              'sha256': None, 'policy_sha': policy_identity(config), 'reason': reason}
    precondition = {'missing': False, 'image_sha256': baseline, 'token': token,
                    'image_path': safe_url(live.get('image_path') or '')}
    precondition.update(retry_precondition(previous))

    def remove():
        value = checked_id(client.update_person(performer, image=''), performer)
        if not owns(value, token) or not missing_image(value, client, performer):
            raise PendingRecognition('Portrait removal acknowledgement was not confirmed.')
        confirmed = checked_id(client.performer(performer), performer)
        if not owns(confirmed, token) or not missing_image(confirmed, client, performer):
            raise PendingRecognition('Live portrait removal/ownership was not confirmed.')
        return value

    perform(database, client, report, scope=scope, scene='', person=person, kind='image', target=performer,
            payload=intent, precondition=precondition, callback=remove)
    with database.connection:
        database.connection.execute("UPDATE portraits SET status='removed' WHERE id=?", (intent['portrait'],))
    counts = {'portraits_removed': 1}
    if 'retry_of' in precondition:
        counts['portraits_retried'] = 1
    return counts
