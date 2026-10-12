"""Wider portraits from verified VTT thumbnails; reuse recognition vectors."""
import hashlib
from itertools import groupby
import json
import uuid

from .clustering import _groups, vector
from .config import plain
from .database import encoded
from .media import safe_url
from .models import checked_detection
from .portraits import hints, identity_reason
from .quality import portrait_bytes, portrait_score, rejection


def choose_vtt(database, client, config, scope, target, source_factory):
    db, policy = database.connection, config.data['portraits']
    person = db.execute('SELECT * FROM people WHERE id=? AND scope=?', (target['person'], scope)).fetchone()
    groups = {group.identifier: group for group in _groups(db, scope, person['namespace'], config.data['clustering'])}
    selected = hints(database, target, config)
    shortlist = [{'scene': scene, 'observation': row['id']} for scene, row in selected]
    policy_sha = config.identity('portraits', 'recognition', 'clustering')
    winner, winning_key = None, None
    # Keep only one bounded sprite source open; several candidates in the same
    # scene share its verified sheets, without reloading a recognition backend.
    for scene, observations in groupby(sorted(selected, key=lambda item: (item[0], item[1]['cue'])),
                                      key=lambda item: item[0]):
        source = None
        source_failed = False
        try:
            for _, observation in observations:
                identifier = str(uuid.uuid5(uuid.NAMESPACE_URL,
                    encoded([scope, target['person'], observation['id'], 'vtt', policy_sha])))
                existing = db.execute('SELECT * FROM portraits WHERE id=?', (identifier,)).fetchone()
                if existing and existing['image'] is not None:
                    metadata = json.loads(existing['metadata'])
                    if not metadata.get('reason'):
                        key = (-metadata['score'], scene, observation['seconds'], identifier)
                        if winning_key is None or key < winning_key:
                            winner, winning_key = existing, key
                    continue
                quality = json.loads(observation['quality'])
                metadata = {'version': 1, 'source_type': 'vtt', 'selection_policy': 'vtt-observation-identity-first-v1',
                            'policy_sha': policy_sha, 'policy': plain(policy), 'shortlist': shortlist,
                            'observation': observation['id'], 'scan': observation['scan'], 'scene': scene,
                            'person': target['person'], 'performer': target['performer'], 'namespace': person['namespace'],
                            'source': observation['source'], 'rect': json.loads(observation['rect']),
                            'cue': observation['cue'], 'cue_seconds': observation['seconds'],
                            'geometry': json.loads(observation['geometry']), 'quality': quality}
                reason, image_bytes = rejection(quality, policy, portrait=True), None
                if not reason:
                    row = db.execute('''SELECT o.embedding,o.dimensions,s.input_sha FROM observations o
                      JOIN members m ON m.observation=o.id JOIN scans s ON s.id=o.scan
                      WHERE o.id=? AND o.scan=? AND m.person=? AND s.scene=? AND s.scope=?
                      AND s.namespace=? AND s.status='complete' AND o.rejected IS NULL''',
                      (observation['id'], observation['scan'], target['person'], scene, scope,
                       person['namespace'])).fetchone()
                    try:
                        if row is None or row['embedding'] is None:
                            raise ValueError('VTT recognition evidence is unavailable.')
                        reason, similarity, runner = identity_reason(vector(row), groups, target['person'], config)
                        metadata.update(similarity=similarity, runner_up=runner, input_sha=row['input_sha'])
                        if not reason:
                            if source_failed:
                                raise ValueError('VTT source could not be verified.')
                            if source is None:
                                try:
                                    source = source_factory(client, client.scene(scene), config).__enter__()
                                except Exception:
                                    source_failed = True
                                    raise
                            if source.fingerprint != row['input_sha']:
                                raise ValueError('VTT source changed after recognition.')
                            cue = source.cues[observation['cue']]
                            if (cue.start != observation['seconds'] or safe_url(cue.image) != observation['source']
                                    or list(cue.rect) != metadata['rect']):
                                raise ValueError('VTT cue does not match its recognition observation.')
                            with source.thumbnail(observation['cue']) as image:
                                face = checked_detection(metadata['geometry'])
                                image_bytes, bounds = portrait_bytes(image, face, policy,
                                    limit=config.data['storage']['max_crop_bytes'])
                                metadata.update(crop=bounds, pixels=[bounds[2] - bounds[0], bounds[3] - bounds[1]],
                                                score=portrait_score(quality, policy))
                    except Exception:
                        reason = 'vtt_source_unavailable_or_changed'
                if reason:
                    metadata['reason'] = reason
                sha = hashlib.sha256(image_bytes).hexdigest() if image_bytes else None
                with db:
                    db.execute('INSERT OR REPLACE INTO portraits VALUES (?,?,?,?,?,?,?,?)',
                        (identifier, scope, target['person'], scene, encoded(metadata), image_bytes, sha,
                         'candidate' if image_bytes else 'skipped'))
                if image_bytes:
                    key = (-metadata['score'], scene, observation['seconds'], identifier)
                    if winning_key is None or key < winning_key:
                        winner, winning_key = db.execute('SELECT * FROM portraits WHERE id=?', (identifier,)).fetchone(), key
        finally:
            if source is not None:
                source.close()
    if winner:
        with db:
            db.execute("UPDATE portraits SET status='selected' WHERE id=?", (winner['id'],))
    return winner
