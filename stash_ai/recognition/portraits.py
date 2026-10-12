"""Identity-first ranking of actual video frames, with SQLite image provenance."""
import base64
import hashlib
import json
import uuid
from urllib.parse import parse_qs, urlsplit

from stash_ai.runtime.cleanup import closing
from stash_ai.stash_api.transport import absolute_stash_url, origin
from .clustering import _groups
from .config import digest, plain
from .database import encoded
from .media import bounded_media, safe_url
from .models import normalized_embedding
from .publication import operation_id, perform, PendingRecognition
from .quality import assess, crop_bytes, portrait_score, rejection
from .stash import checked_id, owns


def missing_image(value, client, performer):
    path = value.get('image_path')
    if path is None or path == '':
        return True
    if not isinstance(path, str):
        raise ValueError('Stash image_path must be a string or null.')
    url = absolute_stash_url(path, client.url)
    parts = urlsplit(url)
    expected = urlsplit(client.url).path.rstrip('/') + f'/performer/{performer}/image'
    # Stash v0.31.1 URL builder appends default=true exactly when HasImage=false.
    return (origin(url) == origin(client.url) and parts.path == expected and
            parse_qs(parts.query).get('default') == ['true'])


def hints(database, target, config):
    """Bounded quality shortlist with scene and temporal diversity."""
    policy = config.data['portraits']
    queues = []
    for item in sorted(target['scans'], key=lambda x: x['scene']):
        rows = database.connection.execute('''SELECT o.* FROM observations o JOIN members m ON m.observation=o.id
          WHERE m.person=? AND o.scan=? AND o.rejected IS NULL
          ORDER BY json_extract(o.quality,'$.sharpness') DESC,o.cue,o.id LIMIT ?''',
          (target['person'], item['scan'], policy['max_candidates'] * 8)).fetchall()
        scored = [(portrait_score(json.loads(row['quality']), policy), row) for row in rows]
        scored.sort(key=lambda pair: (-pair[0], pair[1]['seconds'], pair[1]['id']))
        selected = []
        while scored and len(selected) < policy['max_candidates']:
            if selected:
                scored.sort(key=lambda pair: (-min(abs(pair[1]['seconds'] - r['seconds'])
                                                 for r in selected), -pair[0], pair[1]['id']))
            selected.append(scored.pop(0)[1])
        queues.append((item['scene'], selected))
    result = []
    while any(rows for _, rows in queues) and len(result) < policy['max_candidates']:
        for scene, rows in queues:
            if rows and len(result) < policy['max_candidates']:
                result.append((scene, rows.pop(0)))
    return result


def identity_reason(vector, groups, person, config):
    import numpy as np
    policy = config.data['recognition']
    scores = {}
    for identifier, group in groups.items():
        if not group.samples or len(group.samples[0]) != len(vector):
            raise ValueError('Portrait embedding namespace/dimensions are incompatible.')
        values = [float(np.dot(vector, sample)) for sample in group.samples]
        if min(values) >= config.data['clustering']['cohesion_threshold']:
            scores[identifier] = sum(values) / len(values)
    score = scores.get(person, -1.)
    runner = max((v for k, v in scores.items() if k != person), default=-1.)
    if score < policy['match_threshold']:
        return 'portrait_identity_mismatch', score, runner
    if score - runner < policy['ambiguity_margin']:
        return 'portrait_identity_ambiguous', score, runner
    return None, score, runner


def choose(database, client, config, scope, target, paths, backend_factory, decoder_factory):
    policy = config.data['portraits']
    person = database.connection.execute('SELECT * FROM people WHERE id=? AND scope=?',
                                         (target['person'], scope)).fetchone()
    groups = {g.identifier: g for g in _groups(database.connection, scope, person['namespace'], config.data['clustering'])}
    selected = hints(database, target, config)
    selection = [{'scene': scene, 'observation': row['id']} for scene, row in selected]
    policy_sha = config.identity('portraits', 'recognition', 'clustering')
    winner, winning_key = None, None
    decoders = {}
    with closing(backend_factory(config, paths), name='close portrait face backend') as backend:
        try:
            for scene_id, row in selected:
                for offset in policy['time_offsets']:
                    seconds = max(0., row['seconds'] + offset)
                    identifier = str(uuid.uuid5(uuid.NAMESPACE_URL,
                        encoded([scope, target['person'], row['id'], seconds, policy_sha])))
                    existing = database.connection.execute('SELECT * FROM portraits WHERE id=?', (identifier,)).fetchone()
                    if existing and existing['image'] is not None:
                        metadata = json.loads(existing['metadata'])
                        if existing['image'] is not None and not metadata.get('reason'):
                            key = (-metadata['score'], scene_id, seconds, identifier)
                            if winning_key is None or key < winning_key:
                                winner, winning_key = existing, key
                        continue
                    metadata = {'version': 1, 'selection_policy': 'identity-first-full-face-v1',
                                'policy_sha': policy_sha, 'policy': plain(policy), 'shortlist': selection,
                                'observation': row['id'], 'scan': row['scan'], 'scene': scene_id,
                                'person': target['person'], 'performer': target['performer'],
                                'namespace': person['namespace'], 'requested_seconds': seconds}
                    image_bytes = None
                    try:
                        if scene_id not in decoders:
                            scene = client.scene(scene_id)
                            metadata['source'] = safe_url(absolute_stash_url(scene['paths'].get('stream') or '', client.url))
                            decoders[scene_id] = decoder_factory(client, scene, config).__enter__()
                        metadata['source'] = safe_url(decoders[scene_id].url) if hasattr(decoders[scene_id], 'url') else scene_id
                        image, timing = decoders[scene_id].frame(seconds)
                        with image:
                            metadata.update(timing)
                            candidates = []
                            exclusions = []
                            for face in backend.detect(image):
                                quality = assess(image, face)
                                reason = rejection(quality, policy, portrait=True)
                                x, y, width, height = face.box
                                if not reason and (x - width * policy['margin'] < 0 or y - height * policy['margin'] < 0 or
                                        x + width * (1 + policy['margin']) > image.width or
                                        y + height * (1 + policy['margin']) > image.height):
                                    reason = 'portrait_margin_clipped'
                                similarity, runner = None, None
                                if not reason:
                                    vector = normalized_embedding(backend.embed(image, face), backend.dimensions)
                                    reason, similarity, runner = identity_reason(vector, groups, target['person'], config)
                                record = {'geometry': list(face.row), 'quality': quality, 'reason': reason,
                                          'similarity': similarity, 'runner_up': runner}
                                if reason:
                                    exclusions.append(record)
                                else:
                                    record['score'] = portrait_score(quality, policy)
                                    candidates.append((record, face))
                            metadata['exclusions'] = exclusions
                            if candidates:
                                record, face = min(candidates, key=lambda pair: (-pair[0]['score'], pair[0]['geometry']))
                                image_bytes, bounds = crop_bytes(image, face, margin=policy['margin'],
                                                                limit=config.data['storage']['max_crop_bytes'])
                                metadata.update(record)
                                metadata['crop'] = bounds
                            else:
                                metadata['reason'] = 'no_identity_verified_full_face'
                    except KeyboardInterrupt:
                        raise
                    except Exception:
                        # Decoder/source limitations are honest image skips, not
                        # failed profile/link publication or relabeled VTT crops.
                        metadata['reason'] = 'video_unavailable_or_budget_exceeded'
                    sha = hashlib.sha256(image_bytes).hexdigest() if image_bytes else None
                    with database.connection:
                        database.connection.execute('INSERT OR REPLACE INTO portraits VALUES (?,?,?,?,?,?,?,?)',
                            (identifier, scope, target['person'], scene_id, encoded(metadata), image_bytes, sha,
                             'candidate' if image_bytes else 'skipped'))
                    if image_bytes:
                        key = (-metadata['score'], scene_id, seconds, identifier)
                        if winning_key is None or key < winning_key:
                            winner = database.connection.execute('SELECT * FROM portraits WHERE id=?', (identifier,)).fetchone()
                            winning_key = key
        finally:
            from stash_ai.runtime.cleanup import cleanup
            cleanup([('close portrait video decoder', decoder.close) for decoder in decoders.values()])
    if winner:
        with database.connection:
            database.connection.execute("UPDATE portraits SET status='selected' WHERE id=?", (winner['id'],))
    return winner


def fill_portrait(database, client, report, config, scope, target, paths, backend_factory, decoder_factory):
    performer, person = target['performer'], target['person']
    if not config.data['portraits']['enabled']:
        return {'portraits_disabled': 1}
    token = database.connection.execute('SELECT token FROM people WHERE id=? AND scope=?', (person, scope)).fetchone()[0]
    live = checked_id(client.performer(performer), performer)
    if not owns(live, token):
        raise PendingRecognition('Portrait target ownership changed.')
    identifier = operation_id(scope, '', person, 'image')
    previous = database.connection.execute('SELECT * FROM publications WHERE id=?', (identifier,)).fetchone()
    if not missing_image(live, client, performer):
        if previous and previous['status'] in ('pending', 'unknown'):
            intent = json.loads(previous['payload'])
            data = bounded_media(client, absolute_stash_url(live['image_path'], client.url),
                                 config.data['storage']['max_crop_bytes'], config.data['execution']['timeout_seconds'])
            same = hashlib.sha256(data).hexdigest() == intent['sha256']
            with database.connection:
                database.connection.execute('UPDATE publications SET status=?,error=NULL WHERE id=?',
                                            ('observed' if same else 'conflict', identifier))
                if same:
                    database.connection.execute("UPDATE portraits SET status='published' WHERE id=?", (intent['portrait'],))
        return {'portraits_preserved': 1}
    if previous and previous['status'] in ('pending', 'unknown'):
        raise PendingRecognition('Earlier image write is unresolved; no blind retry.')
    winner = choose(database, client, config, scope, target, paths, backend_factory, decoder_factory)
    if winner is None:
        return {'portraits_skipped': 1}
    # Protect an image supplied while extraction was in progress.
    live = checked_id(client.performer(performer), performer)
    if not owns(live, token):
        raise PendingRecognition('Portrait target ownership changed during extraction.')
    if not missing_image(live, client, performer):
        return {'portraits_preserved': 1}
    payload = 'data:image/jpeg;base64,' + base64.b64encode(winner['image']).decode('ascii')
    def update():
        value = checked_id(client.update_person(performer, image=payload), performer)
        if not owns(value, token) or missing_image(value, client, performer):
            raise PendingRecognition('Portrait acknowledgement was not confirmed.')
        confirmed = checked_id(client.performer(performer), performer)
        if not owns(confirmed, token) or missing_image(confirmed, client, performer):
            raise PendingRecognition('Live portrait ownership/image changed after acknowledgement.')
        data = bounded_media(client, absolute_stash_url(confirmed['image_path'], client.url),
                             config.data['storage']['max_crop_bytes'], config.data['execution']['timeout_seconds'])
        if hashlib.sha256(data).hexdigest() != winner['digest']:
            raise PendingRecognition('Live portrait differs from the submitted image; preserved for reconciliation.')
        return value
    perform(database, client, report, scope=scope, scene='', person=person, kind='image', target=performer,
            payload={'portrait': winner['id'], 'sha256': winner['digest']},
            precondition={'missing': True, 'image_path': safe_url(live.get('image_path') or ''), 'token': token},
            callback=update)
    with database.connection:
        database.connection.execute("UPDATE portraits SET status='published' WHERE id=?", (winner['id'],))
    return {'portraits_filled': 1}
