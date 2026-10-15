"""Identity-first ranking of video frames or the original VTT observations."""
import base64
import hashlib
import json
import uuid
from urllib.parse import parse_qs, urlsplit

from stash_ai.stash_api.transport import absolute_stash_url, origin
from .clustering import _groups, reference_context, vector as retained_vector
from .config import digest, plain
from .database import encoded
from .media import bounded_media, safe_url, SpriteInput
from .models import normalized_embedding
from .publication import operation_id, perform, PendingRecognition, save_operation
from .quality import PortraitFramingError, assess, portrait_bytes, portrait_score, portrait_tiebreak, portrait_shortfalls
from .portrait_availability import measure_available, score_metadata, refresh_cached_quality
from .stash import checked_id, owns
from .visibility import LazyVisibility, CONTRACT_ID
from .expression import LazyExpression, CONTRACT_ID as EXPRESSION_CONTRACT_ID
from .diagnostics import metadata as safe_metadata
from .progress import tracking
from .backend import prepared_frame


def policy_identity(config):
    from .context import policy_digest
    return policy_digest(config, ('portraits-v4', CONTRACT_ID, EXPRESSION_CONTRACT_ID),
                         lambda: _policy_value(config))


def _policy_value(config):
    policy = plain(config.data['portraits'])
    if policy['source'] == 'vtt':
        for name in ('max_candidates', 'time_offsets', 'timeout_seconds', 'max_video_bytes', 'ffmpeg'):
            policy.pop(name)
    from .visibility import required as visibility_required
    from .expression import required as expression_required
    return {'version': 4, 'portraits': policy,
                   'max_crop_bytes': config.data['storage']['max_crop_bytes'],
                   'recognition': plain(config.data['recognition']), 'clustering': plain(config.data['clustering']),
                   'visibility_contract': CONTRACT_ID if visibility_required(policy) else None,
                   'expression_contract': EXPRESSION_CONTRACT_ID if expression_required(policy) else None}


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
        rows = database.connection.execute('''SELECT o.id,o.scan,o.cue,o.seconds,o.source,o.rect,o.geometry,o.quality
          FROM observations o JOIN members m ON m.observation=o.id
          WHERE m.person=? AND o.scan=? AND o.rejected IS NULL
          ORDER BY json_extract(o.quality,'$.sharpness') DESC,o.cue,o.id LIMIT ?''',
          (target['person'], item['scan'], policy['max_candidates'] * 8)).fetchall()
        # These are VTT hints, not the expression of the eventual video frame.
        scored = [(portrait_score(json.loads(row['quality']), policy, expression=False), row) for row in rows]
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
    score, runner = -1., None
    for identifier, group in groups.items():
        if not group.samples or len(group.samples[0]) != len(vector):
            raise ValueError('Portrait embedding namespace/dimensions are incompatible.')
        values = [float(np.dot(vector, sample)) for sample in group.samples]
        if min(values) >= config.data['clustering']['cohesion_threshold']:
            mean = sum(values) / len(values)
            if identifier == person:
                score = mean
            else:
                runner = mean if runner is None else max(runner, mean)
    runner = -1. if runner is None else runner
    if score < policy['match_threshold']:
        return 'portrait_identity_mismatch', score, runner
    if score - runner < policy['ambiguity_margin']:
        return 'portrait_identity_ambiguous', score, runner
    return None, score, runner


def recheck_cached(database, existing, groups, config, target, namespace, context, *, evidence=None,
                   client=None, visibility=None, expression=None):
    """Keep the JPEG but refresh its eligibility from retained identity evidence."""
    import numpy as np
    metadata = json.loads(existing['metadata'])
    updated = safe_metadata(metadata, getattr(client, 'api_key', '') if client is not None else '')
    similarity = runner = None
    data, sha = existing['image'], existing['digest']
    try:
        from .portrait_availability import fit_cached_bytes
        data, sha = fit_cached_bytes(data, sha, config, updated)
        if metadata.get('namespace') != namespace or metadata.get('person') != target['person']:
            raise ValueError('Cached portrait identity namespace changed.')
        if evidence is None:
            stored = metadata['identity_embedding']
            dimensions, values = stored['dimensions'], stored['values']
            if (type(dimensions) is not int or not 1 <= dimensions <= 1024 or not isinstance(values, list)
                    or len(values) != dimensions or any(type(item) not in (int, float) for item in values)):
                raise ValueError('Cached portrait has no valid identity embedding.')
            evidence = {'dimensions': dimensions, 'embedding': np.asarray(values, dtype='<f4').tobytes()}
        query = retained_vector(evidence)
        reason, similarity, runner = identity_reason(query, groups, target['person'], config)
        if not reason:
            if client is not None:
                refresh_cached_quality(client, existing, updated, config.data['portraits'], visibility, expression)
            quality = metadata.get('quality', {})
            score_metadata(updated, updated.get('quality', quality), config.data['portraits'])
    except (KeyError, TypeError, ValueError):
        # Older video portraits lack an embedding. Preserve their bytes, but do
        # not infer from a new source or publish unverified old eligibility.
        reason = 'portrait_identity_unverified'
    updated.update(reference_context=context, similarity=similarity, runner_up=runner)
    if reason:
        updated['reason'] = reason
    else:
        updated.pop('reason', None)
    if updated != metadata or data != existing['image']:
        with database.connection:
            database.connection.execute('UPDATE portraits SET metadata=?,image=?,digest=? WHERE id=?', (encoded(updated), data, sha, existing['id']))
        existing = database.connection.execute('SELECT * FROM portraits WHERE id=?', (existing['id'],)).fetchone()
    return None if reason else existing


def choose(database, client, config, scope, target, paths, backend_factory, decoder_factory,
           source_factory=SpriteInput, *, visibility=None, expression=None):
    if visibility is None:
        from stash_ai.runtime.cleanup import closing
        with closing(LazyVisibility(config, paths), name='close portrait visibility model') as owned:
            return choose(database, client, config, scope, target, paths, backend_factory, decoder_factory,
                          source_factory, visibility=owned, expression=expression)
    if expression is None:
        from stash_ai.runtime.cleanup import closing
        with closing(LazyExpression(config, paths), name='close portrait expression model') as owned:
            return choose(database, client, config, scope, target, paths, backend_factory, decoder_factory,
                          source_factory, visibility=visibility, expression=owned)
    if config.data['portraits']['source'] == 'vtt':
        from .vtt_portraits import choose_vtt
        return choose_vtt(database, client, config, scope, target, source_factory, visibility, expression)
    policy = config.data['portraits']
    person = database.connection.execute('SELECT * FROM people WHERE id=? AND scope=?',
                                         (target['person'], scope)).fetchone()
    from .context import portrait_references
    groups, context = portrait_references(database, scope, person['namespace'], config,
        lambda: _groups(database.connection, scope, person['namespace'], config.data['clustering']))
    selected = hints(database, target, config)
    selection = [{'scene': scene, 'observation': row['id']} for scene, row in selected]
    policy_sha = policy_identity(config)
    winner, winning_key = None, None
    decoders = {}
    backend = None
    def identity_backend():
        nonlocal backend
        if backend is None or not getattr(backend, 'active', True):
            if backend is not None:
                backend.close()
            backend = backend_factory(config, paths)
        return backend

    attempts = ((scene_id, row, offset) for scene_id, row in selected for offset in policy['time_offsets'])
    try:
        with tracking(
            attempts,
            'Check video portrait frames',
            total=len(selected) * len(policy['time_offsets']),
            unit='frame',
        ) as frames:
            for scene_id, row, offset in frames:
                seconds = max(0., row['seconds'] + offset)
                identifier = str(uuid.uuid5(uuid.NAMESPACE_URL,
                    encoded([scope, target['person'], row['id'], seconds, policy_sha])))
                existing = database.connection.execute('SELECT * FROM portraits WHERE id=?', (identifier,)).fetchone()
                if existing is None:
                    from .video_portrait_cache import reuse
                    existing = reuse(database, config, scope, target, row, scene_id, seconds, person['namespace'],
                                     identifier, policy_sha, selection)
                if existing and existing['image'] is not None:
                    existing = recheck_cached(database, existing, groups, config, target, person['namespace'], context,
                                              client=client, visibility=visibility, expression=expression)
                    if existing is not None:
                        metadata = json.loads(existing['metadata'])
                        key = (-metadata['score'], -metadata['quality_tiebreak'], scene_id, seconds, identifier)
                        if winning_key is None or key < winning_key:
                            winner, winning_key = existing, key
                    continue
                metadata = {'version': 3, 'source_type': 'video', 'requested_source': 'video',
                            'selection_policy': 'best-available-video-identity-first-v3',
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
                    with image, prepared_frame(identity_backend(), image):
                        metadata.update(timing)
                        candidates = []
                        exclusions = []
                        for face in identity_backend().detect(image):
                            quality = assess(image, face)
                            similarity, runner = None, None
                            try:
                                model = identity_backend()
                                vector = normalized_embedding(model.embed(image, face), model.dimensions)
                                reason, similarity, runner = identity_reason(vector, groups, target['person'], config)
                            except Exception:
                                reason = 'portrait_identity_unverified'
                            record = {'geometry': list(face.row), 'quality': quality, 'reason': reason,
                                      'similarity': similarity, 'runner_up': runner}
                            if reason:
                                exclusions.append(record)
                            else:
                                record['quality_errors'] = measure_available(client, image, face, quality, policy, visibility, expression)
                                record['identity_embedding'] = {'dimensions': backend.dimensions, 'values': vector.tolist()}
                                record['reference_context'] = context
                                score_metadata(record, quality, policy)
                                candidates.append((record, face))
                        metadata['exclusions'] = exclusions
                        for record, face in sorted(candidates, key=lambda pair: (-pair[0]['score'], -pair[0]['quality_tiebreak'], pair[0]['geometry'])):
                            encoding = {}
                            try:
                                image_bytes, bounds = portrait_bytes(image, face, policy,
                                                                    limit=config.data['storage']['max_crop_bytes'], details=encoding)
                            except PortraitFramingError:
                                exclusions.append({**record, 'reason': 'portrait_encoding_unavailable'})
                                continue
                            metadata.update(record)
                            metadata.update(crop=bounds, encoding=encoding, pixels=encoding['pixels'], crop_coordinate_system='video_frame')
                            break
                        else:
                            metadata['reason'] = 'portrait_encoding_unavailable' if candidates else 'no_identity_verified_face'
                except KeyboardInterrupt:
                    raise
                except PortraitFramingError:
                    metadata['reason'] = 'portrait_encoding_unavailable'
                except Exception:
                    # Decoder/source limitations are honest image skips, not
                    # failed profile/link publication. A fallback retains its
                    # actual VTT/recognition source in separate metadata.
                    metadata['reason'] = 'video_unavailable_or_budget_exceeded'
                sha = hashlib.sha256(image_bytes).hexdigest() if image_bytes else None
                with database.connection:
                    database.connection.execute('INSERT OR REPLACE INTO portraits VALUES (?,?,?,?,?,?,?,?)',
                        (identifier, scope, target['person'], scene_id, encoded(metadata), image_bytes, sha,
                         'candidate' if image_bytes else 'skipped'))
                if image_bytes:
                    key = (-metadata['score'], -metadata['quality_tiebreak'], scene_id, seconds, identifier)
                    if winning_key is None or key < winning_key:
                        winner = database.connection.execute('SELECT * FROM portraits WHERE id=?', (identifier,)).fetchone()
                        winning_key = key
    except Exception as error:
        report = getattr(client, 'report', None)
        if report is not None:
            from .diagnostics import message as safe_error
            report.optional('recognition_portrait_video', report.write,
                            'Video portrait unavailable; using recognition evidence: ' + safe_error(error, getattr(client, 'api_key', '')))
    finally:
        from stash_ai.runtime.cleanup import cleanup
        actions = [('close portrait video decoder', decoder.close) for decoder in decoders.values()]
        if backend is not None:
            actions.append(('close portrait face backend', backend.close))
        cleanup(actions)
    if winner:
        with database.connection:
            database.connection.execute("UPDATE portraits SET status='selected' WHERE id=?", (winner['id'],))
    else:
        from .vtt_portraits import choose_vtt
        winner = choose_vtt(database, client, config, scope, target, source_factory, visibility, expression,
                            fallback_reason='no_usable_identity_verified_video_portrait')
    return winner


def image_digest(client, live, performer, config):
    if missing_image(live, client, performer):
        return None
    data = bounded_media(client, absolute_stash_url(live['image_path'], client.url),
                         8388608, config.data['execution']['timeout_seconds'])
    # Confirmation covers any previously supported owned JPEG, independently
    # of today's encoding budget. Manual/unknown bytes still need an exact hash.
    return hashlib.sha256(data).hexdigest()


def image_baseline(database, client, config, live, performer, previous, *, retry_unresolved_images=False):
    """Recognize only our acknowledged bytes; reconcile uncertain writes first."""
    if previous is None:
        return missing_image(live, client, performer), None, None
    current = image_digest(client, live, performer, config)
    intent = json.loads(previous['payload'])
    if previous['status'] in ('pending', 'unknown'):
        if current == intent['sha256']:
            with database.connection:
                database.connection.execute("UPDATE publications SET status='observed',error=NULL WHERE id=?",
                                            (previous['id'],))
                database.connection.execute('UPDATE portraits SET status=? WHERE id=?',
                    ('removed' if intent.get('action') == 'remove' else 'published', intent['portrait']))
            previous = database.connection.execute('SELECT * FROM publications WHERE id=?', (previous['id'],)).fetchone()
        elif current is None or current == json.loads(previous['precondition']).get('image_sha256'):
            if retry_unresolved_images:
                # Keep the uncertain intent until a freshly checked replacement
                # is ready. Opt-in authorizes an attempt, not a successful outcome.
                return True, current, previous
            raise PendingRecognition(f'Earlier image write is unresolved for performer {performer}; '
                                     'rerun the selected scenes with --retry-unresolved-images to retry.')
        else:
            with database.connection:
                database.connection.execute("UPDATE publications SET status='conflict',error=NULL WHERE id=?",
                                            (previous['id'],))
            return False, current, previous
    if current is None:
        return True, None, previous
    expected = (intent['sha256'] if previous['status'] in ('confirmed', 'observed')
                else json.loads(previous['precondition']).get('image_sha256')
                if previous['status'] == 'failed' else None)
    return current == expected, current, previous


def fill_portrait(database, client, report, config, scope, target, paths, backend_factory, decoder_factory,
                 source_factory=SpriteInput, *, retry_unresolved_images=False, visibility=None, expression=None):
    from .portrait_removal import remove_owned, retry_precondition
    performer, person = target['performer'], target['person']
    enabled = config.data['portraits']['enabled']
    token = database.connection.execute('SELECT token FROM people WHERE id=? AND scope=?', (person, scope)).fetchone()[0]
    live = checked_id(client.performer(performer), performer)
    if not owns(live, token):
        raise PendingRecognition('Portrait target ownership changed.')
    identifier = operation_id(scope, '', person, 'image')
    previous = database.connection.execute('SELECT * FROM publications WHERE id=?', (identifier,)).fetchone()
    managed, baseline, previous = image_baseline(database, client, config, live, performer, previous,
                                                retry_unresolved_images=retry_unresolved_images)
    if not managed:
        return {'portraits_preserved': 1}
    retrying = previous is not None and previous['status'] in ('pending', 'unknown')
    if not enabled:
        if retrying and baseline is None:
            raise PendingRecognition('Earlier image write remains unresolved; no acknowledged portrait for removal.')
        counts = (remove_owned(database, client, report, config, scope, target, token, baseline, previous,
                               reason='portraits_disabled') if baseline is not None else {})
        return {'portraits_disabled': 1, **counts}
    policy_sha = policy_identity(config)
    prior = None
    if baseline is not None and previous['status'] in ('confirmed', 'observed'):
        portrait = database.connection.execute('SELECT * FROM portraits WHERE id=?',
                                               (json.loads(previous['payload'])['portrait'],)).fetchone()
        if portrait:
            metadata = json.loads(portrait['metadata'])
            observation = metadata.get('observation')
            still_member = database.connection.execute('SELECT 1 FROM members m JOIN observations o '
                'ON o.id=m.observation WHERE m.person=? AND m.observation=? AND o.rejected IS NULL',
                (target['person'], observation)).fetchone()
            if metadata['policy_sha'] == policy_sha and still_member:
                prior = portrait
    winner = choose(database, client, config, scope, target, paths, backend_factory, decoder_factory, source_factory,
                    visibility=visibility, expression=expression)
    if winner is None:
        if baseline is not None:
            return {'portraits_preserved': 1, 'portraits_no_candidate': 1}
        if retrying:
            raise PendingRecognition('Earlier image write remains unresolved; no currently eligible portrait for retry.')
        return {'portraits_skipped': 1}
    if prior is not None:
        old, new = json.loads(prior['metadata']), json.loads(winner['metadata'])
        ranks = {'video': 0, 'vtt': 1, 'recognition_crop': 2}
        def key(metadata):
            quality = metadata.get('quality', {})
            return (-portrait_score(quality, config.data['portraits']),
                    -portrait_tiebreak(quality, config.data['portraits']), ranks.get(metadata.get('source_type'), 3))
        if key(old) < key(new):
            return {'portraits_preserved': 1, **quality_counts(prior, config)}
    # Protect manual replacements/removals while extraction was in progress.
    live = checked_id(client.performer(performer), performer)
    if not owns(live, token):
        raise PendingRecognition('Portrait target ownership changed during extraction.')
    if missing_image(live, client, performer) != (baseline is None):
        return {'portraits_preserved': 1}
    if image_digest(client, live, performer, config) != baseline:
        return {'portraits_preserved': 1}
    intent = {'portrait': winner['id'], 'sha256': winner['digest']}
    precondition = {'missing': baseline is None, 'image_sha256': baseline,
                    'image_path': safe_url(live.get('image_path') or ''), 'token': token}
    precondition.update(retry_precondition(previous, api_key=getattr(client, 'api_key', '')))
    if winner['digest'] == baseline:
        if previous is None or previous['status'] not in ('confirmed', 'observed') or json.loads(previous['payload']) != intent:
            save_operation(database, scope=scope, scene='', person=person, kind='image', target=performer,
                           payload=intent, precondition=precondition, status='observed')
        with database.connection:
            database.connection.execute("UPDATE portraits SET status='published' WHERE id=?", (winner['id'],))
        return {'portraits_preserved': 1, **quality_counts(winner, config)}
    if (len(winner['image']) > config.data['storage']['max_crop_bytes']
            or hashlib.sha256(winner['image']).hexdigest() != winner['digest']):
        raise PendingRecognition('Selected portrait bytes do not satisfy the verified encoding budget.')
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
            payload=intent, precondition=precondition, callback=update)
    with database.connection:
        database.connection.execute("UPDATE portraits SET status='published' WHERE id=?", (winner['id'],))
    counts = {'portraits_filled' if baseline is None else 'portraits_updated': 1}
    counts.update(quality_counts(winner, config))
    if retrying:
        counts['portraits_retried'] = 1
    return counts


def quality_counts(winner, config):
    metadata = json.loads(winner['metadata'])
    encoding = metadata.get('encoding', {})
    limited = (portrait_shortfalls(metadata.get('quality', {}), config.data['portraits'])
               or any(encoding.get(name) for name in ('aspect_ratio_adjusted', 'padding_adjusted', 'downscaled'))
               or encoding.get('jpeg_quality', config.data['portraits']['jpeg_quality']) < config.data['portraits']['jpeg_quality'])
    return {**({'portraits_quality_limited': 1} if limited else {}),
            **({'portraits_fallback': 1} if metadata.get('fallback') else {})}
