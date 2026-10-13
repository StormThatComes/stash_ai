"""Optional quality measurements and truthful fallback to retained recognition pixels."""
from io import BytesIO
import hashlib
import json
import uuid

from stash_ai.inference.image_io import open_image
from stash_ai.stash_api.transport import safe_error
from .models import checked_detection
from .quality import portrait_bytes, portrait_score, portrait_shortfalls, portrait_tiebreak
from .visibility import required as visibility_required, cached as visibility_cached
from .expression import required as expression_required, cached as expression_cached


def measure_available(client, image, face, quality, policy, visibility, expression):
    errors = {}
    for name, required, cached, model in (
        ('visibility', visibility_required, visibility_cached, visibility),
        ('expression', expression_required, expression_cached, expression),
    ):
        if not required(policy) or cached(quality):
            continue
        try:
            quality.update(model.measure(image, face))
        except Exception as error:
            errors[name] = safe_error(error, getattr(client, 'api_key', ''))[:500]
            report = getattr(client, 'report', None)
            if report is not None:
                seen = getattr(report, '_portrait_quality_warnings', set())
                if name not in seen:
                    seen.add(name)
                    report._portrait_quality_warnings = seen
                    report.optional('recognition_portrait_quality', report.write,
                                    f'Portrait {name} unavailable; ranking available face quality: ' + errors[name])
    return errors


def refresh_cached_quality(client, existing, metadata, policy, visibility, expression):
    """Retry optional cues from verified cached pixels, without another video seek."""
    quality = dict(metadata.get('quality', {}))
    if not (visibility_required(policy) and not visibility_cached(quality)
            or expression_required(policy) and not expression_cached(quality)):
        return
    if hashlib.sha256(existing['image']).hexdigest() != existing['digest']:
        raise ValueError('Cached portrait bytes do not match their digest.')
    bounds = metadata['crop']
    if (not isinstance(bounds, list) or len(bounds) != 4 or any(type(v) is not int for v in bounds)
            or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]):
        raise ValueError('Cached portrait has invalid crop bounds.')
    with open_image(BytesIO(existing['image'])) as raw:
        with raw.convert('RGB') as image:
            if image.width > bounds[2] - bounds[0] or image.height > bounds[3] - bounds[1]:
                raise ValueError('Cached portrait dimensions exceed its original crop.')
            scale_x, scale_y = image.width / (bounds[2] - bounds[0]), image.height / (bounds[3] - bounds[1])
            geometry = list(metadata['geometry'])
            geometry[0], geometry[1] = (geometry[0] - bounds[0]) * scale_x, (geometry[1] - bounds[1]) * scale_y
            geometry[2], geometry[3] = geometry[2] * scale_x, geometry[3] * scale_y
            for index in range(4, 14, 2):
                geometry[index] = (geometry[index] - bounds[0]) * scale_x
                geometry[index + 1] = (geometry[index + 1] - bounds[1]) * scale_y
            metadata['quality_errors'] = measure_available(client, image, checked_detection(geometry), quality,
                                                           policy, visibility, expression)
    metadata['quality_source'] = 'cached_portrait'
    metadata['quality'] = quality


def score_metadata(metadata, quality, policy):
    metadata.update(quality=quality, score=portrait_score(quality, policy),
                    quality_tiebreak=portrait_tiebreak(quality, policy),
                    quality_shortfalls=portrait_shortfalls(quality, policy))


def observation_identity(database, row, groups, target, config, context):
    """A current accepted assignment can illustrate its own imperfect cluster.

    A stale/historical membership cannot override a current ambiguous decision.
    The fallback requires this policy and either the active reference fingerprint
    or an accepted assignment captured during this run before later scenes.
    Decoded video faces still need the independent embedding comparison.
    """
    from .clustering import vector, matching_policy
    from .identity_merges import canonical
    from .portraits import identity_reason
    person = target['person']
    reason, similarity, runner = identity_reason(vector(row), groups, person, config)
    basis = 'retained_embedding_comparison'
    if reason:
        identifier = str(uuid.uuid5(uuid.UUID(row['id']), matching_policy(config)))
        candidate = database.connection.execute('SELECT person,status,payload FROM candidates WHERE id=? AND scan=?',
                                                (identifier, row['scan'])).fetchone()
        if candidate and candidate['person'] == person and candidate['status'] == 'accepted':
            accepted_context = json.loads(candidate['payload']).get('reference_context')
            proof = target.get('matching', {}).get(row['scan'], {})
            if accepted_context == context:
                basis, reason = 'current_accepted_assignment', None
            elif (proof.get('policy') == matching_policy(config) and proof.get('reference_context') == accepted_context
                  and canonical(database.connection, proof.get('person')) == person):
                # A later scene can add rivals after this run accepted a face.
                # Illustrate the accepted cluster instead of leaving it blank.
                basis, reason = 'current_run_accepted_assignment', None
    return reason, similarity, runner, basis


def retained_image(row):
    if not isinstance(row['crop'], bytes) or not row['crop']:
        raise ValueError('Recognition observation has no retained crop.')
    quality = json.loads(row['quality'])
    bounds = quality.get('crop_bounds')
    if (not isinstance(bounds, list) or len(bounds) != 4 or any(type(v) is not int for v in bounds)
            or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]):
        raise ValueError('Retained recognition crop has no valid source bounds.')
    geometry = json.loads(row['geometry'])
    geometry[0] -= bounds[0]
    geometry[1] -= bounds[1]
    for index in range(4, 14, 2):
        geometry[index] -= bounds[0]
        geometry[index + 1] -= bounds[1]
    face = checked_detection(geometry)
    with open_image(BytesIO(row['crop'])) as raw:
        if raw.size != (bounds[2] - bounds[0], bounds[3] - bounds[1]):
            raise ValueError('Retained crop dimensions do not match recognition evidence.')
        image = raw.convert('RGB')
    return image, face


def retained_portrait(row, metadata, policy, limit):
    image, face = retained_image(row)
    encoding = {}
    with image:
        data, bounds = portrait_bytes(image, face, policy, limit=limit, details=encoding)
    result = dict(metadata)
    result.pop('reason', None)
    result.update(source_type='recognition_crop', crop=bounds,
                  crop_coordinate_system='retained_recognition_crop', encoding=encoding,
                  pixels=encoding['pixels'], recognition_crop_bounds=json.loads(row['quality'])['crop_bounds'],
                  fallback={'from': metadata.get('requested_source', 'vtt'),
                            'reason': metadata.get('source_warning', 'original_source_unavailable')})
    return data, result


def cached_thumbnail(database, config, scope, target, row, identifier, metadata):
    """Re-score the same physical thumbnail JPEG under changed preferences."""
    db, policy = database.connection, config.data['portraits']
    old = db.execute('''SELECT * FROM portraits WHERE scope=? AND person=? AND scene=? AND image IS NOT NULL
      AND json_extract(metadata,'$.source_type')='vtt' AND json_extract(metadata,'$.namespace')=?
      AND json_extract(metadata,'$.scan')=? AND json_extract(metadata,'$.observation')=?
      AND json_extract(metadata,'$.policy.margin') IS ? AND json_extract(metadata,'$.policy.aspect_ratio') IS ?
      AND json_extract(metadata,'$.policy.jpeg_quality') IS ? ORDER BY rowid DESC LIMIT 1''',
      (scope, target['person'], metadata['scene'], metadata['namespace'], row['scan'], row['id'],
       policy['margin'], policy['aspect_ratio'], policy['jpeg_quality'])).fetchone()
    if (old is None or len(old['image']) > config.data['storage']['max_crop_bytes']
            or hashlib.sha256(old['image']).hexdigest() != old['digest']):
        return None
    before = json.loads(old['metadata'])
    if before.get('input_sha') != metadata['input_sha'] or before.get('geometry') != metadata['geometry']:
        return None
    for field in ('crop', 'pixels', 'encoding', 'crop_coordinate_system'):
        if field in before:
            metadata[field] = before[field]
    metadata.pop('recognition_crop_bounds', None)
    metadata.pop('source_warning', None)
    if policy['source'] == 'vtt':
        metadata.pop('fallback', None)
    else:
        metadata['fallback'] = {'from': 'video', 'reason': 'no_usable_identity_verified_video_portrait'}
    metadata.update(source_type='vtt', reused_from=old['id'], crop_coordinate_system='thumbnail')
    from .database import encoded
    with db:
        db.execute('INSERT OR REPLACE INTO portraits VALUES (?,?,?,?,?,?,?,?)',
                   (identifier, scope, target['person'], metadata['scene'], encoded(metadata), old['image'], old['digest'], 'selected'))
    return db.execute('SELECT * FROM portraits WHERE id=?', (identifier,)).fetchone()
