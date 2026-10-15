"""Optional quality measurements and truthful fallback to retained recognition pixels."""
from io import BytesIO
import hashlib
import json
import uuid
import itertools

from stash_ai.inference.image_io import open_image
from .diagnostics import message as safe_error
from .models import checked_detection
from .quality import portrait_bytes, portrait_evaluation
from .visibility import required as visibility_required, cached as visibility_cached
from .expression import required as expression_required, cached as expression_cached


def measure_available(client, image, face, quality, policy, visibility, expression, *, force=()):
    errors = {}
    for name, required, cached, model in (
        ('visibility', visibility_required, visibility_cached, visibility),
        ('expression', expression_required, expression_cached, expression),
    ):
        if not required(policy) or (cached(quality) and name not in force):
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


def measurement_basis(row, source_type):
    """Identify the pixels/geometry behind an optional observation measurement."""
    from .config import digest
    inputs = [row['input_sha'], row['cue'], row['seconds'], row['source'], row['rect'], row['geometry']]
    if source_type == 'recognition_crop':
        inputs.append(hashlib.sha256(row['crop']).hexdigest())
    return {'source_type': source_type, 'identity': digest(inputs)}


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
    metadata.update(quality=quality, **portrait_evaluation(quality, policy))


_UNREAD = object()


def candidate_observations(db, rows, scan, policy):
    """Read exact UUID-addressed assignment proofs in bounded batches."""
    from .database import encoded
    iterator = iter(rows)
    while batch := list(itertools.islice(iterator, 256)):
        cache = {}
        def proof(row, batch=batch, cache=cache):
            identifier = str(uuid.uuid5(uuid.UUID(row['id']), policy))
            if 'candidates' not in cache:
                identifiers = []
                for item in batch:
                    try:
                        identifiers.append(str(uuid.uuid5(uuid.UUID(item['id']), policy)))
                    except (ValueError, TypeError, AttributeError):
                        pass  # Only a requested invalid proof may fail its observation.
                cache['candidates'] = {item['id']: item for item in db.execute(
                    'SELECT id,person,status,payload FROM candidates WHERE scan=? AND id IN '
                    '(SELECT value FROM json_each(?))', (scan, encoded(identifiers)))}
            return cache['candidates'].get(identifier)
        for row in batch:
            yield row, lambda row=row, proof=proof: proof(row)


def crop_row(db, row):
    if 'crop' in row.keys():
        return row
    result = dict(row)
    result['crop'] = db.execute('SELECT crop FROM observations WHERE id=?', (row['id'],)).fetchone()[0]
    return result


def observation_identity(database, row, groups, target, config, context, *, candidate=_UNREAD):
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
        if callable(candidate):
            candidate = candidate()
        if candidate is _UNREAD:
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
    source_size = bounds[2] - bounds[0], bounds[3] - bounds[1]
    pixels = source_size
    encoding = quality.get('crop_encoding')
    if encoding is not None:
        pixels = encoding.get('pixels') if isinstance(encoding, dict) else None
        if (not isinstance(pixels, list) or len(pixels) != 2
                or any(type(value) is not int or not 1 <= value <= maximum
                       for value, maximum in zip(pixels, source_size))
                or encoding.get('downscaled') is not (tuple(pixels) != source_size)):
            raise ValueError('Retained crop has invalid encoding dimensions.')
    with open_image(BytesIO(row['crop'])) as raw:
        if raw.size != tuple(pixels):
            raise ValueError('Retained crop dimensions do not match recognition evidence.')
        scale_x, scale_y = raw.width / source_size[0], raw.height / source_size[1]
        geometry = json.loads(row['geometry'])
        geometry[0], geometry[1] = (geometry[0] - bounds[0]) * scale_x, (geometry[1] - bounds[1]) * scale_y
        geometry[2], geometry[3] = geometry[2] * scale_x, geometry[3] * scale_y
        for index in range(4, 14, 2):
            geometry[index] = (geometry[index] - bounds[0]) * scale_x
            geometry[index + 1] = (geometry[index + 1] - bounds[1]) * scale_y
        face = checked_detection(geometry)
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


def fit_cached_bytes(data, sha, config, metadata):
    """Verify historical bytes and re-encode their pixels for a smaller budget."""
    if (not isinstance(data, bytes) or not data or len(data) > 8388608
            or hashlib.sha256(data).hexdigest() != sha):
        raise ValueError('Cached portrait bytes do not match their bounded digest.')
    if len(data) <= config.data['storage']['max_crop_bytes']:
        return data, sha
    from .quality import encode_portrait
    encoding = dict(metadata.get('encoding', {}))
    was_downscaled = encoding.get('downscaled', False)
    with open_image(BytesIO(data)) as raw, raw.convert('RGB') as image:
        data = encode_portrait(image, config.data['portraits'],
                               limit=config.data['storage']['max_crop_bytes'], details=encoding)
    encoding.update(reencoded_from_cached_jpeg=True,
                    downscaled=was_downscaled or encoding['downscaled'])
    metadata.update(encoding=encoding, pixels=encoding['pixels'])
    return data, hashlib.sha256(data).hexdigest()


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
    if old is None:
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
    try:
        data, sha = fit_cached_bytes(old['image'], old['digest'], config, metadata)
    except (OSError, ValueError):
        return None
    from .database import encoded
    with db:
        db.execute('INSERT OR REPLACE INTO portraits VALUES (?,?,?,?,?,?,?,?)',
                   (identifier, scope, target['person'], metadata['scene'], encoded(metadata), data, sha, 'selected'))
    return db.execute('SELECT * FROM portraits WHERE id=?', (identifier,)).fetchone()
