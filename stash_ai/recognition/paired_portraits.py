"""Frame a face-associated portrait from its measured person, without inference."""
from io import BytesIO
import json
import math

from stash_ai.inference.image_io import open_image
from .portrait_framing import person_bounds
from .quality import PortraitFramingError, encode_portrait


FACE_FRAMING = {'kind': 'face'}


def framing(db, config, observation):
    from .portrait_precedence import pose_contract
    expected = pose_contract(config)
    row = db.execute("SELECT metadata FROM person_evidence WHERE observation=? AND contract=? "
                     "AND status='ready'", (observation, expected)).fetchone()
    if row is None:
        raise PortraitFramingError('Portrait has no current paired person evidence.')
    metadata = json.loads(row['metadata'])
    person = metadata.get('person')
    if metadata.get('identity_rejection') or not isinstance(person, dict):
        raise PortraitFramingError('Portrait person association is unavailable.')
    bounds = person.get('bounds')
    if (not isinstance(bounds, list) or len(bounds) != 4
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds)):
        raise PortraitFramingError('Portrait person bounds are invalid.')
    # PIL rounds both edges when retaining the original person crop.
    left, top, right, bottom = map(round, bounds)
    if not (0 <= left < right and 0 <= top < bottom) or (right-left)*(bottom-top) > 32*1048576:
        raise PortraitFramingError('Portrait person bounds are invalid.')
    return dict(kind='pose', version=1, contract=expected,
                geometry=dict(bounds=[left, top, right, bottom], keypoints=person.get('keypoints', [])))


def matches(metadata, expected):
    """Legacy face pixels must never become a pose crop merely by relabelling."""
    return metadata.get('framing', FACE_FRAMING) == expected


def source_bytes(image, framing, policy, *, limit, details=None):
    geometry = framing['geometry']
    left, top, right, bottom = geometry['bounds']
    if right > image.width or bottom > image.height:
        raise PortraitFramingError('Portrait person bounds exceed the verified thumbnail.')
    with image.crop((left, top, right, bottom)) as person:
        bounds = person_bounds(person.size, geometry, policy['aspect_ratio'])
        with person.crop(bounds) as crop, crop.convert('RGB') as rgb:
            data = encode_portrait(rgb, policy, limit=limit, details=details)
    return data, [bounds[0]+left, bounds[1]+top, bounds[2]+left, bounds[3]+top]


def retained_crop(db, observation, framing):
    from .person_signal import read
    row = read(db, observation, framing['contract'])
    if row is None or not isinstance(row['crop'], bytes) or not row['crop']:
        raise PortraitFramingError('Portrait has no retained paired person pixels.')
    return row['crop']


def retained_portrait(blob, framing, metadata, policy, limit):
    geometry = framing['geometry']
    left, top, right, bottom = geometry['bounds']
    details = {}
    with open_image(BytesIO(blob)) as image:
        if image.size != (right-left, bottom-top):
            raise PortraitFramingError('Retained person dimensions do not match the measured bounds.')
        bounds = person_bounds(image.size, geometry, policy['aspect_ratio'])
        with image.crop(bounds) as crop, crop.convert('RGB') as rgb:
            data = encode_portrait(rgb, policy, limit=limit, details=details)
    result = dict(metadata)
    result.pop('reason', None)
    result.pop('recognition_crop_bounds', None)
    result.update(source_type='person_crop', framing=framing, crop=bounds,
                  crop_coordinate_system='retained_person_crop', person_crop_bounds=geometry['bounds'],
                  encoding=details, pixels=details['pixels'],
                  fallback={'from': metadata.get('requested_source', 'vtt'),
                            'reason': metadata.get('source_warning', 'original_source_unavailable')})
    return data, result
