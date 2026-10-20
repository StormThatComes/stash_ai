"""Measured crop quality and geometric full-face/pose estimates, not attributes."""
from io import BytesIO
import math


class PortraitFramingError(ValueError):
    """The available pixels cannot produce a bounded portrait image."""


# AuraFace/SFace alignment uses the same canonical 112x112 five-point template.
# Full-frame coverage alone is too strict for a legitimate face near a crop
# boundary: require useful real pixels around its eyes, nose and mouth too.
MIN_ALIGNMENT_SOURCE_COVERAGE = .50
MIN_ALIGNMENT_FACE_COVERAGE = .85
ALIGNMENT_FACE_ROWS = slice(36, 105)
ALIGNMENT_FACE_COLUMNS = slice(24, 88)


def alignable(detection, image=None):
    """Basic recognizer alignment validity, independent of optional pose gates."""
    from .aura import similarity_matrix
    try:
        matrix = similarity_matrix(detection.landmarks)
    except ValueError:
        return False
    if image is None:
        return True
    import cv2
    import numpy as np
    inverse = cv2.invertAffineTransform(matrix)
    # A finite affine transform (or even a single surviving pixel) is not
    # sufficient. Measure source-pixel coverage in the full aligned crop and
    # in its canonical facial landmark region, excluding synthetic padding.
    rows, columns = np.ogrid[:112, :112]
    x = inverse[0, 0] * columns + inverse[0, 1] * rows + inverse[0, 2]
    y = inverse[1, 0] * columns + inverse[1, 1] * rows + inverse[1, 2]
    real = (x >= 0) & (x < image.width) & (y >= 0) & (y < image.height)
    return bool(real.mean() >= MIN_ALIGNMENT_SOURCE_COVERAGE and
                real[ALIGNMENT_FACE_ROWS, ALIGNMENT_FACE_COLUMNS].mean() >= MIN_ALIGNMENT_FACE_COVERAGE)


def pose_measurements(landmarks):
    """Yaw/pitch proxies in the eye coordinate frame; roll is aligned away."""
    import numpy as np
    points = np.asarray(landmarks, dtype=np.float64)
    eye = points[1] - points[0]
    distance = float(np.linalg.norm(eye))
    if distance <= 1:
        return dict(pose_deviation=1., roll_deviation=1., pose_contract='eye-frame-v2')
    horizontal = eye / distance
    vertical = np.array([-horizontal[1], horizontal[0]])
    origin = (points[0] + points[1]) / 2
    nose = points[2] - origin
    mouth = (points[3] + points[4]) / 2 - origin
    yaw = abs(float(nose @ horizontal)) / distance
    mouth_height = float(mouth @ vertical)
    pitch = abs(float(nose @ vertical) / max(mouth_height, 1.) - .5)
    return dict(pose_deviation=max(yaw, pitch),
                roll_deviation=abs(float(eye[1])) / distance, pose_contract='eye-frame-v2')


def assess(image, detection):
    import numpy as np
    x, y, width, height = detection.box
    complete = x >= 0 and y >= 0 and x + width <= image.width and y + height <= image.height
    bounds = (max(0, int(x)), max(0, int(y)), min(image.width, math.ceil(x + width)),
              min(image.height, math.ceil(y + height)))
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        return dict(complete=False, landmarks_inside=False, alignment_valid=False, pixels=0, sharpness=0.,
                    exposure=0., clipped_fraction=1., pose_deviation=1.)
    with image.crop(bounds) as crop, crop.convert('L') as grey, grey.resize((112, 112)) as resized:
        values = np.asarray(resized, dtype=np.float32)
        laplacian = (values[:-2, 1:-1] + values[2:, 1:-1] + values[1:-1, :-2]
                     + values[1:-1, 2:] - 4 * values[1:-1, 1:-1])
        sharpness = float(laplacian.var())
        exposure = float(values.mean())
        clipped = float(((values <= 5) | (values >= 250)).mean())
    points = np.asarray(detection.landmarks, dtype=np.float32)
    inside = bool(((points[:, 0] >= x) & (points[:, 0] <= x + width)
                   & (points[:, 1] >= y) & (points[:, 1] <= y + height)).all())
    eye_distance = float(np.linalg.norm(points[0] - points[1]))
    eye_mid, mouth_mid = (points[0] + points[1]) / 2, (points[3] + points[4]) / 2
    ordered = bool(eye_distance > 1 and mouth_mid[1] > eye_mid[1] and
                   eye_mid[1] < points[2, 1] < mouth_mid[1])
    pose = pose_measurements(points)
    return dict(complete=complete, landmarks_inside=inside and ordered, alignment_valid=alignable(detection, image),
                pixels=float(min(width, height)), sharpness=sharpness, exposure=exposure,
                clipped_fraction=clipped, **pose,
                visibility_method='five-landmark-geometry-estimate-v1')


def rejection(quality, policy):
    """Hard detection gates only; portrait quality is a ranking preference."""
    if not quality.get('alignment_valid', True):
        return 'face_landmarks_invalid'
    if quality['pixels'] < policy['min_face_pixels']:
        return 'face_too_small'
    if quality['sharpness'] < policy['min_sharpness']:
        return 'face_blurred'
    if policy['require_complete_face'] and not quality['complete']:
        return 'face_incomplete'
    if policy['max_pose_deviation'] is not None:
        if not quality['landmarks_inside']:
            return 'face_landmarks_invalid'
        if quality['pose_deviation'] > policy['max_pose_deviation']:
            return 'face_pose'
    return None


def crop_bytes(image, detection, *, margin=.15, limit=1048576, details=None):
    x, y, width, height = detection.box
    bounds = (max(0, int(x - width * margin)), max(0, int(y - height * margin)),
              min(image.width, math.ceil(x + width * (1 + margin))),
              min(image.height, math.ceil(y + height * (1 + margin))))
    with image.crop(bounds) as crop, crop.convert('RGB') as rgb:
        data = encode_portrait(rgb, {'jpeg_quality': 90}, limit=limit, details=details)
    return data, list(bounds)


def portrait_bytes(image, detection, policy, *, limit, details=None):
    """Frame from the full source; padding may shrink at its edges, never upscale."""
    x, y, width, height = detection.box
    if min(image.width, x + width) <= max(0., x) or min(image.height, y + height) <= max(0., y):
        raise PortraitFramingError('Portrait has no real face pixels.')
    wanted_width, wanted_height = width * (1 + 2 * policy['margin']), height * (1 + 2 * policy['margin'])
    ratio = policy['aspect_ratio']
    if ratio is not None:
        wanted_width, wanted_height = max(wanted_width, wanted_height * ratio), max(wanted_height, wanted_width / ratio)
        scale = min(1., image.width / wanted_width, image.height / wanted_height)
        wanted_width, wanted_height = wanted_width * scale, wanted_height * scale
        if wanted_width < width or wanted_height < height:
            # Preserve the visible face instead of discarding it to enforce an
            # impossible ratio. The requested ratio remains in policy metadata.
            wanted_width, wanted_height = width * (1 + 2 * policy['margin']), height * (1 + 2 * policy['margin'])
            if details is not None:
                details['aspect_ratio_adjusted'] = True
    wanted_width, wanted_height = min(image.width, wanted_width), min(image.height, wanted_height)
    left = max(0., min(x + width / 2 - wanted_width / 2, image.width - wanted_width))
    top = max(0., min(y + height / 2 - wanted_height / 2, image.height - wanted_height))
    bounds = (int(left), int(top), min(image.width, math.ceil(left + wanted_width)),
              min(image.height, math.ceil(top + wanted_height)))
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise PortraitFramingError('Portrait has no real source pixels.')
    if details is not None:
        details['padding_adjusted'] = any(padding + .01 < wanted for padding, wanted in (
            (x - bounds[0], width * policy['margin']), (bounds[2] - x - width, width * policy['margin']),
            (y - bounds[1], height * policy['margin']), (bounds[3] - y - height, height * policy['margin'])))
    with image.crop(bounds) as crop, crop.convert('RGB') as rgb:
        data = encode_portrait(rgb, policy, limit=limit, details=details)
    return data, list(bounds)


def encode_portrait(image, policy, *, limit, details=None):
    """Encode existing RGB pixels within the byte budget without changing framing."""
    rgb = image
    qualities = sorted({policy['jpeg_quality'], *[min(policy['jpeg_quality'], q) for q in (75, 50, 25, 10, 1)]}, reverse=True)
    data = None
    for jpeg_quality in qualities:
        output = BytesIO()
        rgb.save(output, format='JPEG', quality=jpeg_quality)
        data = output.getvalue()
        if len(data) <= limit:
            break
    resized = rgb
    try:
        while len(data) > limit and resized.size != (1, 1):
            factor = min(.9, math.sqrt(limit / len(data)) * .9)
            smaller = resized.resize((max(1, int(resized.width * factor)), max(1, int(resized.height * factor))))
            if resized is not rgb:
                resized.close()
            resized = smaller
            output = BytesIO()
            resized.save(output, format='JPEG', quality=jpeg_quality)
            data = output.getvalue()
        if details is not None:
            details.update(jpeg_quality=jpeg_quality, pixels=list(resized.size),
                           downscaled=resized.size != rgb.size)
    finally:
        if resized is not rgb:
            resized.close()
    if len(data) > limit:
        raise PortraitFramingError('Portrait cannot fit the configured byte budget.')
    return data


def portrait_score(quality, policy, *, expression=True):
    parts = portrait_components(quality, policy)
    return _weighted_score(quality, policy, parts, expression=expression)


def _weighted_score(quality, policy, parts, *, expression=True, expression_valid=None):
    weights = policy['weights']
    if expression:
        from .expression import WEIGHTS, required, cached
        if required(policy):
            valid = cached(quality) if expression_valid is None else expression_valid
            parts.update({name: quality[name] if valid else 0. for name in WEIGHTS})
    total = sum(weights[k] for k in parts)
    return sum(parts[k] * weights[k] for k in parts) / total if total else 0.


def _number(quality, name):
    value = quality.get(name, 0.)
    if type(value) not in (float, int):
        return 0.
    try:
        value = float(value)
    except OverflowError:
        return 0.
    return max(0., value) if math.isfinite(value) else 0.


def _attainment(value, goal):
    return min(1., value / goal) if goal > 0 else 1.


def _upper_attainment(value, goal):
    return 1 / (1 + max(0., value - goal) / max(.01, goal))


def portrait_components(quality, policy, *, _visibility_valid=None, _numbers=None):
    """Bounded soft scores: missing a preferred target never rejects a face."""
    from .visibility import cached
    number = (lambda name: _number(quality, name)) if _numbers is None else _numbers.__getitem__
    pixels, sharpness, pose = (number(name) for name in ('pixels', 'sharpness', 'pose_deviation'))
    exposure = number('exposure')
    middle = (policy['min_exposure'] + policy['max_exposure']) / 2
    spread = max(1., (policy['max_exposure'] - policy['min_exposure']) / 2)
    clipped = number('clipped_fraction')
    valid = cached(quality) if _visibility_valid is None else _visibility_valid
    visibility = quality['visible_face_fraction'] if valid else 0.
    frontal = 1 / (1 + pose) * _upper_attainment(pose, policy['max_pose_deviation'])
    if not quality.get('complete') or not quality.get('landmarks_inside'):
        frontal *= .5
    return {'frontal': frontal,
            'sharpness': min(1., math.log1p(sharpness) / math.log1p(1000)) * _attainment(sharpness, policy['min_sharpness']),
            'resolution': min(1., pixels / 512) * _attainment(pixels, policy['min_face_pixels']),
            'exposure': 1 / (1 + abs(exposure - middle) / spread) * _upper_attainment(clipped, policy['max_clipped_fraction']),
            'visibility': visibility * _attainment(visibility, policy['min_visible_face_fraction'])}


def portrait_tiebreak(quality, policy):
    parts = portrait_components(quality, policy)
    return _tiebreak(parts)


def _tiebreak(parts):
    weights = {'frontal': 3., 'sharpness': 2., 'resolution': 1., 'exposure': 1., 'visibility': 3.}
    return sum(parts[name] * weight for name, weight in weights.items()) / sum(weights.values())


def portrait_shortfalls(quality, policy, *, _visibility_valid=None, _expression_valid=None, _numbers=None):
    """Explain imperfect candidates without making these flags exclusions."""
    from .visibility import required, cached
    from .expression import required as expression_required, cached as expression_cached
    number = (lambda name: _number(quality, name)) if _numbers is None else _numbers.__getitem__
    result = []
    for name, missed in (
        ('face_too_small', number('pixels') < policy['min_face_pixels']),
        ('face_blurred', number('sharpness') < policy['min_sharpness']),
        ('face_incomplete', not quality.get('complete') or not quality.get('landmarks_inside')),
        ('face_pose', number('pose_deviation') > policy['max_pose_deviation']),
        ('face_exposure', not policy['min_exposure'] <= number('exposure') <= policy['max_exposure']),
        ('face_clipped_exposure', number('clipped_fraction') > policy['max_clipped_fraction']),
    ):
        if missed:
            result.append(name)
    if required(policy):
        if not (cached(quality) if _visibility_valid is None else _visibility_valid):
            result.append('face_visibility_unavailable')
        elif quality['visible_face_fraction'] < policy['min_visible_face_fraction']:
            result.append('face_occluded')
    if expression_required(policy) and not (expression_cached(quality) if _expression_valid is None else _expression_valid):
        result.append('face_expression_unavailable')
    return result


def portrait_evaluation(quality, policy):
    """Evaluate one immutable quality record without repeated component work."""
    from .visibility import cached
    from .expression import required, cached as expression_cached
    visible = cached(quality)
    expression = expression_cached(quality) if required(policy) else False
    numbers = {name: _number(quality, name) for name in
               ('pixels', 'sharpness', 'pose_deviation', 'exposure', 'clipped_fraction')}
    parts = portrait_components(quality, policy, _visibility_valid=visible, _numbers=numbers)
    return {'score': _weighted_score(quality, policy, dict(parts), expression_valid=expression),
            'quality_tiebreak': _tiebreak(parts),
            'quality_shortfalls': portrait_shortfalls(quality, policy, _visibility_valid=visible,
                                                     _expression_valid=expression, _numbers=numbers)}
