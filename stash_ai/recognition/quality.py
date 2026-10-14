"""Measured crop quality and geometric full-face/pose estimates, not attributes."""
from io import BytesIO
import math


class PortraitFramingError(ValueError):
    """The available pixels cannot produce a bounded portrait image."""


def assess(image, detection):
    import numpy as np
    x, y, width, height = detection.box
    complete = x >= 0 and y >= 0 and x + width <= image.width and y + height <= image.height
    bounds = (max(0, int(x)), max(0, int(y)), min(image.width, math.ceil(x + width)),
              min(image.height, math.ceil(y + height)))
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        return dict(complete=False, landmarks_inside=False, pixels=0, sharpness=0.,
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
    yaw = abs(float(points[2, 0] - eye_mid[0])) / max(eye_distance, 1)
    roll = abs(float(points[0, 1] - points[1, 1])) / max(eye_distance, 1)
    pitch = abs(float((points[2, 1] - eye_mid[1]) / max(float(mouth_mid[1] - eye_mid[1]), 1)) - .5)
    return dict(complete=complete, landmarks_inside=inside and ordered,
                pixels=float(min(width, height)), sharpness=sharpness, exposure=exposure,
                clipped_fraction=clipped, pose_deviation=max(yaw, roll, pitch),
                visibility_method='five-landmark-geometry-estimate-v1')


def rejection(quality, policy):
    """Hard detection gates only; portrait quality is a ranking preference."""
    if quality['pixels'] < policy['min_face_pixels']:
        return 'face_too_small'
    if quality['sharpness'] < policy['min_sharpness']:
        return 'face_blurred'
    return None


def crop_bytes(image, detection, *, margin=.15, limit=1048576):
    x, y, width, height = detection.box
    bounds = (max(0, int(x - width * margin)), max(0, int(y - height * margin)),
              min(image.width, math.ceil(x + width * (1 + margin))),
              min(image.height, math.ceil(y + height * (1 + margin))))
    with image.crop(bounds) as crop, crop.convert('RGB') as rgb:
        output = BytesIO()
        rgb.save(output, format='JPEG', quality=90)
        data = output.getvalue()
    if len(data) > limit:
        raise ValueError('Retained face crop exceeds configured BLOB limit.')
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
    weights = policy['weights']
    parts = portrait_components(quality, policy)
    if expression:
        from .expression import WEIGHTS, required, cached
        if required(policy):
            parts.update({name: quality[name] if cached(quality) else 0. for name in WEIGHTS})
    total = sum(weights[k] for k in parts)
    return sum(parts[k] * weights[k] for k in parts) / total if total else 0.


def _number(quality, name):
    value = quality.get(name, 0.)
    return max(0., float(value)) if type(value) in (float, int) and math.isfinite(value) else 0.


def _attainment(value, goal):
    return min(1., value / goal) if goal > 0 else 1.


def _upper_attainment(value, goal):
    return 1 / (1 + max(0., value - goal) / max(.01, goal))


def portrait_components(quality, policy):
    """Bounded soft scores: missing a preferred target never rejects a face."""
    from .visibility import cached
    pixels, sharpness, pose = (_number(quality, name) for name in ('pixels', 'sharpness', 'pose_deviation'))
    exposure = _number(quality, 'exposure')
    middle = (policy['min_exposure'] + policy['max_exposure']) / 2
    spread = max(1., (policy['max_exposure'] - policy['min_exposure']) / 2)
    clipped = _number(quality, 'clipped_fraction')
    visibility = quality['visible_face_fraction'] if cached(quality) else 0.
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
    weights = {'frontal': 3., 'sharpness': 2., 'resolution': 1., 'exposure': 1., 'visibility': 3.}
    return sum(parts[name] * weight for name, weight in weights.items()) / sum(weights.values())


def portrait_shortfalls(quality, policy):
    """Explain imperfect candidates without making these flags exclusions."""
    from .visibility import required, cached
    from .expression import required as expression_required, cached as expression_cached
    result = []
    for name, missed in (
        ('face_too_small', _number(quality, 'pixels') < policy['min_face_pixels']),
        ('face_blurred', _number(quality, 'sharpness') < policy['min_sharpness']),
        ('face_incomplete', not quality.get('complete') or not quality.get('landmarks_inside')),
        ('face_pose', _number(quality, 'pose_deviation') > policy['max_pose_deviation']),
        ('face_exposure', not policy['min_exposure'] <= _number(quality, 'exposure') <= policy['max_exposure']),
        ('face_clipped_exposure', _number(quality, 'clipped_fraction') > policy['max_clipped_fraction']),
    ):
        if missed:
            result.append(name)
    if required(policy):
        if not cached(quality):
            result.append('face_visibility_unavailable')
        elif quality['visible_face_fraction'] < policy['min_visible_face_fraction']:
            result.append('face_occluded')
    if expression_required(policy) and not expression_cached(quality):
        result.append('face_expression_unavailable')
    return result
