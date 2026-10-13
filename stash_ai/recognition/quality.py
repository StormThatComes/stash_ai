"""Measured crop quality and geometric full-face/pose estimates, not attributes."""
from io import BytesIO
import math


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


def rejection(quality, policy, *, portrait=False, visibility=True):
    if quality['pixels'] < policy['min_face_pixels']:
        return 'face_too_small'
    if quality['sharpness'] < policy['min_sharpness']:
        return 'face_blurred'
    if portrait:
        if not quality['complete'] or not quality['landmarks_inside']:
            return 'face_incomplete'
        if quality['pose_deviation'] > policy['max_pose_deviation']:
            return 'face_pose'
        if not policy['min_exposure'] <= quality['exposure'] <= policy['max_exposure']:
            return 'face_exposure'
        if quality['clipped_fraction'] > policy['max_clipped_fraction']:
            return 'face_clipped_exposure'
        if visibility:
            from .visibility import required, valid_fraction
            if required(policy):
                fraction = quality.get('visible_face_fraction')
                if not valid_fraction(fraction):
                    return 'face_visibility_unavailable'
                if fraction < policy['min_visible_face_fraction']:
                    return 'face_occluded'
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


def portrait_bytes(image, detection, policy, *, limit):
    """Frame from the full source; padding may shrink at its edges, never upscale."""
    x, y, width, height = detection.box
    wanted_width, wanted_height = width * (1 + 2 * policy['margin']), height * (1 + 2 * policy['margin'])
    ratio = policy['aspect_ratio']
    if ratio is not None:
        wanted_width, wanted_height = max(wanted_width, wanted_height * ratio), max(wanted_height, wanted_width / ratio)
        scale = min(1., image.width / wanted_width, image.height / wanted_height)
        wanted_width, wanted_height = wanted_width * scale, wanted_height * scale
        if wanted_width < width or wanted_height < height:
            raise ValueError('Portrait aspect ratio cannot contain the full face in this source.')
    wanted_width, wanted_height = min(image.width, wanted_width), min(image.height, wanted_height)
    left = max(0., min(x + width / 2 - wanted_width / 2, image.width - wanted_width))
    top = max(0., min(y + height / 2 - wanted_height / 2, image.height - wanted_height))
    bounds = (int(left), int(top), min(image.width, math.ceil(left + wanted_width)),
              min(image.height, math.ceil(top + wanted_height)))
    with image.crop(bounds) as crop, crop.convert('RGB') as rgb:
        output = BytesIO()
        rgb.save(output, format='JPEG', quality=policy['jpeg_quality'])
        data = output.getvalue()
    if len(data) > limit:
        raise ValueError('Portrait exceeds configured BLOB limit.')
    return data, list(bounds)


def portrait_score(quality, policy):
    weights = policy['weights']
    parts = {'frontal': max(0., 1 - quality['pose_deviation']),
             'sharpness': min(1., math.log1p(quality['sharpness']) / math.log1p(1000)),
             'resolution': min(1., quality['pixels'] / 512),
             'exposure': max(0., 1 - abs(quality['exposure'] - 128) / 128),
             'visibility': quality.get('visible_face_fraction', 0.)}
    return sum(parts[k] * weights[k] for k in parts) / sum(weights.values())
