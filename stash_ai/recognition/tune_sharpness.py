"""Optional, inference-free blur preview on encoder-sized inputs."""
from .sharpness_gates import FIELDS, failure

EDITABLE = frozenset(scope+'.'+key for scope in ('face.quality', 'body.collection')
                     for key in (*FIELDS, 'min_sharpness'))


def preview(image, faces, people, data):
    result = []
    if data['detection'].get('sharpness_mode') == 'split':
        from .foreground_quality import face_geometry
        for index, face in enumerate(faces):
            quality = face_geometry(image, face)
            result.append(dict(label=f'F{index}', region='ellipse preview (FaceOcc not applied)',
                               quality=quality, failed=failure(quality, data['detection'])))
    if data['sapiensid']['collection'].get('sharpness_mode') == 'split':
        from .body_collection import quality as body_quality
        for index, person in enumerate(people):
            quality, _ = body_quality(image, person, people, data['sapiensid']['collection'])
            result.append(dict(label=f'P{index}', region=quality['sharpness_method'],
                               quality=quality, failed=failure(quality, data['sapiensid']['collection'])))
    return result


def caption(rows):
    def value(q, key):
        v = q.get(key)
        return 'unavailable' if v is None else f'{v:.1f}'
    return '\n'.join(f"{row['label']} focus {value(row['quality'], 'focus_sharpness')} | "
                     f"motion {value(row['quality'], 'motion_sharpness')} | "
                     f"confidence {value(row['quality'], 'sharpness_confidence')} | "
                     f"{row['region']} | {row['failed'] or 'blur gates pass'}" for row in rows)
