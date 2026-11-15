"""Ownership evidence for PMPose prompts in original cue coordinates."""
from contextlib import contextmanager
from contextvars import ContextVar
import math

import numpy as np

_faces = ContextVar('pose_prompt_faces', default={})


@contextmanager
def face_inputs(frames):
    token = _faces.set({id(image): tuple(f for f in faces if hasattr(f, 'landmarks'))
                        for image, faces in frames})
    try:
        yield
    finally:
        _faces.reset(token)


def faces_for(image):
    return _faces.get().get(id(image), ())


def prompt_points(points, records):
    if not any('face_prompt' in r or 'temporal_prompts' in r for r in records):
        return points
    points = points.copy()
    for i, record in enumerate(records):
        for index,xy in record.get('temporal_prompts', ()):
            points[i,index] = xy
        if 'face_prompt' in record:
            index, xy = record['face_prompt']
            points[i, index] = xy
    return points


def assigned_faces(faces, bounds, points, valid):
    if not faces:
        return {}
    from .sapiensid_backend import associate
    people = [dict(bounds=box, keypoints=np.where(ok[:, None], xy, -1).tolist())
              for box, xy, ok in zip(bounds, points, valid)]
    return {owner: face for face, owner in zip(faces, associate(faces, people)) if owner is not None}


def eligible(points, scores, probabilities, bounds, settings):
    boxes = np.asarray(bounds).reshape(-1, 4)
    tests = dict(crop=((points >= boxes[:, None, :2]) & (points < boxes[:, None, 2:])).all(axis=2),
                 score=scores > settings['min_keypoint_score'],
                 presence=probabilities['presence'] > settings['min_presence'],
                 visibility=probabilities['visibility'] > settings['min_visibility'])
    if 'flip_agreement' in probabilities:
        tests['flip'] = probabilities['flip_agreement'] >= .5
    return np.logical_and.reduce(list(tests.values())), tests


def recover(i, indices, points, valid, scores, probabilities, bounds, face, foreign, settings, record):
    box = bounds[i]
    scale = math.hypot(box[2] - box[0], box[3] - box[1])
    clearance = max(3., .06 * scale)
    def distinct(xy):
        return not len(foreign) or np.linalg.norm(foreign - xy, axis=1).min() > clearance

    minimum = settings['min_prompt_points']
    if len(indices) < minimum and face is not None:
        for joint, landmark in ((0, 2), (1, 1), (2, 0)):
            xy = np.asarray(face.landmarks[landmark])
            x, y, w, h = face.box
            if (not valid[i, joint] and x <= xy[0] < x+w and y <= xy[1] < y+h
                    and box[0] <= xy[0] < box[2] and box[1] <= xy[1] < box[3] and distinct(xy)):
                record['face_prompt'] = [joint, xy.tolist()]
                indices = np.append(indices, joint)
                break
    if len(indices) < minimum:
        for joint, adjacent in ((7, (5, 9)), (8, (6, 10)), (13, (11, 15)), (14, (12, 16))):
            if not valid[i, joint] or not distinct(points[i, joint]):
                continue
            if any(valid[i, k] and distinct(points[i, k])
                   and 2 < np.linalg.norm(points[i, k] - points[i, joint]) < .55 * scale
                   for k in adjacent):
                indices = np.append(indices, joint)
    if len(indices) >= minimum:
        return indices, minimum
    if len(indices) == 1:
        k = indices[0]
        confirmed = ('face_prompt' in record or min(scores[i,k], probabilities['presence'][i,k],
                                                    probabilities['visibility'][i,k]) >= .75)
        # A peer with no reliable core cannot exclude ownership in a shared box.
        xy = record['face_prompt'][1] if 'face_prompt' in record else points[i,k]
        if confirmed and distinct(xy) and not any(j != i and b[0] <= xy[0] < b[2] and b[1] <= xy[1] < b[3]
                                                   for j,b in enumerate(bounds)):
            return indices, 1
    return indices, minimum
