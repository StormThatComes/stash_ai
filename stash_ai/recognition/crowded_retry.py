"""A bounded second person-detector pass for two heads in one source box."""
import math
import numpy as np


def retry_people(model, image, faces, people):
    from .sapiensid_backend import associate
    if len(faces) != 2 or len(people) != 1 or getattr(model, 'pose_refiner', None) is None:
        return people
    settings = model.config.data['sapiensid']['pose']
    if settings['max_people_per_frame'] < 2:
        return people
    left, top, right, bottom = people[0]['bounds']
    heads = [(x+w/2, y+h/2) for x,y,w,h in (f.box for f in faces)]
    if (not all(left <= x < right and top <= y < bottom for x,y in heads)
            or math.dist(*heads) < max(4., .7 * max(min(f.box[2:]) for f in faces))):
        return people

    # A padded crop changes detector scale while preserving both heads. Only
    # replace the original after two independent poses confirm the two faces.
    pad = max(2, round(.05 * max(right-left, bottom-top)))
    x0, y0 = max(0, left-pad), max(0, top-pad)
    x1, y1 = min(image.width, right+pad), min(image.height, bottom+pad)
    if (x0,y0,x1,y1) == (0,0,image.width,image.height):
        x0,y0,x1,y1 = left,top,right,bottom
    with image.crop((x0,y0,x1,y1)) as crop:
        [raw] = model._predict_pose(crop)
        boxes = raw.boxes.xyxy.detach().cpu().numpy()
        scores = raw.boxes.conf.detach().cpu().numpy()
        if (scores.ndim != 1 or boxes.shape != (len(scores),4) or not np.isfinite(boxes).all()
                or not np.isfinite(scores).all() or np.any((scores<0)|(scores>1))):
            raise ValueError('Invalid local person detector output.')
        candidates = []
        for box, score in zip(boxes, scores):
            if score <= settings['score_threshold']:
                continue
            bounds = [max(x0, x0+math.floor(box[0])), max(y0,y0+math.floor(box[1])),
                      min(x1,x0+math.ceil(box[2])), min(y1,y0+math.ceil(box[3]))]
            if any(bounds[0]<=x<bounds[2] and bounds[1]<=y<bounds[3] for x,y in heads):
                candidates.append((bounds, float(score)))
    if len(candidates) != 2:
        return people
    bounds, scores = zip(*candidates)
    found = model._people(list(bounds), list(scores), *model.pose_refiner.predict(image, list(bounds)))
    matches = associate(faces, found)
    if None in matches or len(set(matches)) != 2:
        return people
    for person in found:
        person['crowded_box_retry'] = True
    return found
