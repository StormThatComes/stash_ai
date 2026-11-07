"""A bounded second person-detector pass for two heads in one source box."""
import math
import numpy as np


def _regions(image, faces, heads, box):
    left,top,right,bottom = box
    pad = max(2, round(.05 * max(right-left, bottom-top)))
    padded = (max(0,left-pad),max(0,top-pad),min(image.width,right+pad),min(image.height,bottom+pad))
    full = (0,0,*image.size)
    if padded != full or tuple(box) != full:
        return [(padded if padded != full else box, heads)]

    # The same image would reproduce the same detector result. Use overlapping
    # head-conditioned tiles instead, preserving each complete target face.
    axis = 0 if abs(heads[0][0]-heads[1][0]) >= abs(heads[0][1]-heads[1][1]) else 1
    middle = sum(h[axis] for h in heads)/2
    regions = []
    for face,head in zip(faces,heads):
        region = list(full)
        begin, extent = face.box[axis], face.box[axis+2]
        if head[axis] < middle:
            region[axis+2] = min(full[axis+2], math.ceil(max(begin+extent, middle+.25*extent)))
        else:
            region[axis] = max(0, math.floor(min(begin,middle-.25*extent)))
        if tuple(region)==full:
            return []
        regions.append((region,[head]))
    return regions


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
    candidates = []
    regions = _regions(image,faces,heads,(left,top,right,bottom))
    for region, required_heads in regions:
        x0,y0,x1,y1 = region
        with image.crop(region) as crop:
            [raw] = model._predict_pose(crop)
            boxes = raw.boxes.xyxy.detach().cpu().numpy()
            scores = raw.boxes.conf.detach().cpu().numpy()
        if (scores.ndim != 1 or boxes.shape != (len(scores),4) or not np.isfinite(boxes).all()
                or not np.isfinite(scores).all() or np.any((scores<0)|(scores>1))):
            raise ValueError('Invalid local person detector output.')
        admitted = []
        for box, score in zip(boxes, scores):
            if score <= settings['score_threshold']:
                continue
            bounds = [max(x0, x0+math.floor(box[0])), max(y0,y0+math.floor(box[1])),
                      min(x1,x0+math.ceil(box[2])), min(y1,y0+math.ceil(box[3]))]
            if any(bounds[0]<=x<bounds[2] and bounds[1]<=y<bounds[3] for x,y in required_heads):
                admitted.append((bounds, float(score)))
        if len(admitted)!=len(required_heads):
            return people
        candidates.extend(admitted)
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
