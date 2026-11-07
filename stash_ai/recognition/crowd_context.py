"""Bounded temporal seeds for recovering two torsos inside one detected person box."""
import json
import math


def torso(person):
    if not person.get('keypoints'):
        return None
    points = []
    for i in (5,6,11,12):
        point = person['keypoints'][i]
        visibility = person.get('keypoint_visibility')
        if min(point)>=0 and (visibility is None or visibility[i]>=.5):
            points.append((i,point))
    if not any(i<7 for i,_ in points) or not any(i>10 for i,_ in points):
        return None
    return tuple(sum(p[axis] for _,p in points)/len(points) for axis in (0,1))


def pair(people):
    if len(people)!=2 or any(p.get('crowded_box_retry') or p.get('pose_refinement',{}).get('mask_conflict') for p in people):
        return None
    centres = [torso(p) for p in people]
    if None in centres:
        return None
    scale = min(math.hypot(b[2]-b[0],b[3]-b[1]) for b in (p['bounds'] for p in people))
    if scale<=0 or math.dist(*centres)<max(4.,.12*scale):
        return None
    return centres,scale


def follows(first, second):
    if not first or not second:
        return False
    before, after = first[0], second[0]
    scale = min(first[1],second[1])
    a,b = sorted(max(math.dist(p,q) for p,q in zip(before,order))
                 for order in (after, after[::-1]))
    return a<=.30*scale and b-a>=.06*scale


def attach_context(model, image, scan, cue, seconds, *, db=None, contract=None):
    # Store only two small summaries. At a sparse resume boundary recover the
    # same previous poses from committed cue records as in a continuous run.
    captured = getattr(image,'_recognition_cue_signature',None)
    if not captured or captured[0]!=cue or captured[1] is None:
        return
    shot = captured[1]
    history = getattr(model,'_crowd_history',{})
    prior = []
    for index in (cue-2,cue-1):
        entry = history.get((scan,index))
        if entry is None and db is not None and index>=0:
            records = list(db.execute('SELECT seconds,geometry,quality FROM body_observations '
                                      'WHERE scan=? AND contract=? AND cue=? ORDER BY id',
                                      (scan,contract,index)))
            if len(records)==2:
                qualities = [json.loads(r[2])['quality'] for r in records]
                if all(q.get('view_shot')==shot and q.get('detected_people')==2 for q in qualities):
                    entry = shot,records[0][0],pair([json.loads(r[1]) for r in records])
        prior.append(entry)
    support = None
    if all(prior) and all(item[0]==shot for item in prior):
        first,second = prior
        if 0<second[1]-first[1]<=12 and 0<seconds-second[1]<=12 and follows(first[2],second[2]):
            support = second[2]
    image._recognition_crowd = (scan,cue,shot,seconds,support)


def remember(model, image, people):
    context = getattr(image,'_recognition_crowd',None)
    if context is None:
        return
    scan,cue,shot,seconds,_ = context
    history = getattr(model,'_crowd_history',{})
    model._crowd_history = {k:v for k,v in history.items() if k==(scan,cue-1)}
    model._crowd_history[scan,cue] = shot,seconds,pair(people)
