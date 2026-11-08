"""Physical VTT labels shared by holdout evaluation and detector benchmarks."""
import json
import math
from collections import Counter

from .crowd_context import torso


def validate(labels,scenes):
    if not isinstance(labels,dict) or not set(labels)<=set(scenes):
        raise ValueError('visible_people must map selected scenes to labelled cues.')
    for cues in labels.values():
        if not isinstance(cues,dict):
            raise ValueError('Expected a map of VTT cue numbers.')
        for cue,people in cues.items():
            if not isinstance(cue,str) or not cue.isascii() or not cue.isdecimal() or str(int(cue))!=cue:
                raise ValueError('Expected a nonnegative canonical VTT cue number.')
            if not isinstance(people,list) or not people:
                raise ValueError('Each labelled cue needs at least one visible person.')
            seen = set()
            for person in people:
                if not isinstance(person,dict):
                    raise ValueError('Expected a visible person label.')
                truth,point = person.get('truth'),person.get('point')
                if not isinstance(truth,str) or not truth.strip() or truth in seen:
                    raise ValueError('Each visible person needs a distinct truth label per cue.')
                if (not isinstance(point,(tuple,list)) or len(point)!=2 or
                        any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in point)):
                    raise ValueError('Expected torso point [x,y] in source VTT pixels.')
                seen.add(truth)


def torso_owner(pose,people):
    center = torso(pose)
    if center is None:
        return None
    left,top,right,bottom = pose['bounds']
    scale = max(1.,math.hypot(right-left,bottom-top))
    ranked = sorted((math.dist(center,p['point'])/scale,p['truth']) for p in people)
    if ranked[0][0]>.25 or len(ranked)>1 and ranked[1][0]-ranked[0][0]<.05:
        return None
    return ranked[0][1]


def detection_counts(db,scans,labels,contract=None):
    counts = dict(labelled_cues=0,expected_people=0,missing_poses=0,duplicate_poses=0,
                  poses_without_face=0,ambiguous_poses=0)
    for scan in scans:
        for raw_cue,people in labels.get(scan['scene'],{}).items():
            cue = int(raw_cue)
            if cue>=scan['total']:
                raise ValueError(f"Label outside VTT range for scene {scan['scene']} cue {cue}.")
            detected = Counter()
            counts['labelled_cues'] += 1
            counts['expected_people'] += len(people)
            for geometry,face in db.execute('SELECT geometry,face FROM body_observations WHERE scan=? AND cue=? '
                    'AND (? IS NULL OR contract=?)',(scan['id'],cue,contract,contract)):
                truth = torso_owner(json.loads(geometry),people)
                if truth is None:
                    counts['ambiguous_poses'] += 1
                else:
                    detected[truth] += 1
                    counts['poses_without_face'] += face is None
            counts['missing_poses'] += sum(not detected[p['truth']] for p in people)
            counts['duplicate_poses'] += sum(max(0,n-1) for n in detected.values())
    return counts
