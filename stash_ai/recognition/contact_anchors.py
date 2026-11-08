"""Corroborate a crowded face/body link before it can train a person's bank."""
from collections import defaultdict
import json

from .track_geometry import rejection, torso_points


def crowded_link(q):
    return q.get('detected_people',1)>1 and q.get('overlap',0)>=.25


def questionable_links(rows,settings,scores):
    clean = defaultdict(list)
    crowded = []
    for row in rows:
        if not row.get('face_person'):
            continue
        quality = json.loads(row['quality']).get('quality',{})
        if crowded_link(quality):
            crowded.append((row,quality))
        elif row.get('face_anchor'):
            clean[row['cue']].append((row,quality))
    questionable = set()
    for row,q in crowded:
        data = json.loads(row['geometry'])
        points = torso_points(data)
        supported = False
        if len(points)>=3 and q.get('view_shot') is not None:
            for cue in (row['cue']-1,row['cue']+1):
                ranked = []
                for peer,other in clean[cue]:
                    gap = abs(peer['seconds']-row['seconds'])
                    if other.get('view_shot')!=q['view_shot'] or not 0<gap<=12:
                        continue
                    geom = json.loads(peer['geometry'])
                    if len(torso_points(geom))<3 or rejection(data['bounds'],geom['bounds'],points,torso_points(geom),gap):
                        continue
                    ranked.append((scores.face_similarity(row['_vector'],peer['_vector']),peer['face_person']))
                own = max((v for v,p in ranked if p==row['face_person']),default=-1.)
                rival = max((v for v,p in ranked if p!=row['face_person']),default=-1.)
                if own>=settings['cohesion_threshold'] and own-rival>=settings['ambiguity_margin']:
                    supported = True
                    break
        if not supported:
            questionable.add(row['id'])
    return questionable
