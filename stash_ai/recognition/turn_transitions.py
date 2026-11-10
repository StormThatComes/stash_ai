"""Scene-local, visually verified transitions between sampled cues."""
from bisect import bisect_right
from itertools import pairwise
import numpy as np

from .crowd_context import pair
from .perspective_shots import change
from .track_geometry import rejection, torso_points, translation


def transitions(by_cue, geometry, metadata, signals, anchors, angles, settings, scores, occupied, cuts):
    strict = settings.get('track_similarity_threshold')
    strict = settings['cohesion_threshold'] if strict is None else strict
    margin = max(.025,settings['ambiguity_margin'])
    occupied,cuts = sorted(occupied),sorted(cuts)
    links = {}
    for cue,target in pairwise(sorted(by_cue)):
        left,right = by_cue[cue],by_cue[target]
        gap = target-cue
        if (gap>4 or bisect_right(occupied,cue)!=bisect_right(occupied,target-1)
                or bisect_right(cuts,cue)!=bisect_right(cuts,target)):
            continue
        if any(len(side)>8 or any(
                r['id'] not in signals or not geometry[r['id']].get('bounds')
                or max(metadata[r['id']].get('detected_people',len(side)),metadata[r['id']].get('expected_people',0))!=len(side)
                for r in side) for side in (left,right)):
            continue
        first,last = left[0]['id'],right[0]['id']
        qa,qb = metadata[first],metadata[last]
        sa,sb = signals[first],signals[last]
        if qa.get('view_shot') is None or qa['view_shot']!=qb.get('view_shot') or change(sa,sb,True)!='continuous':
            continue
        if gap>1 and (np.abs(sa-sb).mean()>8 or len(left)!=len(right)):
            continue
        offset = translation(qa,qb,sa,sb)
        def separated(member,side):
            return all(peer is member or pair([geometry[member['id']],geometry[peer['id']]]) for peer in side)
        edges = {}
        for a in left:
            for b in right:
                x,y = a['id'],b['id']
                ga,gb = geometry[x],geometry[y]
                if not rejection(ga['bounds'],gb['bounds'],torso_points(ga),torso_points(gb),translation=offset):
                    edges[x,y] = scores.face_similarity(a['_vector'],b['_vector'])
        for a in left:
            x = a['id']
            if metadata[x].get('query_only') or not separated(a,left):
                continue
            for b in right:
                y = b['id']
                score = edges.get((x,y))
                if score is None or metadata[y].get('query_only') or not separated(b,right):
                    continue
                other_left = {anchors[r['id']]:r for r in left if r is not a and r['id'] in anchors}
                other_right = {anchors[r['id']]:r for r in right if r is not b and r['id'] in anchors}
                partners = sorted(p for p in other_left.keys() & other_right.keys()
                                  if sum(anchors.get(r['id'])==p for r in left)==1 and
                                  sum(anchors.get(r['id'])==p for r in right)==1)
                if any(anchors.get(k) in partners for k in (x,y)):
                    continue
                if len(left)>1 or len(right)>1:
                    if not partners:
                        continue
                    if gap>1 and not any(edges.get((other_left[p]['id'],other_right[p]['id']),-1.)>=strict for p in partners):
                        continue
                # Exclude only a strongly rooted partner with a feasible continuation
                # and reliable, separated torsos. Unmeasured entrants remain rivals.
                proven = {p for p in partners if edges.get((other_left[p]['id'],other_right[p]['id']),-1.)>=strict
                          and all(len(torso_points(geometry[k]))>=3 and not metadata[k].get('query_only')
                                  for k in (x,y,other_left[p]['id'],other_right[p]['id']))
                          and not any(metadata[k].get('overlap',0) for k in
                                      (x,y,other_left[p]['id'],other_right[p]['id']))}

                rival,peer = max([(scores.face_similarity(a['_vector'],r['_vector']),r['id']) for r in right if r is not b
                                     and ((x,r['id']) in edges or anchors.get(r['id']) not in proven)]+
                                 [(scores.face_similarity(l['_vector'],b['_vector']),l['id']) for l in left if l is not a
                                  and ((l['id'],y) in edges or anchors.get(l['id']) not in proven)],
                                 default=(-1.,None))
                turn = angles[x]!=angles[y] and {angles[x],angles[y]} <= {'front','side','back'}
                floor = max(0.,strict-(settings.get('turn_similarity_discount',.06) if turn else 0.))
                if score>=floor and score-rival>=margin:
                    links[x] = (y,tuple(partners),score,rival,floor,peer,turn)
    return links
