"""Measure bounded two-person paths for solver proposals on retained goal cues."""
import json
import numpy as np
from .body_identity import vector
from .body_observations import contract


def path_interval(layers):
    """Best bottleneck among alternative paths that leave and return to the labelled order."""
    own = 1.
    states = {(0,False):1.}
    for table in layers:
        own = min(own,table[0][0])
        following = {}
        for (before,swapped),strength in states.items():
            for after in (0,1):
                key = after,swapped or bool(after)
                following[key] = max(following.get(key,-1.),min(strength,table[before][after]))
        states = following
    return own, states.get((0,True),-1.)


def labelled_path_wins(pairs,cutoff,margin):
    """Challenge the labelled order with the same physical path scorer as replay."""
    from .crossing_tracks import _path
    layers = [[0]]+[[0,1] for _ in range(len(pairs)-1)]+[[0]]
    edges = []
    for i,measured in enumerate(pairs):
        table = {}
        for before in layers[i]:
            for after in layers[i+1]:
                links = ((before,after),(1-before,1-after))
                values = [measured[a,b] for a,b in links]
                if None in values:
                    continue
                table[before,after] = (sum(values)/2, min(values)>=cutoff, frozenset(links))
        edges.append(table)
    winner = _path(layers,edges)
    if winner is None or any(winner[1]):
        return False
    for position in range(1,len(layers)-1):
        rival = _path(layers,edges,rival_at=position,best=winner[1],margin=margin)
        if rival and (rival[0]>winner[0]+1e-12 or margin==0 and rival[0]>=winner[0]):
            return False
    return True


def contact_interval(db,config,owners,oid,peer,*,measured=None):
    source = db.execute('SELECT scan,cue FROM body_observations WHERE id=?',(oid,)).fetchone()
    if not source:
        return None
    scan,center = source
    by_cue = {}
    for row in db.execute('SELECT id,cue,seconds,face,embedding,quality,geometry,rejected FROM body_observations '
            'WHERE scan=? AND contract=? AND cue BETWEEN ? AND ? ORDER BY cue,id',
            (scan,contract(config),center-5,center+5)):
        by_cue.setdefault(row['cue'],[]).append(row)
    layers = {}
    index = owners[oid]
    for cue,rows in by_cue.items():
        if len(rows)!=2 or any(r['embedding'] is None or r['rejected'] for r in rows):
            continue
        tagged = [(owners.get(r['id'],owners.get(r['face'])),r) for r in rows]
        participants = {k for k,r in tagged}
        if None in participants or index not in participants or len(participants)!=2:
            continue
        tagged.sort(key=lambda x:x[0]!=index)
        qualities = [json.loads(r['quality']).get('quality',{}) for k,r in tagged]
        shots = {q.get('view_shot') for q in qualities}
        if len(shots)!=1 or None in shots or any(max(q.get('expected_people',2), q.get('detected_people',2))>2
                or not q.get('frame_signature') for q in qualities):
            continue
        layers[cue] = (tuple(k for k,r in tagged), shots.pop(), tagged[0][1]['seconds'],
                       [vector(r) for k,r in tagged], [r['id'] for k,r in tagged], [json.loads(r['geometry']) for k,r in tagged],
                       np.frombuffer(bytes.fromhex(qualities[0]['frame_signature']), np.uint8).astype(float).reshape(-1,3))
    if center not in layers:
        return None
    start=end=center
    from .perspective_shots import change
    from .track_geometry import rejection, torso_points
    def adjacent(i,j):
        return (i in layers and j in layers and layers[i][:2]==layers[j][:2] and
                0<layers[j][2]-layers[i][2]<=12 and change(layers[i][6],layers[j][6],True)=='continuous')
    while adjacent(start-1,start):
        start -= 1
    while adjacent(end,end+1):
        end += 1
    if end-start<2 or peer not in [layers[k][4][0] for k in range(start,end+1)]:
        return None
    edges, physical = [],[]
    for cue in range(start,end):
        a,b = layers[cue][3],layers[cue+1][3]
        def score(i,j):
            left,right = layers[cue][5][i],layers[cue+1][5][j]
            if not left.get('bounds') or not right.get('bounds') or rejection(
                    left['bounds'],right['bounds'],torso_points(left),torso_points(right),
                    layers[cue+1][2]-layers[cue][2]):
                return None
            return float(np.dot(a[i],b[j]))
        values = {(i,j):score(i,j) for i in (0,1) for j in (0,1)}
        physical.append(values)
        def bottleneck(values):
            return -1. if None in values else min(values)
        direct = bottleneck([values[i,i] for i in (0,1)])
        crossed = bottleneck([values[i,1-i] for i in (0,1)])
        edges.append(((direct,crossed),(crossed,direct)))
    own, rival = path_interval(edges)
    from .merge_report_policy import _floor,parameter
    key = 'sapiensid.identities.track_similarity_threshold'
    limit = min(own,measured) if measured is not None else own
    cutoff = min(parameter(config.data,key),_floor(max(0.,limit)))
    margin = max(.025,parameter(config.data,'sapiensid.identities.ambiguity_margin'))
    return dict(first_cue=start,last_cue=end,own_floor=own,rival_floor=rival,
                separated=own>=0 and labelled_path_wins(physical,cutoff,margin))
