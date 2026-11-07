"""Bounded two-person assignment through one continuous, face-bracketed contact."""
from itertools import permutations
import math


def crossings(by_cue, roots, decisions, transition, margin):
    claims = {}
    for start, first in by_cue.items():
        owners = tuple(roots.get(row['id']) for row in first)
        if len(owners) != 2 or None in owners or len(set(owners)) != 2:
            continue
        layers = [{0:tuple(first)}]
        for cue in range(start + 1, start + 8):
            people = by_cue.get(cue, ())
            if len(people) != 2:
                break
            states = {}
            for index, order in enumerate(permutations(people)):
                known = [roots.get(row['id']) or (decisions[row['id']].get('person') if row.get('face_person') else None)
                         for row in order]
                if all(person in (None,owner) for person,owner in zip(known,owners)):
                    states[index] = order
            if not states:
                break
            layers.append(states)
            if all(row['id'] in roots for row in people):
                if len(layers) > 2 and (people[0].get('seconds', 0)-first[0].get('seconds', 0) <= 60):
                    _window(layers, owners, decisions, transition, margin, claims)
                break
    return claims


def _window(layers, owners, decisions, transition, margin, claims):
    # Forward and backward maximum scores constrained to each possible state.
    # Keeping the alternate state per cue allows abstention on just the tied cues.
    edges = []
    forward = [{0:0.}]
    for earlier, later in zip(layers, layers[1:]):
        weights = {}
        for left, previous in earlier.items():
            for right, current in later.items():
                values = [transition(a,b) for a,b in zip(previous,current)]
                if None not in values:
                    weights[left,right] = sum(values)/2
        edges.append(weights)
        forward.append({right:max((forward[-1][left]+value for (left,r),value in weights.items()
                                   if r==right),default=-math.inf) for right in later})
    backward = dict.fromkeys(layers[-1],0.)
    steps = len(edges)
    last = next(iter(layers[-1].values()))
    for i in range(steps-1,0,-1):
        backward = {left:max((value+backward[right] for (l,right),value in edges[i].items()
                              if l==left),default=-math.inf) for left in layers[i]}
        ranked = sorted(((forward[i][state]+backward[state],state) for state in layers[i]),reverse=True)
        score,state = ranked[0]
        if not math.isfinite(score) or len(ranked)>1 and (score-ranked[1][0])/steps < margin:
            continue
        for owner, row, before, after in zip(owners,layers[i][state],layers[0][0],last):
            if row.get('face_person') or decisions[row['id']].get('track_blocker',{}).get('reason')=='different_face_root':
                continue
            claims[row['id']] = owner,before,after,score/steps
