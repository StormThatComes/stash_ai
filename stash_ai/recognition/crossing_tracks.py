"""Bounded two-person assignment through one continuous, face-bracketed contact."""
from itertools import permutations


def crossings(by_cue, roots, decisions, transition, margin, *, occluded=()):
    claims = {}
    for start, first in by_cue.items():
        owners = tuple(roots.get(row['id']) for row in first)
        if len(owners) != 2 or None in owners or len(set(owners)) != 2:
            continue
        layers = [{0:tuple(first)}]
        hidden = None
        for cue in range(start + 1, start + 8):
            people = by_cue.get(cue, ())
            if cue in occluded and len(people)==1 and hidden is None:
                hidden = people[0]
                continue
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
                if (len(layers) > 2 or hidden is not None) and (people[0].get('seconds', 0)-first[0].get('seconds', 0) <= 60):
                    if _window(layers, owners, decisions, transition, margin, claims) and hidden is not None:
                        decisions[hidden['id']].update(person=None, status='withheld', role='auxiliary',
                            reason='unresolved_partner', track_blocker=dict(reason='unresolved_partner',cue=cue))
                break
    return claims


def _window(layers, owners, decisions, transition, margin, claims):
    # At most six interior pairs. Score each possible path with cached edges,
    # then charge the margin only on transitions where an alternative differs.
    paths = [(0., (0,))]
    physical_edges = []
    for earlier, later in zip(layers,layers[1:]):
        edges = {}
        physical = {}
        for left, previous in earlier.items():
            for right, current in later.items():
                scores = [transition(a,b) for a,b in zip(previous,current)]
                if None not in scores:
                    edges[left,right] = sum(scores)/2
                    physical[left,right] = frozenset((a['id'],b['id']) for a,b in zip(previous,current))
        physical_edges.append(physical)
        paths = [(score+edges[states[-1],state], (*states,state))
                 for score,states in paths for state in later if (states[-1],state) in edges]
        if not paths:
            return False
    score, best = max(paths)
    before, after = layers[0][0], layers[-1][best[-1]]
    for i in range(1,len(layers)-1):
        def separated(alternative):
            other_score, other = alternative
            if other[i]==best[i]:
                return True
            different = sum(edge[a,c]!=edge[b,d] for edge,a,b,c,d in
                            zip(physical_edges,best,other,best[1:],other[1:]))
            return score-other_score >= margin*different and score>other_score
        if not all(separated(path) for path in paths):
            continue
        for owner,row,first,last in zip(owners,layers[i][best[i]],before,after):
            if row.get('face_person') or decisions[row['id']].get('track_blocker',{}).get('reason')=='different_face_root':
                continue
            claims[row['id']] = owner,first,last,score/(len(layers)-1)
    return True
