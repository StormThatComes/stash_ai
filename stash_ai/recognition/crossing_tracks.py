"""Resolve a bounded two-person contact using strict endpoints and physical edges."""
from itertools import permutations


def _path(layers, edges, *, rival_at=None, best=None, margin=0.):
    # At each layer keep the best prefix ending in each of the two permutations.
    paths = {i:(0.,(i,)) for i in layers[0]}
    for index, table in enumerate(edges,1):
        current = {}
        for (left,right),(score,eligible,physical) in table.items():
            if left not in paths or (best is None and not eligible):
                continue
            if rival_at==index and best[index]==right:
                continue
            if best is not None and physical!=table[best[index-1],best[index]][2]:
                score += margin
            value, prefix = paths[left]
            candidate = value+score, (*prefix,right)
            if right not in current or candidate>current[right]:
                current[right] = candidate
        paths = current
    return max(paths.values(),default=None)


def _window(pairs, owners, roots, decisions, transition, raw, margin, claims, *, complementary=False):
    layers = []
    terminal = set()
    for position,pair in enumerate(pairs):
        states = {}
        for i,order in enumerate(permutations(pair)):
            known = [roots.get(r['id']) or (decisions[r['id']].get('person') if r.get('face_person') else None) for r in order]
            agrees = all(p in (None,owner) for p,owner in zip(known,owners))
            last = position == len(pairs)-1
            if agrees or last and complementary:
                states[i] = order
            if last and agrees:
                terminal.add(i)
        if not states:
            return False
        layers.append(states)
    edges = []
    for left,right in zip(layers,layers[1:]):
        table = {}
        cache = {}
        for i,previous in left.items():
            for j,current in right.items():
                physical = []
                values = []
                for a,b in zip(previous,current):
                    key = a['id'],b['id']
                    if key not in cache:
                        score = transition(a,b)
                        cache[key] = score if score is not None or raw is None else raw(a,b), score is not None
                    physical.append(key)
                    values.append(cache[key])
                if all(score is not None for score,_ in values):
                    table[i,j] = sum(s for s,_ in values)/2, all(ok for _,ok in values), frozenset(physical)
        edges.append(table)
    admissible = [*edges[:-1],{key:v for key,v in edges[-1].items() if key[1] in terminal}]
    winner = _path(layers,admissible)
    if winner is None:
        return False
    score,best = winner
    if complementary:
        # Challenge the implied association with the opposite physical ending.
        rival = _path(layers,edges,rival_at=len(layers)-1,best=best,margin=margin)
        if rival and rival[0]>=score-1e-12:
            return False
    before,after = layers[0][best[0]],layers[-1][best[-1]]
    for i in range(1,len(layers)-1):
        rival = _path(layers,admissible,rival_at=i,best=best,margin=margin)
        if rival and (rival[0]>score+1e-12 or margin==0 and rival[0]>=score):
            continue
        for owner,row,first,last in zip(owners,layers[i][best[i]],before,after):
            if row.get('face_person') or decisions[row['id']].get('track_blocker',{}).get('reason')=='different_face_root':
                continue
            claims.setdefault(row['id'],{})[owner] = owner,first,last,score/len(edges)
    return before,after


def crossings(by_cue, roots, decisions, transition, margin, *, occluded=(), raw=None, visible_single=None):
    claims = {}
    for start,first in sorted(by_cue.items()):
        if len(first)!=2 or not any(r['id'] in roots for r in first):
            continue
        pairs = [first]
        hidden = []
        for cue in range(start+1,start+13):
            people = by_cue.get(cue,())
            if len(people)==1 and cue in occluded and len(hidden)<2:
                hidden.append(people[0])
                continue
            if len(people)!=2:
                break
            pairs.append(people)
            final_roots = [roots.get(r['id']) for r in people]
            first_roots = [roots.get(r['id']) for r in first]
            if not any(final_roots):
                continue
            complete_end = all(final_roots)
            owners = sorted(set(first_roots+final_roots)-{None})
            complementary = not complete_end and not all(first_roots)
            if len(owners)==2 and (len(pairs)>2 or hidden):
                ends = _window(pairs,owners,roots,decisions,transition,raw,margin,claims,
                               complementary=complementary)
                if ends:
                    before,after = ends
                    for row in hidden:
                        decision = decisions[row['id']]
                        if row.get('face_person') or decision.get('track_blocker',{}).get('reason')=='different_face_root':
                            continue
                        choices = []
                        if visible_single and visible_single(row):
                            for owner,left,right in zip(owners,before,after):
                                edges = [transition(left,row),transition(row,right)]
                                if all(value is not None for value in edges):
                                    choices.append((min(edges),owner,left,right))
                        if choices and raw is not None:
                            score,owner,left,right = max(choices,key=lambda choice:choice[0])
                            rivals = []
                            for person,a,b in zip(owners,before,after):
                                if person!=owner:
                                    values = raw(a,row),raw(row,b)
                                    if None not in values:
                                        rivals.append(min(values))
                            if not rivals or score-max(rivals)>=margin:
                                claims.setdefault(row['id'],{})[owner] = owner,left,right,score
                                continue
                        decision.update(person=None,status='withheld',role='auxiliary',reason='unresolved_partner',
                                        track_blocker=dict(reason='unresolved_partner',cue=row.get('cue',cue)))
            if complete_end:
                break
    return {oid:next(iter(options.values())) for oid,options in claims.items() if len(options)==1}
