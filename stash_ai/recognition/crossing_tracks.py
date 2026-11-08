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


def _window(pairs, owners, roots, decisions, transition, raw, margin, claims):
    layers = []
    for pair in pairs:
        states = {}
        for i,order in enumerate(permutations(pair)):
            known = [roots.get(r['id']) or (decisions[r['id']].get('person') if r.get('face_person') else None) for r in order]
            if all(p in (None,owner) for p,owner in zip(known,owners)):
                states[i] = order
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
    winner = _path(layers,edges)
    if winner is None:
        return False
    score,best = winner
    before,after = layers[0][best[0]],layers[-1][best[-1]]
    for i in range(1,len(layers)-1):
        rival = _path(layers,edges,rival_at=i,best=best,margin=margin)
        if rival and (rival[0]>score+1e-12 or margin==0 and rival[0]>=score):
            continue
        for owner,row,first,last in zip(owners,layers[i][best[i]],before,after):
            if row.get('face_person') or decisions[row['id']].get('track_blocker',{}).get('reason')=='different_face_root':
                continue
            claims.setdefault(row['id'],{})[owner] = owner,first,last,score/len(edges)
    return True


def crossings(by_cue, roots, decisions, transition, margin, *, occluded=(), raw=None):
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
            if len(people)!=2 or people[0].get('seconds',0)-first[0].get('seconds',0)>60:
                break
            pairs.append(people)
            final_roots = [roots.get(r['id']) for r in people]
            first_roots = [roots.get(r['id']) for r in first]
            if not any(final_roots):
                continue
            complete_end = all(final_roots)
            owners = final_roots if complete_end else first_roots
            if all(owners) and len(set(owners))==2 and (len(pairs)>2 or hidden):
                if _window(pairs,owners,roots,decisions,transition,raw,margin,claims):
                    for row in hidden:
                        decision = decisions[row['id']]
                        if decision.get('track_blocker',{}).get('reason')!='different_face_root':
                            decision.update(person=None,status='withheld',role='auxiliary',reason='unresolved_partner',
                                            track_blocker=dict(reason='unresolved_partner',cue=row.get('cue',cue)))
            if complete_end:
                break
    return {oid:next(iter(options.values())) for oid,options in claims.items() if len(options)==1}
