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
                    table[i,j] = sum(s for s,_ in values)/len(values), all(ok for _,ok in values), frozenset(physical)
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
    certified = {0:before,len(layers)-1:after}
    for i in range(1,len(layers)-1):
        rival = _path(layers,admissible,rival_at=i,best=best,margin=margin)
        if rival and (rival[0]>score+1e-12 or margin==0 and rival[0]>=score):
            continue
        certified[i] = layers[i][best[i]]
        for owner,row,first,last in zip(owners,layers[i][best[i]],before,after):
            if row.get('face_person') or decisions[row['id']].get('track_blocker',{}).get('reason')=='different_face_root':
                continue
            claims.setdefault(row['id'],{})[owner] = owner,first,last,score/len(edges)
    return certified


def crossings(by_cue, roots, decisions, transition, margin, *, occluded=(), raw=None, visible_single=None, max_cues=12, max_hidden=2, max_people=4):
    claims = {}
    for start,first in sorted(by_cue.items()):
        if not 2<=len(first)<=max_people or not any(r['id'] in roots for r in first):
            continue
        pairs = [first]
        hidden = []
        for cue in range(start+1,start+max_cues+1):
            people = by_cue.get(cue,())
            if 0<len(people)<len(first) and cue in occluded and len(hidden)<max_hidden:
                hidden.append((len(pairs)-1,people))
                continue
            if len(people)!=len(first):
                break
            pairs.append(people)
            final_roots = [roots.get(r['id']) for r in people]
            first_roots = [roots.get(r['id']) for r in first]
            if not any(final_roots):
                continue
            complete_end = all(final_roots)
            owners = sorted(set(first_roots+final_roots)-{None})
            complementary = not complete_end and not all(first_roots)
            if len(owners)==len(first) and (len(pairs)>2 or hidden):
                ends = _window(pairs,owners,roots,decisions,transition,raw,margin,claims,
                               complementary=complementary)
                if ends:
                    for position,members in hidden:
                        before = ends[max(i for i in ends if i<=position)]
                        after = ends[min(i for i in ends if i>position)]
                        _partial_cue(members,owners,before,after,ends[0],ends[len(pairs)-1],
                                     decisions,transition,raw,visible_single,margin,claims)
            if complete_end:
                break
    return {oid:next(iter(options.values())) for oid,options in claims.items() if len(options)==1}


def _partial_cue(members,owners,before,after,first,last,decisions,transition,raw,visible,margin,claims):
    """Visible lanes compete jointly; an absent lane never supplies a witness."""
    from .joint_claims import solve
    fixed = {decisions[r['id']].get('person') for r in members if r.get('face_person')}
    # A measured entrant starts another episode, rather than taking an absent lane.
    if fixed - {None,*owners}:
        return
    queries = [r for r in members if not r.get('face_person') and
               decisions[r['id']].get('track_blocker',{}).get('reason')!='different_face_root']
    domains=[]
    for row in queries:
        options=[]
        if raw is not None and visible and visible(row):
            for i,(owner,a,b) in enumerate(zip(owners,before,after)):
                values=transition(a,row),transition(row,b)
                if owner in fixed or None in values:
                    continue
                score=min(values)
                rivals=[min(v) for j,(x,y) in enumerate(zip(before,after))
                        if j!=i and None not in (v:=(raw(x,row),raw(row,y)))]
                if not rivals or score-max(rivals)>=margin:
                    options.append((owner,score,i))
        domains.append(options)
    solution=solve(domains,lambda i,a,j,b:a[0]!=b[0],max(margin,1e-12)*len(domains))
    for row in queries:
        decisions[row['id']].update(person=None,status='withheld',role='auxiliary',reason='unresolved_partner',
            track_blocker=dict(reason='unresolved_partner',cue=row.get('cue')))
    if solution:
        for row,(owner,score,index) in zip(queries,solution):
            claims.setdefault(row['id'],{})[owner]=owner,first[index],last[index],score
