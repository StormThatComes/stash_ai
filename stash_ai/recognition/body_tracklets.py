"""Short, mutually separated SapiensID paths on retained VTT observations."""
from collections import defaultdict
import json
import uuid
import numpy as np

from .perspective_shots import change
from .track_geometry import rejection, torso_points, translation


def paths(rows, settings, scores, occupied=()):
    cues = defaultdict(list)
    geometry = {}
    physical = {}
    occupied = set(occupied)
    for row in rows:
        q = json.loads(row['quality']).get('quality',{})
        person = json.loads(row['geometry'])
        cues[row['cue']].append(row)
        physical[row['id']] = person
        if q.get('frame_signature') and person.get('bounds') and q.get('view_shot') is not None:
            signature = np.frombuffer(bytes.fromhex(q['frame_signature']),np.uint8).astype(float).reshape(-1,3)
            geometry[row['id']] = q,person['bounds'],torso_points(person),signature
    threshold = settings.get('track_similarity_threshold')
    threshold = settings['cohesion_threshold'] if threshold is None else threshold
    margin = max(.025,settings['ambiguity_margin'])
    def complete(members):
        return bool(members) and all(r['id'] in geometry and
            not geometry[r['id']][0].get('query_only') and
            max(geometry[r['id']][0].get('detected_people',len(members)),
                geometry[r['id']][0].get('expected_people',0))==len(members) for r in members)
    links, incoming = {},set()
    for cue,left in sorted(cues.items()):
        skipped = cue+1 not in cues and cue+1 not in occupied
        right = cues.get(cue+(2 if skipped else 1),())
        if not right:
            continue
        partial = not (complete(left) and complete(right))
        left_valid = [r for r in left if r['id'] in geometry]
        right_valid = [r for r in right if r['id'] in geometry]
        if not left_valid or not right_valid or skipped and partial:
            continue
        if partial and not any(len(side)>=2 and all(len(torso_points(physical[r['id']]))>=3 for r in side)
                               for side in (left,right)):
            continue
        qa,_,_,sa = geometry[left_valid[0]['id']]
        qb,_,_,sb = geometry[right_valid[0]['id']]
        continuous = np.abs(sa-sb).mean()<=8 if skipped else change(sa,sb,True)=='continuous'
        if qa['view_shot']!=qb['view_shot'] or not continuous:
            continue
        offset = None if skipped else translation(qa,qb,sa,sb)
        edges = {}
        for a in left_valid:
            _,ba,ta,_ = geometry[a['id']]
            for b in right_valid:
                _,bb,tb,_ = geometry[b['id']]
                if not rejection(ba,bb,ta,tb,translation=offset):
                    edges[a['id'],b['id']] = scores.face_similarity(a['_vector'],b['_vector'])
        if not partial:
            from .track_assignment import joint_links
            for a,b,value in joint_links(edges, physical, threshold, margin):
                links[a] = b,value
                incoming.add(b)
        if max(len(left),len(right)) > 4:
            continue
        for (a,b),value in edges.items():
            rival = max((s for (i,j),s in edges.items() if (i==a)!=(j==b)),default=-1.)
            if partial:
                from .crowd_context import pair
                # An incomplete rival must be physically separated at its own cue
                # and physically unable to take the other endpoint across the transition.
                def excluded(r, own, dest, reverse):
                    p = physical[r['id']]
                    if r['id'] in geometry:
                        return True
                    if not p.get('bounds') or pair([p,physical[own]]) is None:
                        return False
                    boxes = [p['bounds'],geometry[dest][1]]
                    torsos = [torso_points(p),geometry[dest][2]]
                    if reverse:
                        boxes.reverse()
                        torsos.reverse()
                    return bool(rejection(*boxes,*torsos,translation=offset))
                if not all(excluded(r,a,b,False) for r in left) or not all(excluded(r,b,a,True) for r in right):
                    continue
                left_q,_,left_torso,_ = geometry[a]
                right_q,_,right_torso,_ = geometry[b]
                if (left_q.get('query_only') or right_q.get('query_only') or
                        len(left_torso.keys() & right_torso.keys())<3 or rival!=-1. or
                        max(left_q.get('expected_people',0),right_q.get('expected_people',0),
                            left_q.get('detected_people',0),right_q.get('detected_people',0))>max(len(left),len(right))):
                    continue
            if a not in links and b not in incoming and value>=threshold and value-rival>=margin:
                links[a] = b,value
                incoming.add(b)
    lookup = {r['id']:r for r in rows}
    for start in sorted(set(links)-incoming):
        chain = [lookup[start]]
        strengths = []
        oid = start
        while oid in links:
            oid,strength = links[oid]
            chain.append(lookup[oid])
            strengths.append(strength)
        yield chain,strengths


def compatible_segments(chain, strengths, decisions, *, side=None):
    """Only the interval between contradictory measured owners is ambiguous."""
    start = 0
    previous = None
    for i,row in enumerate(chain):
        if row['id'] not in decisions:
            continue
        owner = decisions[row['id']].get('person')
        if owner is None:
            if i>start:
                yield chain[start:i],strengths[start:i-1]
            start,previous = i+1,None
        elif previous is not None and previous[1]!=owner:
            j = previous[0]
            left, right = j+1, i
            if side:
                while left < right and side(chain[left], chain[j], chain[i]):
                    left += 1
                while right > left and side(chain[right-1], chain[i], chain[j]):
                    right -= 1
            yield chain[start:left],strengths[start:left-1]
            for member in chain[left:right]:
                decisions[member['id']] = dict(person=None,status='withheld',role='auxiliary',
                    reason='different_face_root',track_blocker=dict(reason='different_face_root',cue=row['cue']))
            start,previous = right,(i,owner)
        else:
            previous = i,owner
    if start<len(chain):
        yield chain[start:],strengths[start:len(chain)-1]


def plan(record, rows, seeds, decisions, settings, scores, guard, frozen):
    """Name paths only from strict roots. Anonymous paths stay scene-local."""
    from .body_identity import copresent, independent
    from .body_origins import physical_key
    roots = {r['id']:owner for owner,bank in seeds.items() for r in bank}
    anonymous = {}
    split_sides = {}
    def side(row, own, other):
        if own['id'] not in roots or other['id'] not in roots:
            return False
        score = scores.face_similarity(row['_vector'], own['_vector'])
        rival = scores.face_similarity(row['_vector'], other['_vector'])
        if score >= settings['match_threshold'] and score-rival >= max(.025,settings['ambiguity_margin']):
            split_sides[row['id']] = (own['id'], other['id'])
            return True
        return False
    segments = (part for chain,strengths in paths(rows,settings,scores,record.get('occupied_cues',()))
                for part in compatible_segments(chain,strengths,decisions,side=side))
    for chain,strengths in segments:
        known = {roots[r['id']] for r in chain if r['id'] in roots}
        if not known:
            if len(independent(chain,settings))>=2:
                key = uuid.uuid5(uuid.NAMESPACE_URL,'anonymous-track:'+physical_key(record,chain[0]))
                for row in chain:
                    prior = decisions.get(row['id'])
                    if prior is None:
                        anonymous[row['id']] = str(key)
                    elif prior.get('person') is not None:
                        prior['anonymous_fragment'] = str(key)
            continue
        owner = next(iter(known))
        # Nearest roots on either side dominate more distant roots on that side.
        left_routes, right_routes = {}, {}
        for indices, routes, step in ((range(len(chain)),left_routes,1),
                                      (range(len(chain)-1,-1,-1),right_routes,-1)):
            root_id = None
            for i in indices:
                if chain[i]['id'] in roots:
                    root_id, position, floor = chain[i]['id'], i, 1.
                elif root_id is not None:
                    floor = min(floor, strengths[i-1 if step==1 else i])
                    routes[i] = (floor, -abs(i-position), root_id)
        for i,row in enumerate(chain):
            if row['id'] in decisions or copresent([row],seeds[owner]) or not guard.allows(row,seeds[owner],owner):
                continue
            bracketed = i in left_routes and i in right_routes
            routes = [bank[i] for bank in (left_routes,right_routes)
                      if i in bank and (bank[i][1]>=-12 or bracketed)]
            if not routes:
                continue
            floor, _, root_id = max(routes)
            ranked = frozen.rank(row,guard)
            rival, rival_owner = max(((v[5],v[2]) for v in ranked if v[2]!=owner),default=(-1.,None))
            if floor-rival < max(.025,settings['ambiguity_margin']):
                continue
            decisions[row['id']] = dict(person=owner,status='accepted',role='auxiliary',
                reason='local_body_track',root=root_id,track_floor=floor,runner=rival,rival=rival_owner,
                threshold=settings.get('track_similarity_threshold') if settings.get('track_similarity_threshold') is not None else settings['cohesion_threshold'],
                required_margin=max(.025,settings['ambiguity_margin']))
            if row['id'] in split_sides:
                decisions[row['id']]['conflict_endpoints'] = split_sides[row['id']]
    return anonymous


def claim_fragments(rows,anonymous,decisions,frozen,guard,settings):
    """Challenge an entire anonymous path against independent frozen face-seeded modes."""
    from .body_identity import independent
    if not anonymous or not frozen or not frozen.seeds:
        return
    components = defaultdict(list)
    for row in rows:
        key = anonymous.get(row['id']) or decisions.get(row['id'],{}).get('anonymous_fragment')
        if key:
            components[key].append(row)
    for fragment,members in components.items():
        queries = [r for r in members if r['id'] in anonymous]
        votes = independent(queries,settings)
        if len(votes)<2:
            continue
        known = {decisions[r['id']]['person'] for r in members if r['id'] in decisions}
        rankings = {r['id']:{v[2]:v for v in frozen.rank(r,guard)} for r in queries}
        common = set.intersection(*(set(bank) for bank in rankings.values()))
        if known:
            common &= known if len(known)==1 else set()
        options = []
        for owner in sorted(common):
            matches = [rankings[r['id']][owner] for r in queries]
            if any(v[1]<settings['cohesion_threshold'] or frozen.support[id(v[3])]<settings['attach_min_support']
                   for v in matches):
                continue
            def rival(oid):
                return max((v[5] for p,v in rankings[oid].items() if p!=owner),default=-1.)
            if any(rival(r['id'])>=v[0]+settings['ambiguity_margin'] for r,v in zip(queries,matches)):
                continue
            mean = sum(rankings[r['id']][owner][0] for r in votes)/len(votes)
            # A rival identity must explain the same physical votes. An
            # ineligible owner cannot explain that cue; keep the per-cue veto above.
            peers = set().union(*(rankings[r['id']] for r in votes)) - {owner}
            runner = max((sum(rankings[r['id']][p][5] if p in rankings[r['id']] else -1.
                              for r in votes)/len(votes) for p in peers), default=-1.)
            if mean>=settings['match_threshold'] and mean-runner>=settings['ambiguity_margin']:
                options.append((owner,mean,runner))
        if len(options)==1:
            owner,mean,runner = options[0]
            for row in queries:
                decisions[row['id']] = dict(person=owner,status='accepted',role='auxiliary',
                    reason='fragment_body_attachment',fragment=fragment,score=mean,runner=runner,
                    witness_ids=[r['id'] for r in rankings[row['id']][owner][3][:8]],
                    fragment_support=len(votes),threshold=settings['match_threshold'],
                    required_margin=settings['ambiguity_margin'])


def split_members(core, proposed, occupied, minimum):
    """Release a contiguous supported side only across a physical contradiction."""
    from .body_identity import independent
    destinations = {proposed[r['id']] for r in core if r['id'] in proposed}
    if len(destinations)==2:
        # Independently supported, disjoint prefix/suffix proposals. A measured
        # contradiction with each opposite owner guards the physical boundary.
        sides = sorted(([r for r in core if proposed.get(r['id'])==p] for p in destinations),
                       key=lambda rows:min(r['cue'] for r in rows))
        left,right = sides
        if (len(independent(left,minimum))<max(2,minimum['attach_min_support']) or
                len(independent(right,minimum))<max(2,minimum['attach_min_support'])):
            return set()
        end = max(r['cue'] for r in left)
        start = min(r['cue'] for r in right)
        if (end>=start or any(r['id'] not in proposed and not end<r['cue']<start for r in core)
                or not any(occupied(r,proposed[right[0]['id']]) for r in left)
                or not any(occupied(r,proposed[left[0]['id']]) for r in right)):
            return set()
        return {r['id'] for side in sides for r in side}
    if len(destinations)!=1:
        return set()
    target = destinations.pop()
    yes = [r for r in core if proposed.get(r['id'])==target]
    no = [r for r in core if proposed.get(r['id'])!=target]
    if not yes or not no or len(independent(yes,minimum))<max(2,minimum['attach_min_support']):
        return set()
    before = max(r['cue'] for r in yes)<min(r['cue'] for r in no)
    after = max(r['cue'] for r in no)<min(r['cue'] for r in yes)
    boundary = (min if before else max)(no,key=lambda r:r['cue'])
    if not (before or after) or not occupied(boundary,target):
        return set()
    return {r['id'] for r in yes}
