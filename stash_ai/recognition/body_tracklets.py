"""Short, mutually separated SapiensID paths on retained VTT observations."""
from collections import defaultdict
import json
import uuid
import numpy as np

from .perspective_shots import change
from .track_geometry import rejection, torso_points, translation


def paths(rows, settings, scores):
    cues = defaultdict(list)
    geometry = {}
    for row in rows:
        q = json.loads(row['quality']).get('quality',{})
        person = json.loads(row['geometry'])
        cues[row['cue']].append(row)
        if q.get('frame_signature') and person.get('bounds') and q.get('view_shot') is not None:
            signature = np.frombuffer(bytes.fromhex(q['frame_signature']),np.uint8).astype(float).reshape(-1,3)
            geometry[row['id']] = q,person['bounds'],torso_points(person),signature
    threshold = settings.get('track_similarity_threshold')
    threshold = settings['cohesion_threshold'] if threshold is None else threshold
    margin = max(.025,settings['ambiguity_margin'])
    def complete(members):
        return 0<len(members)<=4 and all(r['id'] in geometry and
            not geometry[r['id']][0].get('query_only') and
            max(geometry[r['id']][0].get('detected_people',len(members)),
                geometry[r['id']][0].get('expected_people',0))==len(members) for r in members)
    links, incoming = {},set()
    for cue,left in sorted(cues.items()):
        right = cues.get(cue+1,())
        if not complete(left) or not complete(right):
            continue
        qa,_,_,sa = geometry[left[0]['id']]
        qb,_,_,sb = geometry[right[0]['id']]
        gap = right[0]['seconds']-left[0]['seconds']
        if qa['view_shot']!=qb['view_shot'] or not 0<gap<=12 or change(sa,sb,True)!='continuous':
            continue
        offset = translation(qa,qb,sa,sb)
        edges = {}
        for a in left:
            _,ba,ta,_ = geometry[a['id']]
            for b in right:
                _,bb,tb,_ = geometry[b['id']]
                if not rejection(ba,bb,ta,tb,gap,translation=offset):
                    edges[a['id'],b['id']] = scores.face_similarity(a['_vector'],b['_vector'])
        for (a,b),value in edges.items():
            rival = max((s for (i,j),s in edges.items() if (i==a)!=(j==b)),default=-1.)
            if value>=threshold and value-rival>=margin:
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


def plan(record, rows, seeds, decisions, settings, scores, guard, frozen):
    """Name paths only from strict roots. Anonymous paths stay scene-local."""
    from .body_identity import copresent, independent
    from .body_origins import physical_key
    roots = {r['id']:owner for owner,bank in seeds.items() for r in bank}
    anonymous = {}
    for chain,strengths in paths(rows,settings,scores):
        known = {roots[r['id']] for r in chain if r['id'] in roots}
        if len(known)>1 or any(r['id'] in decisions and
                               decisions[r['id']].get('person') not in known for r in chain):
            # A change of face owner is a physical boundary, not a bridge.
            for row in chain:
                if row['id'] not in decisions:
                    decisions[row['id']] = dict(person=None,status='withheld',role='auxiliary',
                        reason='different_face_root',track_blocker=dict(reason='different_face_root',cue=row['cue']))
            continue
        if not known:
            if len(independent(chain,settings))>=2:
                key = uuid.uuid5(uuid.NAMESPACE_URL,'anonymous-track:'+physical_key(record,chain[0]))
                for row in chain:
                    anonymous[row['id']] = str(key)
            continue
        owner = next(iter(known))
        anchors = [(i,r) for i,r in enumerate(chain) if r['id'] in roots]
        for i,row in enumerate(chain):
            if row['id'] in decisions or copresent([row],seeds[owner]) or not guard.allows(row,seeds[owner],owner):
                continue
            j,root = min(anchors,key=lambda v:(abs(i-v[0]),v[0]))
            if abs(i-j)>12 or abs(row['seconds']-root['seconds'])>60:
                continue
            floor = min(strengths[min(i,j):max(i,j)])
            ranked = frozen.rank(row,guard)
            rival = max((v[5] for v in ranked if v[2]!=owner),default=-1.)
            if floor-rival < max(.025,settings['ambiguity_margin']):
                continue
            decisions[row['id']] = dict(person=owner,status='accepted',role='auxiliary',
                reason='local_body_track',root=root['id'],track_floor=floor,runner=rival)
    return anonymous


def split_members(core, proposed, occupied, minimum):
    """Release a contiguous supported side only across a physical contradiction."""
    from .body_identity import independent
    destinations = {proposed[r['id']] for r in core if r['id'] in proposed}
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
