"""Two-sided local turn evidence from a frozen, physically separated partner."""
from collections import defaultdict
import json

import numpy as np

from .body_identity import quality, independent
from .body_perspectives import orientation
from .crowd_context import pair
from .perspective_shots import change
from .track_geometry import rejection, torso_points


def recover_turns(rows, decisions, seeds, settings, scores, guard, retained):
    strict = settings.get('track_similarity_threshold')
    strict = settings['cohesion_threshold'] if strict is None else strict
    margin = max(.025, settings['ambiguity_margin'])
    allowed = {'body_scene_core','measured_face_association','local_body_track'}
    by_cue, owners, anchors = defaultdict(list), defaultdict(list), {}
    geometry, metadata, angles, signals = {}, {}, {}, {}
    for row in rows:
        oid = row['id']
        decision = decisions[oid]
        q = metadata[oid] = quality(row)
        g = geometry[oid] = json.loads(row['geometry'])
        angles[oid] = orientation(g) if g.get('keypoints') else q.get('orientation')
        by_cue[row['cue']].append(row)
        if decision['status']=='accepted':
            owner = decision['person']
            owners[owner].append(row)
            if decision['reason'] in allowed and not q.get('query_only'):
                anchors[oid] = owner
        if q.get('frame_signature'):
            signals[oid] = np.frombuffer(bytes.fromhex(q['frame_signature']),np.uint8).astype(float).reshape(-1,3)
    lookup = {r['id']:r for r in rows}
    # Build independent lanes. Every step needs an already recognised partner
    # in both cues, uniquely separated torsos, an unchanged shot and a rival margin.
    successors, incoming = {}, set()
    measurements = {}
    for cue,left in by_cue.items():
        right = by_cue.get(cue+1,())
        if not all(pair([geometry[r['id']] for r in side]) for side in (left,right)):
            continue
        if any(metadata[r['id']].get('query_only') or r['id'] not in signals or
               max(metadata[r['id']].get('detected_people',2),metadata[r['id']].get('expected_people',0))!=2
               for r in (*left,*right)):
            continue
        for i,a in enumerate(left):
            partner = anchors.get(left[1-i]['id'])
            matches = [j for j,b in enumerate(right) if partner and anchors.get(right[1-j]['id'])==partner]
            if len(matches)!=1:
                continue
            j = matches[0]
            b, rival = right[j], right[1-j]
            x,y = a['id'],b['id']
            if (anchors.get(x)==partner or anchors.get(y)==partner or y in incoming or
                    metadata[x].get('view_shot') is None or metadata[x]['view_shot']!=metadata[y].get('view_shot') or
                    change(signals[x],signals[y],True)!='continuous'):
                continue
            ga,gb = geometry[x],geometry[y]
            if rejection(ga['bounds'],gb['bounds'],torso_points(ga),torso_points(gb)):
                continue
            score = scores.face_similarity(a['_vector'],b['_vector'])
            alternate = max(scores.face_similarity(a['_vector'],rival['_vector']),
                            scores.face_similarity(left[1-i]['_vector'],b['_vector']))
            turn = angles[x]!=angles[y] and {angles[x],angles[y]} <= {'front','side','back'}
            floor = max(0.,strict-(settings.get('turn_similarity_discount',.06) if turn else 0.))
            if score>=floor and score-alternate>=margin:
                measurements[x] = score,alternate,floor
                successors[x] = y,partner
                incoming.add(y)

    from .turn_witnesses import TurnWitnesses
    witnesses = TurnWitnesses(lookup,settings,scores)
    proposals, joins = {}, defaultdict(list)
    for start in sorted(successors.keys()-incoming):
        lane = [start]
        while lane[-1] in successors:
            nxt,partner = successors[lane[-1]]
            if len(lane)>1 and partner!=successors[lane[-2]][1]:
                break
            lane.append(nxt)
        # Visible ends must independently identify the same performer. Rear cores
        # only affiliate after two disjoint, bracketed episodes and a scene-wide veto.
        previous = None
        for at,oid in enumerate(lane):
            decision = decisions[oid]
            if angles[oid] not in ('front','side') or oid not in anchors or decision['reason']=='local_body_track':
                continue
            if previous is not None:
                first = lane[previous]
                middle = lane[previous+1:at]
                target = anchors[oid]
                if (middle and at-previous<=16 and anchors[first]==target and
                        len(independent([lookup[first],lookup[oid]],settings))==2 and
                        all(not lookup[key].get('face_person') and not decisions[key].get('view_exclusion')
                            and decisions[key]['reason'] not in ('different_face_root','weak_face_track_conflict',
                                                                  'unconfirmed_contact_face','unresolved_partner')
                            for key in middle)):
                    foreign = {decisions[key]['person'] for key in middle
                               if decisions[key]['status']=='accepted'}-{target}
                    if not foreign or len(foreign)==1 and all(angles[key]=='back' for key in middle):
                        roots = first,oid
                        edges = [measurements[k] for k in lane[previous:at]]
                        weakest = min(edges,key=lambda edge:edge[0]-edge[2])
                        proof = dict(track_floor=weakest[0],threshold=weakest[2],runner=weakest[1],
                                     required_margin=margin,turn_discount=settings.get('turn_similarity_discount',.06))
                        for key in middle:
                            if angles[key] not in ('front','back','side'):
                                continue
                            # Pick one middle observation per episode as a potential witness.
                            witnesses.add(target,angles[key],lookup[key],roots,metadata[key].get('view_shot'))
                            if not foreign and decisions[key]['status']!='accepted':
                                proposals[key] = target,roots,proof
                            elif decisions[key]['person'] in foreign:
                                joins[decisions[key]['person'],target].append((key,roots,proof))
            previous = at

    occupied = {owner:{r['cue'] for r in members} for owner,members in owners.items()}
    for oid,(target,roots,proof) in proposals.items():
        row = lookup[oid]
        if row['cue'] in occupied[target] or not guard.allows(row,owners[target],target):
            continue
        corroboration = witnesses.verify(target,angles[oid],row,roots)
        if corroboration:
            decisions[oid] = dict(person=target,status='accepted',role='auxiliary',reason='bracketed_body_turn',
                                  witness_ids=list(roots),**proof,**corroboration)
            occupied[target].add(row['cue'])

    targets = defaultdict(set)
    for donor,target in joins:
        targets[donor].add(target)
    for (donor,target),members in sorted(joins.items()):
        if (len(targets[donor])!=1 or donor in seeds or donor in retained or target in targets or
                occupied[donor] & occupied[target] or any(r.get('face_person') for r in owners[donor])):
            continue
        # Transfer only observations supported by an actual turn episode.
        for oid,roots,proof in members:
            row = lookup[oid]
            if row['cue'] in occupied[target] or not guard.allows(row,owners[target],target):
                continue
            corroboration = witnesses.verify(target,angles[oid],row,roots,confirmed_core=True)
            if corroboration:
                decisions[oid].update(person=target,role='auxiliary',reason='bracketed_scene_core',
                                      original_core=donor,witness_ids=list(roots),**proof,**corroboration)
                occupied[target].add(row['cue'])
