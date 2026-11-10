"""Two-sided local turn evidence from a frozen, physically separated partner."""
from collections import defaultdict
import json

import numpy as np

from .body_identity import quality, independent
from .body_perspectives import orientation
from .turn_transitions import transitions


def recover_turns(rows, decisions, seeds, settings, scores, guard, retained, *, occupied_cues=(), cuts=()):
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
    successors = transitions(by_cue,geometry,metadata,signals,anchors,angles,settings,scores,occupied_cues,cuts)
    incoming = {edge[0] for edge in successors.values()}
    from .turn_witnesses import TurnWitnesses
    witnesses = TurnWitnesses(lookup,settings,scores)
    proposals, joins = {}, defaultdict(list)
    tails = {}
    certified = TurnWitnesses(lookup,settings,scores)
    for start in sorted(successors.keys()-incoming):
        lane = [start]
        while lane[-1] in successors:
            nxt = successors[lane[-1]][0]
            lane.append(nxt)
        # Visible ends must independently identify the same performer. Rear cores
        # only affiliate after two disjoint, bracketed episodes and a scene-wide veto.
        visible = [(i,anchors[k]) for i,k in enumerate(lane) if k in anchors
                   and angles[k] in ('front','side') and decisions[k]['reason']!='local_body_track']
        if visible:
            for end,indices in ((visible[0],range(visible[0][0])),(visible[-1],range(visible[-1][0]+1,len(lane)))):
                at,owner = end
                for i in indices:
                    if abs(i-at)<=16:
                        tails[lane[i]] = owner,lane[at]
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
                        edges = [successors[k][2:] for k in lane[previous:at]]
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
            certified.add(target,angles[oid],row,roots,metadata[oid].get('view_shot'))

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

                occupied[donor].discard(row['cue'])
                certified.add(target,angles[oid],row,roots,metadata[oid].get('view_shot'))

    # Query one frozen generation. Recovered tails never become witnesses.
    for row in rows:
        oid = row['id']
        shot = metadata[oid].get('view_shot')
        decision = decisions[oid]
        if (angles[oid]!='back' or shot is None or metadata[oid].get('query_only') or row.get('face_person')
                or decision.get('view_exclusion') or decision['reason'] in ('different_face_root','weak_face_track_conflict',
                    'unconfirmed_contact_face','unresolved_partner','bracketed_scene_core','bracketed_body_turn')):
            continue
        old = decision.get('person')
        candidates = set()
        if oid in tails:
            candidates.add(tails[oid][0])
        if old in targets and old not in retained and old not in seeds and len(targets[old])==1:
            candidates.update(targets[old])
        if len(candidates)!=1:
            continue
        if decision['status']=='accepted' and old not in targets:
            continue
        for target in sorted(candidates):
            if (row['cue'] in occupied[target] or not guard.allows(row,owners[target],target)
                    or decision['status']=='accepted' and (occupied[old] & occupied[target] or old in retained or old in seeds)):
                continue
            roots = (tails[oid][1],) if oid in tails else ()
            proof = certified.verify(target,'back',row,roots,outside_shot=shot,distinct_shots=True)
            if not proof:
                continue
            runner = max((scores.face_similarity(row['_vector'],ref['_vector'])
                for (other,angle),bank in certified.by_view.items() if angle=='back' and other!=target
                and row['cue'] not in occupied[other] for ref,_,_ in bank),default=-1.)
            if proof['view_score']-runner<margin:
                continue
            decisions[oid] = dict(person=target,status='accepted',role='auxiliary',reason='corroborated_rear_view',
                witness_ids=list(roots),runner=runner,required_margin=margin,**proof)
            occupied[target].add(row['cue'])
            break
