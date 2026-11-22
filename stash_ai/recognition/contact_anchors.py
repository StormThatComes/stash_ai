"""Corroborate a crowded face/body link before it can train a person's bank."""
from .body_numeric import body_similarity
from collections import defaultdict
import json

from .track_geometry import rejection, torso_points, translation
import numpy as np


def crowded_link(q):
    return q.get('detected_people',1)>1 and max(q.get('overlap',0), q.get('box_overlap',0))>=.25


def questionable_links(rows,settings,scores,graph=(), *, evidence=None):
    from .contact_face_evidence import retained
    evidence = {} if evidence is None else evidence
    clean = defaultdict(list)
    crowded = []
    for row in rows:
        if not row.get('face_person'):
            continue
        quality = json.loads(row['quality']).get('quality',{})
        if crowded_link(quality):
            crowded.append((row,quality))
        elif row.get('face_anchor'):
            clean[row['cue']].append((row,quality))
    questionable = set()
    contradicted = set()
    for row,q in crowded:
        data = json.loads(row['geometry'])
        points = torso_points(data)
        supported = False
        conflicted = False
        local = retained(row, q)
        temporal = dict(reason='insufficient_torso_points' if len(points)<3 else
                        'shot_unavailable' if q.get('view_shot') is None else 'no_clean_neighbour',
                        threshold=settings['cohesion_threshold'], margin=settings['ambiguity_margin'], neighbours=[])
        check = evidence[row['id']] = dict(status='withheld',
            crowding=dict(detected_people=q.get('detected_people'), overlap=q.get('overlap',0),
                          box_overlap=q.get('box_overlap',0), threshold=.25),
            same_frame=local, temporal=temporal)
        if len(points)>=3 and q.get('view_shot') is not None:
            for cue in (row['cue']-1,row['cue']+1):
                ranked = []
                neighbour = dict(cue=cue, candidates=len(clean[cue]), rejected={})
                temporal['neighbours'].append(neighbour)
                def reject(reason):
                    neighbour['rejected'][reason] = neighbour['rejected'].get(reason,0)+1
                for peer,other in clean[cue]:
                    if other.get('view_shot')!=q['view_shot']:
                        reject('different_shot')
                        continue
                    geom = json.loads(peer['geometry'])
                    offset = None
                    if q.get('frame_signature') and other.get('frame_signature'):
                        signatures = [np.frombuffer(bytes.fromhex(v['frame_signature']), np.uint8).astype(float).reshape(-1,3)
                                      for v in (q,other)]
                        from .cue_continuity import continuous
                        later = other if cue>row['cue'] else q
                        if not continuous(q['frame_signature'],other['frame_signature'],verified=later.get('transition_verified',False)):
                            reject('shot_discontinuity')
                            continue
                        offset = translation(q,other,*signatures)
                    if len(torso_points(geom))<3:
                        reject('insufficient_peer_torso')
                        continue
                    blocked = rejection(data['bounds'],geom['bounds'],points,torso_points(geom),translation=offset)
                    if blocked:
                        reject(blocked[1])
                        continue
                    ranked.append((body_similarity(scores,row,peer),peer['face_person'],peer['id']))
                own, own_id = max(((v,i) for v,p,i in ranked if p==row['face_person']),default=(-1.,None))
                rival, rival_id = max(((v,i) for v,p,i in ranked if p!=row['face_person']),default=(-1.,None))
                neighbour.update(own_score=own if own_id else None, rival_score=rival if rival_id else None,
                                 own_observation=own_id, rival_observation=rival_id)
                neighbour['reason'] = ('no_eligible_witness' if not ranked else 'own_witness_missing' if own_id is None else
                    'low_body_similarity' if own<settings['cohesion_threshold'] else
                    'insufficient_body_margin' if own-rival<settings['ambiguity_margin'] else 'confirmed')
                if own>=settings['cohesion_threshold'] and own-rival>=settings['ambiguity_margin']:
                    supported = True
                if rival>=settings['cohesion_threshold'] and own-rival<settings['ambiguity_margin']:
                    conflicted = True
            reasons = [n['reason'] for n in temporal['neighbours']]
            temporal['reason'] = ('temporal_identity_conflict' if conflicted else 'confirmed' if supported else
                next((r for r in ('insufficient_body_margin','low_body_similarity','own_witness_missing') if r in reasons),
                     'no_eligible_witness' if any(n['candidates'] for n in temporal['neighbours']) else 'no_clean_neighbour'))
        if conflicted:
            contradicted.add(row['id'])
        if (not supported and not local.get('confirmed')) or conflicted:
            questionable.add(row['id'])
            check['reason'] = temporal['reason']
        else:
            check.update(status='confirmed', reason='adjacent_clean_witness' if supported else local['reason'])
    # Two independent clean endpoints can corroborate a longer contact. Each
    # intervening edge already competed against the other physical lane.
    from .independent_evidence import count
    for chain,_ in graph or ():
        previous = None
        for i,r in enumerate(chain):
            if not r.get('face_anchor') or crowded_link(json.loads(r['quality'])['quality']):
                continue
            if previous is not None and r['cue']-chain[previous]['cue']<=settings.get('contact_max_cues',12):
                first = chain[previous]
                owner = r['face_person']
                middle = chain[previous+1:i]
                witnesses = [dict(scan=k['scan'],seconds=k['seconds'],quality=k.get('face_quality') or '{}')
                             for k in (first,r)]
                if (first.get('face_person')==owner and count(witnesses)>=2 and
                    all(k.get('face_person') in (None,owner) for k in middle)):
                    questionable.difference_update(k['id'] for k in middle if k.get('face_person')==owner
                                                   and k['id'] not in contradicted)
                    for k in middle:
                        if k['id'] in evidence and k.get('face_person')==owner and k['id'] not in contradicted:
                            evidence[k['id']].update(status='confirmed', reason='independent_clean_endpoints',
                                                    witness_ids=[first['id'],r['id']])
            previous = i
    return questionable


def body_pair_queries(rows,decisions):
    """Only unresolved/competing auxiliary assignments may change owners."""
    from .crowd_context import pair
    mutable = {'body_core_attachment','body_anchor_quality','body_anchor_pixels','body_anchor_keypoint_rules',
               'ambiguous_or_insufficient_face_body_bank','frozen_face_body_bank','body_group_ambiguous'}
    cues = defaultdict(list)
    for row in rows:
        cues[row['cue']].append(row)
    eligible = []
    for members in cues.values():
        if not 2<=len(members)<=4 or any(r.get('_face_vector') is not None or r.get('face_person') for r in members):
            continue
        old = [decisions[r['id']] for r in members]
        if any(d.get('reason') not in mutable for d in old) or all(d['person'] for d in old) and len({d['person'] for d in old})==len(members):
            continue
        if _distinct(members):
            eligible.extend(members)
    return eligible


def joint_body_claims(rows, decisions, rankings, settings, *, reflection_threshold=1.):
    """Resolve a physical pair against frozen face-seeded body banks."""
    from .crowd_context import pair
    from .joint_claims import solve
    cues = defaultdict(list)
    for row in rows:
        cues[row['cue']].append(row)
    for members in cues.values():
        if not 2<=len(members)<=4 or any(r['id'] not in rankings for r in members):
            continue
        qualities = [json.loads(r['quality']).get('quality',{}) for r in members]
        if any(q.get('query_only') or max(q.get('expected_people',0),q.get('detected_people',0))>len(members) for q in qualities):
            continue
        if not _distinct(members):
            continue
        faces = [r.get('_face_vector') for r in members]
        if len(faces)==2 and all(v is not None for v in faces) and len(faces[0])==len(faces[1]) and float(faces[0] @ faces[1])>=reflection_threshold:
            continue
        previous = [decisions[r['id']] for r in members]
        if all(d['person'] for d in previous) and len({d['person'] for d in previous})==len(members):
            continue
        margin = max(.000001,settings['ambiguity_margin'])
        domains = []
        for r in members:
            options = [(p,mean,mode) for mean,_,p,mode,valid,_ in rankings[r['id']] if valid]
            domains.append([*options,(None,settings['match_threshold']-margin,())])
        winner = solve(domains,lambda i,a,j,b:a[0] is None or a[0]!=b[0],margin)
        if not winner:
            continue
        for i,(owner,score,bank) in enumerate(winner):
            if owner is None:
                continue
            oid = members[i]['id']
            # Only the other physically occupied owner's mode is explained away.
            rival = max((v[5] for v in rankings[oid] if v[2] not in {owner,*(choice[0] for j,choice in enumerate(winner) if j!=i)}),default=-1.)
            if score-rival<margin:
                break
        else:
            for row,(owner,score,bank) in zip(members,winner):
                target = decisions[row['id']]
                if owner is None:
                    target.update(person=None,status='withheld',role='auxiliary',reason='body_copresence_conflict')
                else:
                    target.update(person=owner,status='accepted',role='auxiliary',reason='joint_body_claim',
                                  score=score,witness_ids=[r['id'] for r in bank[:8]])


def bridge_faces(graph, decisions, settings, uncertain, guard):
    """Attach strong bodies between independent strict facial observations."""
    from .body_identity import anchor, quality
    from .independent_evidence import count
    for chain,strengths in graph:
        last = None
        for i,row in enumerate(chain):
            if not row.get('face_anchor') or row['id'] in uncertain:
                continue
            if last is not None and i-last>1 and row['cue']-chain[last]['cue']<=settings.get('contact_max_cues',12):
                first = chain[last]
                owner = decisions[row['id']]['person']
                between = chain[last+1:i]
                refs = [dict(scan=k['scan'],seconds=k['seconds'],quality=k.get('face_quality') or '{}')
                        for k in (first,row)]
                if (owner and decisions[first['id']]['person']==owner and count(refs)>=2
                    and min(strengths[last:i])>=settings['match_threshold']
                    and all(k['id'] not in uncertain and not quality(k).get('query_only')
                            and decisions.get(k['id'],{}).get('person',owner)==owner for k in between)):
                    for k in between:
                        if (k['id'] not in decisions and anchor(k,settings)
                            and guard.allows(k,[first,row],owner)):
                            decisions[k['id']] = dict(person=owner,status='accepted',role='auxiliary',
                                reason='adjacent_face_body_bridge',witness_ids=[first['id'],row['id']])
            last = i


def _distinct(members):
    from .crowd_context import pair
    geometry=[json.loads(r['geometry']) for r in members]
    return all(pair([a,b]) is not None for i,a in enumerate(geometry) for b in geometry[i+1:])
