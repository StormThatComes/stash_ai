"""Corroborate a crowded face/body link before it can train a person's bank."""
from collections import defaultdict
import json

from .track_geometry import rejection, torso_points, translation
import numpy as np


def crowded_link(q):
    return q.get('detected_people',1)>1 and q.get('overlap',0)>=.25


def questionable_links(rows,settings,scores,graph=()):
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
        if len(points)>=3 and q.get('view_shot') is not None:
            for cue in (row['cue']-1,row['cue']+1):
                ranked = []
                for peer,other in clean[cue]:
                    if other.get('view_shot')!=q['view_shot']:
                        continue
                    geom = json.loads(peer['geometry'])
                    offset = None
                    if q.get('frame_signature') and other.get('frame_signature'):
                        signatures = [np.frombuffer(bytes.fromhex(v['frame_signature']), np.uint8).astype(float).reshape(-1,3)
                                      for v in (q,other)]
                        from .perspective_shots import change
                        if change(*signatures,True)!='continuous':
                            continue
                        offset = translation(q,other,*signatures)
                    if len(torso_points(geom))<3 or rejection(data['bounds'],geom['bounds'],points,torso_points(geom),translation=offset):
                        continue
                    ranked.append((scores.face_similarity(row['_vector'],peer['_vector']),peer['face_person']))
                own = max((v for v,p in ranked if p==row['face_person']),default=-1.)
                rival = max((v for v,p in ranked if p!=row['face_person']),default=-1.)
                if own>=settings['cohesion_threshold'] and own-rival>=settings['ambiguity_margin']:
                    supported = True
                if rival>=settings['cohesion_threshold'] and own-rival<settings['ambiguity_margin']:
                    conflicted = True
        if conflicted:
            contradicted.add(row['id'])
        if not supported or conflicted:
            questionable.add(row['id'])
    # Two independent clean endpoints can corroborate a longer contact. Each
    # intervening edge already competed against the other physical lane.
    from .independent_evidence import count
    for chain,_ in graph or ():
        previous = None
        for i,r in enumerate(chain):
            if not r.get('face_anchor') or crowded_link(json.loads(r['quality'])['quality']):
                continue
            if previous is not None and i-previous<=12:
                first = chain[previous]
                owner = r['face_person']
                middle = chain[previous+1:i]
                witnesses = [dict(scan=k['scan'],seconds=k['seconds'],quality=k.get('face_quality') or '{}')
                             for k in (first,r)]
                if (first.get('face_person')==owner and count(witnesses)>=2 and
                    all(k.get('face_person') in (None,owner) for k in middle)):
                    questionable.difference_update(k['id'] for k in middle if k.get('face_person')==owner
                                                   and k['id'] not in contradicted)
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
        if len(members)!=2 or any(r.get('_face_vector') is not None or r.get('face_person') for r in members):
            continue
        old = [decisions[r['id']] for r in members]
        if any(d.get('reason') not in mutable for d in old) or all(d['person'] for d in old) and old[0]['person']!=old[1]['person']:
            continue
        if pair([json.loads(r['geometry']) for r in members]) is not None:
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
        if len(members)!=2 or any(r['id'] not in rankings for r in members):
            continue
        qualities = [json.loads(r['quality']).get('quality',{}) for r in members]
        if any(q.get('query_only') or max(q.get('expected_people',0),q.get('detected_people',0))>2 for q in qualities):
            continue
        if pair([json.loads(r['geometry']) for r in members]) is None:
            continue
        faces = [r.get('_face_vector') for r in members]
        if all(v is not None for v in faces) and len(faces[0])==len(faces[1]) and float(faces[0] @ faces[1])>=reflection_threshold:
            continue
        previous = [decisions[r['id']] for r in members]
        if all(d['person'] for d in previous) and previous[0]['person']!=previous[1]['person']:
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
            rival = max((v[5] for v in rankings[oid] if v[2] not in (owner,winner[1-i][0])),default=-1.)
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
            if last is not None and 1<i-last<=6:
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
