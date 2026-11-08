"""Scene-local body appearances derived from disjoint, face-rooted VTT tracks."""
from collections import defaultdict
import math
import json
import numpy as np


def frame_signature(image):
    # Low-frequency RGB scene context for a cut veto, computed once per cue.
    with image.resize((4, 4)) as small, small.convert('RGB') as rgb:
        return rgb.tobytes().hex()


def orientation_measurement(person):
    points = person['keypoints']
    visibility = person.get('keypoint_visibility')
    def visible(i):
        return min(points[i]) >= 0 and (visibility is None or visibility[i] >= .5)
    visible_torso = [i for i in (5,6,11,12) if visible(i)]
    evidence = dict(label='unknown', visible_torso=visible_torso, confidence=0.)
    if len(visible_torso)==3 and not all(visible(i) for i in (1,2)):
        # Signed span of the complete pair along the torso's anatomical axis.
        top = [np.asarray(points[i]) for i in visible_torso if i<7]
        bottom = [np.asarray(points[i]) for i in visible_torso if i>10]
        axis = np.mean(top,axis=0)-np.mean(bottom,axis=0)
        height = np.linalg.norm(axis)
        pair = top if len(top)==2 else bottom
        if height>=6 and (pair[0]-pair[1]) @ np.array([-axis[1],axis[0]]) < -.27*height**2:
            evidence['possible'] = 'back'
    if len(visible_torso)!=4:
        return evidence
    ls, rs, lh, rh = (np.asarray(points[i]) for i in (5,6,11,12))
    height = float(np.linalg.norm((ls+rs-lh-rh)/2))
    if height < 6:
        return evidence
    up = (ls+rs-lh-rh)/(2*height)
    across = np.array([-up[1], up[0]])
    shoulders, hips = float((ls-rs)@across/height), float((lh-rh)@across/height)
    evidence.update(shoulder_span=shoulders,hip_span=hips)
    if shoulders*hips < 0:
        return evidence
    if max(abs(shoulders),abs(hips)) < .27:
        label = 'side'
    elif min(abs(shoulders),abs(hips)) < .13:
        return evidence
    elif shoulders > 0:
        label = 'front'
    else:
        label = 'unknown' if all(visible(i) for i in (1,2)) else 'back'
        if label=='unknown':
            evidence.update(possible='back', conflict='predicted_eyes')
    confidence = min((visibility[i] if visibility is not None else 1.) for i in visible_torso)
    evidence.update(label=label,confidence=float(confidence) if label!='unknown' else 0.)
    return evidence


def orientation(person):
    return orientation_measurement(person)['label']


def recover(rows, decisions, seeds, settings, scores, guard, *, occupied=()):
    """Build one generation of auxiliary claims. No recovered sample becomes an anchor."""
    from .body_identity import quality
    if not seeds and not any(quality(r).get('orientation') in ('side','unknown') for r in rows):
        for row in rows:
            if not row.get('face_person') and quality(row).get('orientation') == 'back':
                decisions[row['id']].setdefault('track_blocker',dict(reason='no_face_root', cue=row['cue']))
        return
    by_cue = defaultdict(list)
    geometry = {}
    torsos = {}
    root = {r['id']: owner for owner, bank in seeds.items() for r in bank}
    for row in rows:
        q = quality(row)
        if not q.get('frame_signature'):
            continue
        person = json.loads(row['geometry'])
        box = person.get('bounds')
        if box is None:
            continue
        if person.get('keypoints') is not None:
            measurement = orientation_measurement(person)
            q['orientation'] = measurement['label']
            decisions[row['id']]['orientation_measurement'] = measurement
        signature = np.frombuffer(bytes.fromhex(q['frame_signature']), dtype=np.uint8).astype(float)
        geometry[row['id']] = (q, box, signature)
        if person.get('keypoints'):
            from .track_geometry import torso_points
            torsos[row['id']] = torso_points(person)
        by_cue[row['cue']].append(row)
    if not geometry:
        return

    from .perspective_shots import boundaries
    shot_ids, boundary = boundaries(by_cue, geometry)
    # A filtered partner is still an unknown competitor in a crowded transition.
    partial = {cue for cue,members in by_cue.items()
               if any(max(geometry[r['id']][0].get('detected_people',len(members)),
                          geometry[r['id']][0].get('expected_people',0)) != len(members)
                      for r in members)}

    track_threshold = settings.get('track_similarity_threshold')
    if track_threshold is None:
        track_threshold = settings['cohesion_threshold']
    outgoing_failures, incoming_failures = {}, {}
    appearance_edges = {}
    camera_offsets = {}
    def transition(a, b, *, diagnose=False, raw=False):
        def fail(order, reason, measured, limit):
            if diagnose:
                # Retain one candidate per endpoint, preferring the one that
                # cleared the most gates and came closest to the failed gate.
                priority = (order,-abs(measured-limit),a['id'],b['id'])
                for row,peer,failures in ((a,b,outgoing_failures),(b,a,incoming_failures)):
                    prior = failures.get(row['id'])
                    if prior is None or priority > prior[0]:
                        detail = dict(reason=reason,cue=peer['cue'],observation_cue=row['cue'],
                                      peer=peer['id'],measured=round(float(measured),7),limit=limit)
                        failures[row['id']] = priority,detail
            return None
        qa, ba, sa = geometry[a['id']]
        qb, bb, sb = geometry[b['id']]
        gap = b['seconds'] - a['seconds']
        skipped = b['cue'] - a['cue'] > 1
        if skipped:
            continuous = (shot_ids[a['cue']] == shot_ids[b['cue']] and np.abs(sa-sb).mean() <= 8)
        else:
            continuous = boundary[b['cue']] == 'continuous'
        if not 0 < gap <= 12:
            return fail(0,'elapsed_seconds',gap,12)
        if not continuous:
            if not diagnose:
                return None
            from .perspective_shots import measurements
            difference,structure,colour = measurements(sa.reshape(-1,3),sb.reshape(-1,3))
            if skipped and shot_ids[a['cue']]!=shot_ids[b['cue']]:
                return fail(1,'shot_change',1,0)
            if structure>16:
                return fail(1,'cut_structure',structure,16)
            if colour>30:
                return fail(1,'cut_colour',colour,30)
            return fail(1,'uncertain_shot',difference,8 if skipped else 20)
        from .track_geometry import rejection, translation
        pair = a['cue'],b['cue']
        if pair not in camera_offsets:
            camera_offsets[pair] = translation(qa,qb,sa,sb) if not skipped else None
        blocked = rejection(ba, bb, torsos.get(a['id'], {}), torsos.get(b['id'], {}), gap,
                            translation=camera_offsets[pair])
        if blocked:
            return fail(*blocked)
        similarity = scores.face_similarity(a['_vector'],b['_vector'])
        if diagnose:
            appearance_edges[a['id'],b['id']] = similarity
        return similarity if raw or similarity >= track_threshold else fail(
            5,'body_similarity',similarity,track_threshold)

    # Ambiguous transitions, cuts and sparse cue gaps end a track.
    next_row, stops = {}, {}
    occupied = set(occupied) | set(by_cue)
    margin = max(.025, settings['ambiguity_margin'])
    for cue, left in by_cue.items():
        target = cue+1 if cue+1 in occupied else cue+2
        right = by_cue.get(target, ())
        default_stop = ('cut' if boundary.get(target)=='cut' else
                        'cue_gap' if not right else 'geometry_or_appearance')
        for row in left:
            stops[row['id']] = dict(reason=default_stop, cue=target)

        appearance_edges.clear()
        forward, backward = defaultdict(list), defaultdict(list)
        for a in left:
            for b in right:
                score = transition(a,b,diagnose=True)
                score = appearance_edges.get((a['id'],b['id']))
                if score is not None:
                    forward[a['id']].append((score,b['id']))
                    backward[b['id']].append((score,a['id']))
        # Only alternatives passing the same time, cut and geometry gates
        # bound a threshold proposal; keep one competitor per failed endpoint.
        for members,failures,forward_order in ((left,outgoing_failures,True),(right,incoming_failures,False)):
            for member in members:
                failure = failures.get(member['id'])
                if not failure or failure[1]['reason']!='body_similarity':
                    continue
                detail = failure[1]
                source,target_id = ((member['id'],detail['peer']) if forward_order else
                                    (detail['peer'],member['id']))
                rivals = [(value,b if a==source else a) for (a,b),value in appearance_edges.items()
                          if (a==source and b!=target_id) or (b==target_id and a!=source)]
                if rivals:
                    detail['partner_score'],detail['partner_observation'] = max(rivals)
        for a in left:
            edges = sorted(forward[a['id']], reverse=True)
            if not edges or edges[0][0]<track_threshold:
                if a['id'] in outgoing_failures and right:
                    stops[a['id']] = outgoing_failures[a['id']][1]
                continue
            if len(edges)>1 and edges[0][0]-edges[1][0] < margin:
                stops[a['id']] = dict(reason='competing_transition', cue=target, rival=edges[1][1],
                                          measured=edges[0][0]-edges[1][0], limit=margin)
                continue
            _, bid = edges[0]
            rivals = sorted(backward[bid], reverse=True)
            if rivals[0][1] == a['id'] and (len(rivals)==1 or rivals[0][0]-rivals[1][0] >= margin):
                next_row[a['id']] = bid
            else:
                stops[a['id']] = dict(reason='competing_transition', cue=target, rival=rivals[0][1],
                                          measured=edges[0][0]-max((v for v,i in rivals if i!=a['id']),default=-1.), limit=margin)
    # A side/uncertain cue can query a rear bank only when bracketed by two
    # back observations on unique links. It never trains the rear bank.
    previous_links = {end:start for start,end in next_row.items()}
    lookup = {r['id']:r for r in rows}
    from .crowd_context import torso
    for start,(measurement,_,_) in geometry.items():
        kind = measurement.get('orientation')
        if kind not in ('back','side'):
            continue
        middle = []
        end = next_row.get(start)
        while end and len(middle)<4 and geometry[end][0].get('orientation') in (('side','unknown') if kind=='back' else ('unknown',)):
            middle.append(end)
            end = next_row.get(end)
        if not middle or len(middle)>3 or not end or geometry[end][0].get('orientation') != kind:
            continue
        if any(lookup[i]['cue'] in partial or lookup[i].get('face_person') for i in middle):
            continue
        if len(middle)>1 and any(torso(json.loads(lookup[i]['geometry'])) is None for i in middle):
            continue
        for i in middle:
            decisions[i].update(query_perspective=kind,orientation_confidence='temporal')
    for oid,decision in decisions.items():
        if (decision.get('orientation_measurement',{}).get('possible')!='back'
                or lookup[oid]['cue'] in partial or lookup[oid].get('face_person')):
            continue
        neighbours = [n for n in (previous_links.get(oid),next_row.get(oid)) if n is not None]
        eye_conflict = decision['orientation_measurement'].get('conflict')=='predicted_eyes'
        if eye_conflict:
            if lookup[oid].get('_face_vector') is not None:
                continue
            neighbours = [n for n in neighbours if abs(lookup[n]['cue']-lookup[oid]['cue'])==1
                          and lookup[n]['cue'] not in partial and shot_ids[lookup[n]['cue']]==shot_ids[lookup[oid]['cue']]]
        directions = {geometry[n][0].get('orientation') for n in neighbours}
        if 'back' in directions and not directions.intersection(('front','side')):
            decision.update(query_perspective='back',orientation_confidence=('rear_torso_and_neighbour' if eye_conflict else 'partial_torso_and_neighbour'))
    if not seeds:
        for row in rows:
            decisions[row['id']].setdefault('track_blocker',dict(reason='no_face_root',cue=row['cue']))
        return
    from .body_identity import copresent
    measured = defaultdict(list)
    for row in rows:
        assigned = decisions[row['id']]
        if assigned.get('reason') == 'measured_face_association':
            measured[assigned['person']].append(row)
    looks = defaultdict(list)
    source_paths = {}
    for start, owner in root.items():
        path = []
        successor = next_row.get(start)
        skipped = 0
        previous = lookup[start]
        while successor and successor not in root:
            member = lookup[successor]
            skipped += member['cue']-previous['cue']-1
            previous = member
            if skipped > 1 or member['seconds'] - lookup[start]['seconds'] > 60:
                successor = None
                break
            if member.get('face_person') is not None:
                if decisions[successor].get('person') != owner:
                    break
            else:
                path.append(member)
            successor = next_row.get(successor)
        if not path:
            continue
        last = path[-1]
        if successor in root and (lookup[successor]['seconds'] - lookup[start]['seconds'] > 60
                or skipped + lookup[successor]['cue'] - previous['cue'] - 1 > 1):
            successor = None
        if root.get(successor) != owner:
            stop = stops.get(last['id'], dict(reason='no_returning_face', cue=last['cue']))
            if successor is not None:
                stop = dict(reason='different_face_root', cue=lookup[successor]['cue'])
            for member in path:
                decision = decisions[member['id']]
                decision['track_blocker'] = stop
                if stop['reason'] == 'different_face_root':
                    decision.update(person=None, role='auxiliary', status='withheld', reason='different_face_root')
                if stop['reason'] not in ('competing_transition','different_face_root'):
                    decisions[member['id']]['open_face_root'] = dict(owner=owner, observation=start)
            continue
        incomplete = any(r['cue'] in partial and not r.get('face_person') for r in path)
        for r in path:
            if incomplete:
                decisions[r['id']]['open_face_root'] = dict(owner=owner, observation=start, returning=successor)
                continue
            source_paths[r['id']] = (start, successor)
            q = geometry[r['id']][0]
            if (q.get('orientation') in ('front','back','side') and not q.get('query_only')
                    and not copresent([r], measured[owner]) and guard.allows(r,seeds[owner],owner)):
                looks[owner, q['orientation']].append((start, successor, r))

    # A prefix ending at a strict face may query independent galleries. It has
    # only one strict endpoint and can never supply a training witness.
    for end,owner in root.items():
        cursor = end
        skipped = 0
        while before := previous_links.get(cursor):
            row = lookup[before]
            skipped += lookup[cursor]['cue']-row['cue']-1
            if before in root or skipped > 1 or lookup[end]['seconds']-row['seconds']>60:
                break
            decision = decisions[before]
            if row.get('face_person') and decision.get('person') != owner:
                break
            blocker = decision.get('track_blocker',{}).get('reason')
            if blocker in ('different_face_root','competing_transition'):
                break
            hint = decision.get('open_face_root')
            if hint and hint['owner']!=owner:
                decision.update(person=None,status='withheld',role='auxiliary',reason='different_face_root',
                                track_blocker=dict(reason='different_face_root',cue=lookup[end]['cue']))
                break
            if not row.get('face_person') and before not in source_paths:
                decision.setdefault('open_face_root',dict(owner=owner, observation=end))
            cursor = before
    # Only a unique, same-shot path to a nearby strict face nominates an
    # owner for a lower appearance threshold in the global competitor check.
    for oid,decision in decisions.items():
        hint = decision.get('open_face_root',{})
        origin = lookup.get(hint.get('observation'))
        row = lookup[oid]
        if (origin is not None and row['cue'] not in partial and
                shot_ids.get(origin['cue'])==shot_ids.get(row['cue']) and
                abs(origin['cue']-row['cue'])<=3 and settings.get('scene_context_bonus',0.)>0):
            decision['shot_context'] = dict(owner=hint['owner'],root=origin['id'],
                                            bonus=settings['scene_context_bonus'])
    if settings.get('scene_context_bonus',0.)>0:
        record_scene_roster(by_cue,root,shot_ids,partial,decisions,settings.get('roster_support',3),
                            weighted=settings.get('scene_bonus_mode')=='evidence',settings=settings)
    for cue in partial:
        for row in by_cue[cue]:
            decision = decisions[row['id']]
            if not row.get('face_person') and not decision.get('open_face_root',{}).get('returning'):
                if decision.get('track_blocker',{}).get('reason') not in ('different_face_root','competing_transition'):
                    decision['track_blocker'] = dict(reason='unresolved_partner',cue=cue)

    # Keep training routes independent by shot, face roots, pixels and cue.
    banks = {}
    for key, tracks in looks.items():
        from .person_anchor import pixels
        def witness_rank(track):
            row = track[2]
            assessed = geometry[row['id']][0]
            return (-assessed.get('overlap', 0.), assessed.get('keypoints', 0),
                    assessed.get('sharpness', 0.), pixels(row['quality']) or 0, row['id'])
        from .rooted_witnesses import modes
        for number, kept in enumerate(modes(tracks,settings,scores,witness_rank,lambda row:shot_ids[row['cue']])):
            validated = []
            for start,end,witness in kept:
                # Challenge each witness with face-confirmed foreign tracks from
                # a different shot; reassess exclusions on every evidence rebuild.
                peers = [r for _,_,r in kept if r is not witness]
                rivals = [(scores.face_similarity(witness['_vector'],r['_vector']), other,r['id'])
                          for other,bank in seeds.items() if other!=key[0] for r in bank
                          if r['id'] in geometry and shot_ids[r['cue']]!=shot_ids[witness['cue']]]
                if rivals:
                    own = min((scores.face_similarity(witness['_vector'],r['_vector']) for r in peers),default=-1.)
                    rival,other,rid = max(rivals)
                    if rival >= settings['match_threshold'] and own-rival < settings['ambiguity_margin']:
                        decisions[witness['id']]['view_exclusion'] = dict(reason='held_out_rival',
                            owner=key[0],rival=other,observation=rid,own_score=own,rival_score=rival)
                        continue
                validated.append((start,end,witness))
            kept = validated
            for start,end,witness in kept:
                decisions[witness['id']]['rooted_track'] = dict(owner=key[0], perspective=key[1],
                                                                roots=[start,end], shot=shot_ids[witness['cue']],
                                                                boundary=boundary[witness['cue']])
            if len(kept) >= max(2,settings['attach_min_support']):
                witnesses = [r for _,_,r in kept]
                proof = [(start,end,r['id']) for start,end,r in kept]
                banks[(*key,number)] = witnesses, proof
                for start,end,witness in proof:
                    decisions[witness]['rooted_view'] = dict(owner=key[0], perspective=key[1],
                                                             roots=[start,end], tracks=proof,
                                                             shot=shot_ids[lookup[witness]['cue']],
                                                             boundary=boundary[lookup[witness]['cue']])

    for row in rows:
        decision = decisions[row['id']]
        if not row.get('face_person') and decision.get('status') != 'accepted':
            failure = incoming_failures.get(row['id'])
            stop = stops.get(row['id']) if row['id'] not in next_row else None
            if failure and (stop is None or stop['reason'] in ('cue_gap','geometry_or_appearance')):
                stop = failure[1]
            decision.setdefault('track_blocker', stop or dict(reason='no_independent_view',cue=row['cue']))
    from .crossing_tracks import crossings
    occluded = {cue for cue,members in by_cue.items() if len(members)==1
                and boundary.get(cue)=='continuous'
                and boundary.get(cue+1)=='continuous'
                and shot_ids.get(cue-1)==shot_ids.get(cue)==shot_ids.get(cue+1)}
    crossed = crossings({cue:members for cue,members in by_cue.items() if cue not in partial or cue in occluded},
                        root,decisions,transition,margin,occluded=occluded,
                        raw=lambda a,b:transition(a,b,raw=True),
                        visible_single=lambda row: row['cue'] not in partial and len(torsos.get(row['id'],{}))>=3
                        and not geometry[row['id']][0].get('query_only'))
    for cue in partial:
        for row in by_cue[cue]:
            if geometry[row['id']][0].get('query_only_reason')=='unresolved_partner':
                decisions[row['id']].update(person=None,status='withheld',role='auxiliary',
                                            reason='unresolved_partner')
    from .fragment_endpoints import nominations
    from .fragment_endpoints import split_owner_nominations
    for member,proof in split_owner_nominations(looks,lookup,root,shot_ids,scores,settings):
        decisions[member].setdefault('endpoint_pairs',[]).append(proof)
    bridges = nominations(looks,lookup,root,shot_ids,scores,settings)
    if not banks and not crossed and not bridges:
        return
    owner_rows = defaultdict(list)
    rival_evidence = {}
    entrant_banks = {}
    for r in rows:
        p = decisions[r['id']]['person']
        if p is not None:
            owner_rows[p].append(r)
    candidates = []
    for oid,(owner,proof,score) in bridges.items():
        row = lookup[oid]
        if (oid not in crossed and not row.get('face_person') and
                decisions[oid].get('reason') not in ('unresolved_partner','different_face_root') and
                (decisions[oid]['status']!='accepted' or decisions[oid]['reason']=='body_scene_core') and
                not copresent([row],owner_rows[owner]) and guard.allows(row,seeds[owner],owner)):
            candidates.append((oid,owner,'endpoint_faces',proof,score))
    for oid,(owner,first,last,score) in crossed.items():
        row = lookup[oid]
        prior = decisions[oid]
        if prior.get('status')=='accepted' and prior.get('reason')!='body_scene_core':
            continue
        if not copresent([row],owner_rows[owner]) and guard.allows(row,seeds[owner],owner):
            candidates.append((oid,owner,'crossing',[(first['id'],last['id'],oid)],score))
    for row in rows:
        decision = decisions[row['id']]
        if (row['id'] in crossed or decision.get('track_blocker', {}).get('reason') in ('different_face_root','competing_transition','unresolved_partner')
                or row['id'] not in geometry or row.get('face_person') or
                decision['status']=='accepted' and decision.get('reason')!='body_scene_core'):
            continue
        kind = decision.get('query_perspective') or geometry[row['id']][0].get('orientation')
        for (owner, perspective, _), (bank, proof) in banks.items():
            if decision.get('open_face_root', {}).get('owner', owner) != owner:
                continue
            if perspective != kind or copresent([row],owner_rows[owner]) or not guard.allows(row,bank,owner):
                continue
            # A rooted track has its own temporal proof, but cannot vote for
            # its own appearance. Exact source pixels never add a second vote.
            pixel = geometry[row['id']][0].get('pixel_digest')
            external = [(w, track) for w, track in zip(bank, proof)
                        if shot_ids[w['cue']] != shot_ids[row['cue']] and not set(track[:2]).intersection(source_paths.get(row['id'], ()))
                        and not (pixel and pixel == geometry[w['id']][0].get('pixel_digest'))]
            required = 1 if row['id'] in source_paths else max(2,settings['attach_min_support'])
            if len(external)<required:
                continue
            comparisons = [scores.face_similarity(row['_vector'], w['_vector']) for w, _ in external]
            mean, low = sum(comparisons)/len(comparisons), min(comparisons)
            bonus = owner_bonus(decision,owner,settings.get('scene_context_bonus',0.))
            if mean < settings['match_threshold']-bonus or low < settings['cohesion_threshold']:
                continue
            # Every foreign owner including provisional scene groups competes.
            competing = [(scores.compare([row],others)[0], p) for p,others in owner_rows.items()
                         if p not in (owner,decision['person']) and not copresent([row],others)
                         and guard.allows(row,others)]
            competing.extend((scores.compare([row],other)[0], p) for (p,_,_),(other,_) in banks.items() if p!=owner)
            runner, rival = max(competing, default=(-1.,None))
            if bonus:
                from .roster_presence import entrant_score
                from .body_identity import independent
                own = decision.get('person'),shot_ids[row['cue']]
                if own not in entrant_banks:
                    entrant_banks[own] = independent((r for r in owner_rows.get(own[0],())
                        if r['cue'] in shot_ids and shot_ids[r['cue']]!=own[1]),settings)
                entrant = entrant_score(scores,row,entrant_banks[own],settings,settings['scene_context_bonus'])
                if entrant>runner:
                    runner,rival = entrant,'new_participant'
            if rival is not None:
                rival_evidence[row['id'],owner,mean] = dict(rival=rival,runner=runner)
            if mean-runner < margin:
                decision['track_blocker'] = dict(reason='competing_owner', cue=row['cue'], rival=rival)
            else:
                candidates.append((row['id'],owner,perspective,[track for _,track in external],mean))
    # Several modes can support one claim. Keep its strongest independent proof.
    best = {}
    for item in candidates:
        address = item[:2]
        if address not in best or item[-1] > best[address][-1]:
            best[address] = item
    by_observation = defaultdict(list)
    for item in best.values():
        by_observation[item[0]].append(item)
    candidates = []
    for oid,options in sorted(by_observation.items()):
        options.sort(key=lambda item:(-item[-1],item[1]))
        if len(options)>1 and options[0][-1]-options[1][-1]<margin:
            decisions[oid]['track_blocker'] = dict(reason='competing_owner',rival=options[1][1],cue=lookup[oid]['cue'])
        else:
            candidates.append(options[0])
    proposed = {oid: owner for oid, owner, *_ in candidates}
    # Check all core votes before moving anything. Weak satellites cannot
    # veto a coherent core, but a measured face or simultaneous claim can.
    from collections import Counter
    claims = Counter((owner,lookup[oid]['cue']) for oid,owner in proposed.items())
    protected = set()
    split = set()
    retry = {}
    for old,members in owner_rows.items():
        core = [r for r in members if decisions[r['id']]['reason']=='body_scene_core']
        if not core:
            continue
        targets = {proposed.get(r['id']) for r in core}
        if None in targets or len(targets)!=1:
            if settings.get('tracklet_matching'):
                from .body_tracklets import split_members
                def occupied(row,target):
                    from .crowd_context import pair
                    return ((row.get('face_person') and decisions[row['id']].get('person') not in (None,target))
                            or any(other['cue']==row['cue'] and other['id']!=row['id'] and
                                   pair([json.loads(row['geometry']),json.loads(other['geometry'])]) is not None
                                   for other in seeds.get(target,())))
                allowed = split_members(core,proposed,occupied,settings)
                if allowed and all(claims[proposed[oid],lookup[oid]['cue']]==1 for oid in allowed):
                    split.update(allowed)
                    if len({proposed[oid] for oid in allowed})==2:
                        for r in core:
                            if r['id'] not in allowed:
                                decisions[r['id']] = dict(person=None,status='withheld',role='auxiliary',
                                    reason='split_contact_interior',track_blocker=dict(reason='unresolved_partner',cue=r['cue']))
            protected.add(old)
            continue
        target = next(iter(targets))
        if (any(r.get('face_person') or not guard.allows(r,seeds[target],target) for r in members)
                or any(claims[target,r['cue']]!=1 for r in core)):
            protected.add(old)
            continue
        for r in members:
            if proposed.get(r['id']) != target:
                retry[r['id']] = target
    for oid,target in retry.items():
        decisions[oid] = dict(person=None,status='withheld',role='auxiliary',
                              reason='fragment_auxiliary_retry',retry_owner=target,
                              retry_when='confirmed_core_changes')
    for oid,owner,angle,proof,score in candidates:
        if oid in retry or (decisions[oid]['person'] in protected and oid not in split) or claims[owner,lookup[oid]['cue']] != 1:
            continue
        previous = decisions[oid]
        rooted = {k:v for k,v in previous.items() if k in ('rooted_view','rooted_track','view_exclusion','orientation_measurement')}
        decisions[oid] = dict(person=owner, status='accepted', role='auxiliary',
                              reason='face_rooted_body_view', perspective=angle,
                              source_tracks=proof, score=score)
        if angle == 'crossing':
            decisions[oid].update(reason='face_rooted_crossing', perspective=geometry[oid][0].get('orientation'))
        if angle == 'endpoint_faces':
            decisions[oid].update(reason='fragment_face_endpoints',perspective=geometry[oid][0].get('orientation'))
        if angle not in ('crossing','endpoint_faces'):
            decisions[oid].update(rival_evidence.get((oid,owner,score),{}))
        if angle not in ('crossing','endpoint_faces') and score<settings['match_threshold']:
            decisions[oid]['roster_context'] = dict(owner=owner,
                bonus=owner_bonus(previous,owner,settings.get('scene_context_bonus',0.)),
                witnesses=previous.get('roster_witnesses',{}).get(owner,[]))
        decisions[oid].update(rooted)


def record_scene_roster(by_cue, roots, shots, partial, decisions, minimum, *, weighted=False, settings=None):
    from .body_identity import independent, defaults
    settings = settings or defaults()
    sources = defaultdict(list)
    queries = defaultdict(list)
    for cue,rows in sorted(by_cue.items()):
        for row in rows:
            owner = roots.get(row['id'])
            if owner is not None:
                sources[owner].append(row)
            elif not row.get('face_person') and cue not in partial:
                queries[shots[cue]].append(row)
    for owner,rows in sorted(sources.items()):
        supported = independent(rows,settings)[:minimum]
        if len(supported)<minimum:
            continue
        origins = {shots[r['cue']] for r in supported}
        default_ids = [r['id'] for r in supported]
        for shot,ids in queries.items():
            witnesses = default_ids
            other_shots = None
            if weighted or shot in origins:
                other_shots = independent((r for r in rows if shots[r['cue']]!=shot),settings)
                if len(other_shots)<minimum:
                    continue
                witnesses = [r['id'] for r in other_shots[:minimum]]
            if weighted:
                from .roster_presence import Presence
                presence = Presence(other_shots,shots)
            for row in ids:
                oid = row['id']
                proof_ids = witnesses
                if weighted:
                    factor,nearby = presence.weight(row)
                    decisions[oid].setdefault('roster_weights',{})[owner] = factor
                    proof_ids = list(dict.fromkeys([*witnesses,*nearby]))
                decisions[oid].setdefault('scene_roster',[]).append(owner)
                decisions[oid].setdefault('roster_witnesses',{})[owner] = proof_ids


def owner_bonus(decision,owner,maximum):
    context = decision.get('shot_context',{})
    if owner and context.get('owner')==owner and decision.get('open_face_root',{}).get('owner')==owner:
        return maximum
    if owner not in decision.get('scene_roster',()):
        return 0.
    return maximum*decision.get('roster_weights',{}).get(owner,1.)
