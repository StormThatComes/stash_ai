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
                decisions[row['id']]['track_blocker'] = dict(reason='no_face_root', cue=row['cue'])
        return
    by_cue = defaultdict(list)
    geometry = {}
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
        by_cue[row['cue']].append(row)
    if not geometry:
        return

    from .perspective_shots import boundaries
    shot_ids, boundary = boundaries(by_cue, geometry)
    # A filtered partner is still an unknown competitor in a crowded transition.
    partial = {cue for cue,members in by_cue.items()
               if any(geometry[r['id']][0].get('detected_people',len(members)) != len(members)
                      for r in members)}

    track_threshold = settings.get('track_similarity_threshold')
    if track_threshold is None:
        track_threshold = settings['cohesion_threshold']
    outgoing_failures, incoming_failures = {}, {}
    def transition(a, b, *, diagnose=False):
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
        skipped = b['cue'] - a['cue'] == 2
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
        wa, ha = ba[2]-ba[0], ba[3]-ba[1]
        wb, hb = bb[2]-bb[0], bb[3]-bb[1]
        if min(wa,ha,wb,hb)<=0:
            return fail(2,'empty_box',min(wa,ha,wb,hb),0)
        scale = max(wa/wb,wb/wa,ha/hb,hb/ha)
        if scale>1.5:
            return fail(3,'box_scale',scale,1.5)
        displacement = math.hypot((ba[0]+ba[2]-bb[0]-bb[2])/2,
                                  (ba[1]+ba[3]-bb[1]-bb[3])/2)
        relative = displacement/min(math.hypot(wa,ha), math.hypot(wb,hb))
        if relative>.30:
            return fail(4,'displacement',relative,.30)
        similarity = scores.face_similarity(a['_vector'],b['_vector'])
        return similarity if similarity >= track_threshold else fail(
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

        forward, backward = defaultdict(list), defaultdict(list)
        for a in left:
            for b in right:
                score = transition(a,b,diagnose=True)
                if score is not None:
                    forward[a['id']].append((score,b['id']))
                    backward[b['id']].append((score,a['id']))
        for a in left:
            edges = sorted(forward[a['id']], reverse=True)
            if not edges:
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
                if settings.get('assess_witnesses',True) and peers and rivals:
                    own = min(scores.face_similarity(witness['_vector'],r['_vector']) for r in peers)
                    rival,other,rid = max(rivals)
                    if rival >= settings['match_threshold'] and own-rival < settings['ambiguity_margin']:
                        decisions[witness['id']]['view_exclusion'] = dict(reason='held_out_rival',
                            owner=key[0],rival=other,observation=rid,own_score=own,rival_score=rival)
                        continue
                validated.append((start,end,witness))
            kept = validated
            for start,end,witness in kept:
                decisions[witness['id']]['rooted_track'] = dict(owner=key[0], perspective=key[1],
                                                                roots=[start,end])
            if len(kept) >= max(2,settings['attach_min_support']):
                witnesses = [r for _,_,r in kept]
                proof = [(start,end,r['id']) for start,end,r in kept]
                banks[(*key,number)] = witnesses, proof
                for start,end,witness in proof:
                    decisions[witness]['rooted_view'] = dict(owner=key[0], perspective=key[1],
                                                             roots=[start,end], tracks=proof)

    for row in rows:
        decision = decisions[row['id']]
        if not row.get('face_person') and decision.get('status') != 'accepted':
            failure = incoming_failures.get(row['id'])
            stop = stops.get(row['id']) if row['id'] not in next_row else None
            if failure and (stop is None or stop['reason'] in ('cue_gap','geometry_or_appearance')):
                stop = failure[1]
            decision.setdefault('track_blocker', stop or dict(reason='no_independent_view',cue=row['cue']))
    from .crossing_tracks import crossings
    crossed = crossings({cue:members for cue,members in by_cue.items() if cue not in partial},
                        root,decisions,transition,margin)
    if not banks and not crossed:
        return
    owner_rows = defaultdict(list)
    rival_evidence = {}
    for r in rows:
        p = decisions[r['id']]['person']
        if p is not None:
            owner_rows[p].append(r)
    candidates = []
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
                        if shot_ids[w['cue']] != shot_ids[row['cue']] and track[:2] != source_paths.get(row['id'])
                        and not (pixel and pixel == geometry[w['id']][0].get('pixel_digest'))]
            if not external:
                continue
            comparisons = [scores.face_similarity(row['_vector'], w['_vector']) for w, _ in external]
            mean, low = sum(comparisons)/len(comparisons), min(comparisons)
            if mean < settings['match_threshold'] or low < settings['cohesion_threshold']:
                continue
            # Every foreign owner including provisional scene groups competes.
            competing = [(scores.compare([row],others)[0], p) for p,others in owner_rows.items()
                         if p not in (owner,decision['person']) and not copresent([row],others)
                         and guard.allows(row,others)]
            competing.extend((scores.compare([row],other)[0], p) for (p,_,_),(other,_) in banks.items() if p!=owner)
            runner, rival = max(competing, default=(-1.,None))
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
    candidates = list(best.values())
    proposed = {oid: owner for oid, owner, *_ in candidates}
    # Check all core votes before moving anything. Weak satellites cannot
    # veto a coherent core, but a measured face or simultaneous claim can.
    from collections import Counter
    claims = Counter((owner,lookup[oid]['cue']) for oid,owner in proposed.items())
    protected = set()
    retry = {}
    for old,members in owner_rows.items():
        core = [r for r in members if decisions[r['id']]['reason']=='body_scene_core']
        if not core:
            continue
        targets = {proposed.get(r['id']) for r in core}
        if None in targets or len(targets)!=1:
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
        if oid in retry or decisions[oid]['person'] in protected or claims[owner,lookup[oid]['cue']] != 1:
            continue
        rooted = {k:v for k,v in decisions[oid].items() if k in ('rooted_view','rooted_track','view_exclusion','orientation_measurement')}
        decisions[oid] = dict(person=owner, status='accepted', role='auxiliary',
                              reason='face_rooted_body_view', perspective=angle,
                              source_tracks=proof, score=score)
        if angle == 'crossing':
            decisions[oid].update(reason='face_rooted_crossing', perspective=geometry[oid][0].get('orientation'))
        if angle != 'crossing':
            decisions[oid].update(rival_evidence.get((oid,owner,score),{}))
        decisions[oid].update(rooted)
