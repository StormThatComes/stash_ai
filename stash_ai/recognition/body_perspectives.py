"""Scene-local body appearances derived from disjoint, face-rooted VTT tracks."""
from collections import defaultdict
import math
import json
import numpy as np


def frame_signature(image):
    # Low-frequency RGB scene context for a cut veto, computed once per cue.
    with image.resize((4, 4)) as small, small.convert('RGB') as rgb:
        return rgb.tobytes().hex()


def orientation(person):
    points = person['keypoints']
    visibility = person.get('keypoint_visibility')
    def visible(i):
        return min(points[i]) >= 0 and (visibility is None or visibility[i] >= .5)
    if not all(visible(i) for i in (5, 6, 11, 12)):
        return 'unknown'
    ls, rs, lh, rh = (np.asarray(points[i]) for i in (5, 6, 11, 12))
    height = float(np.linalg.norm((ls+rs-lh-rh)/2))
    if height < 6:
        return 'unknown'
    # A rolled camera needs a torso-relative horizontal axis. Preserve the
    # anatomical labels produced by the detector, before encoder flips.
    up = (ls+rs-lh-rh)/(2*height)
    across = np.array([-up[1], up[0]])
    shoulders, hips = float((ls-rs)@across/height), float((lh-rh)@across/height)
    if shoulders*hips < 0:
        return 'unknown'
    if max(abs(shoulders),abs(hips)) < .27:
        return 'side'
    if min(abs(shoulders),abs(hips)) < .13:
        return 'unknown'
    if shoulders > 0:
        return 'front'
    # A reversed torso with visible frontal eyes is ambiguous (twist/mirror).
    return 'unknown' if all(visible(i) for i in (1,2)) else 'back'


def recover(rows, decisions, seeds, settings, scores, guard, *, occupied=()):
    """Build one generation of auxiliary claims. No recovered sample becomes an anchor."""
    from .body_identity import quality, independent
    if not seeds:
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
            q['orientation'] = orientation(person)
        signature = np.frombuffer(bytes.fromhex(q['frame_signature']), dtype=np.uint8).astype(float)
        geometry[row['id']] = (q, box, signature)
        by_cue[row['cue']].append(row)
    if not geometry:
        return

    from .perspective_shots import boundaries
    shot_ids, boundary = boundaries(by_cue, geometry)
    # A filtered partner is still an unknown competitor in a crowded transition.
    for cue, members in list(by_cue.items()):
        if any(geometry[r['id']][0].get('detected_people', len(members)) != len(members) for r in members):
            by_cue.pop(cue)

    def transition(a, b):
        qa, ba, sa = geometry[a['id']]
        qb, bb, sb = geometry[b['id']]
        gap = b['seconds'] - a['seconds']
        skipped = b['cue'] - a['cue'] == 2
        if skipped:
            continuous = (shot_ids[a['cue']] == shot_ids[b['cue']] and np.abs(sa-sb).mean() <= 8)
        else:
            continuous = boundary[b['cue']] == 'continuous'
        if not 0 < gap <= 12 or not continuous:
            return None
        wa, ha = ba[2]-ba[0], ba[3]-ba[1]
        wb, hb = bb[2]-bb[0], bb[3]-bb[1]
        if min(wa,ha,wb,hb) <= 0 or max(wa/wb,wb/wa,ha/hb,hb/ha) > 1.5:
            return None
        displacement = math.hypot((ba[0]+ba[2]-bb[0]-bb[2])/2,
                                  (ba[1]+ba[3]-bb[1]-bb[3])/2)
        if displacement > .30 * min(math.hypot(wa,ha), math.hypot(wb,hb)):
            return None
        similarity = scores.face_similarity(a['_vector'],b['_vector'])
        return similarity if similarity >= settings['cohesion_threshold'] else None

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
                score = transition(a,b)
                if score is not None:
                    forward[a['id']].append((score,b['id']))
                    backward[b['id']].append((score,a['id']))
        for a in left:
            edges = sorted(forward[a['id']], reverse=True)
            if not edges:
                continue
            if len(edges)>1 and edges[0][0]-edges[1][0] < margin:
                stops[a['id']] = dict(reason='competing_transition', cue=target, rival=edges[1][1])
                continue
            _, bid = edges[0]
            rivals = sorted(backward[bid], reverse=True)
            if rivals[0][1] == a['id'] and (len(rivals)==1 or rivals[0][0]-rivals[1][0] >= margin):
                next_row[a['id']] = bid
            else:
                stops[a['id']] = dict(reason='competing_transition', cue=target, rival=rivals[0][1])
    lookup = {r['id']: r for r in rows}
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
        while successor and successor not in root:
            member = lookup[successor]
            previous = path[-1] if path else lookup[start]
            skipped += member['cue']-previous['cue']-1
            if skipped > 1 or member['seconds'] - lookup[start]['seconds'] > 60:
                successor = None
                break
            if member.get('face_person') is not None:
                break
            path.append(member)
            successor = next_row.get(successor)
        if not path:
            continue
        last = path[-1]
        if successor in root and (lookup[successor]['seconds'] - lookup[start]['seconds'] > 60
                or skipped + lookup[successor]['cue'] - last['cue'] - 1 > 1):
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
        for r in path:
            source_paths[r['id']] = (start, successor)
            q = geometry[r['id']][0]
            if (q.get('orientation') in ('front','back','side') and not q.get('query_only')
                    and not copresent([r], measured[owner]) and guard.allows(r,seeds[owner],owner)):
                looks[owner, q['orientation']].append((start, successor, r))

    # Keep training routes independent by shot, face roots, pixels and cue.
    banks = {}
    for key, tracks in looks.items():
        from .person_anchor import pixels
        def witness_rank(track):
            row = track[2]
            assessed = geometry[row['id']][0]
            return (-assessed.get('overlap', 0.), assessed.get('keypoints', 0),
                    assessed.get('sharpness', 0.), pixels(row['quality']) or 0, row['id'])
        kept, roots, shots = [], set(), set()
        for start, end, row in sorted(tracks, key=witness_rank, reverse=True):
            shot = shot_ids[row['cue']]
            if start in roots or end in roots or shot in shots:
                continue
            if any(scores.face_similarity(row['_vector'], other['_vector']) < settings['cohesion_threshold']
                   for _, _, other in kept):
                continue
            kept.append((start,end,row))
            roots.update((start,end))
            shots.add(shot)
            if len(kept) >= max(8,settings['attach_min_support']):
                break
        kept.sort(key=lambda item: item[2]['cue'])
        witnesses = [r for _,_,r in kept]
        if len(kept)>=2 and len(independent(witnesses, settings)) >= max(2,settings['attach_min_support']):
            proof = [(start,end,r['id']) for start,end,r in kept]
            banks[key] = witnesses, proof
            for start,end,witness in proof:
                decisions[witness]['rooted_view'] = dict(owner=key[0], perspective=key[1],
                                                         roots=[start,end], tracks=proof)

    for row in rows:
        decision = decisions[row['id']]
        if not row.get('face_person') and decision.get('status') != 'accepted':
            decision.setdefault('track_blocker', stops.get(row['id']) if row['id'] not in next_row
                                and row['id'] in stops else dict(reason='no_independent_view', cue=row['cue']))
    if not banks:
        return
    owner_rows = defaultdict(list)
    for r in rows:
        p = decisions[r['id']]['person']
        if p is not None:
            owner_rows[p].append(r)
    candidates = []
    for row in rows:
        decision = decisions[row['id']]
        if (decision.get('track_blocker', {}).get('reason') in ('different_face_root','competing_transition')
                or row['id'] not in geometry or row.get('face_person') or
                decision['status']=='accepted' and decision.get('reason')!='body_scene_core'):
            continue
        kind = geometry[row['id']][0].get('orientation')
        for (owner, perspective), (bank, proof) in banks.items():
            if decision.get('open_face_root', {}).get('owner', owner) != owner:
                continue
            if perspective != kind or copresent([row],owner_rows[owner]) or not guard.allows(row,bank,owner):
                continue
            # A rooted track has its own temporal proof, but cannot vote for
            # its own appearance. Exact source pixels never add a second vote.
            pixel = geometry[row['id']][0].get('pixel_digest')
            external = [(w, track) for w, track in zip(bank, proof)
                        if track[:2] != source_paths.get(row['id'])
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
            competing.extend((scores.compare([row],other)[0], p) for (p,_),(other,_) in banks.items() if p!=owner)
            runner, rival = max(competing, default=(-1.,None))
            if mean-runner < margin:
                decision['track_blocker'] = dict(reason='competing_owner', cue=row['cue'], rival=rival)
            else:
                candidates.append((row['id'],owner,perspective,[track for _,track in external],mean))
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
        rooted = decisions[oid].get('rooted_view')
        decisions[oid] = dict(person=owner, status='accepted', role='auxiliary',
                              reason='face_rooted_body_view', perspective=angle,
                              source_tracks=proof, score=score)
        if rooted is not None:
            decisions[oid]['rooted_view'] = rooted
