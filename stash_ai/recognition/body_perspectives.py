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
    if any(min(points[i]) < 0 for i in (5, 6, 11, 12)):
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
    return 'unknown' if all(min(points[i]) >= 0 for i in (1,2)) else 'back'


def recover(rows, decisions, seeds, settings, scores, guard):
    """Build one generation of auxiliary claims. No recovered sample becomes an anchor."""
    from .body_identity import quality, independent
    if not seeds:
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
        signature = np.frombuffer(bytes.fromhex(q['frame_signature']), dtype=np.uint8).astype(float)
        geometry[row['id']] = (q, box, signature)
        by_cue[row['cue']].append(row)
    if not geometry:
        return

    def transition(a, b):
        qa, ba, sa = geometry[a['id']]
        qb, bb, sb = geometry[b['id']]
        gap = b['seconds'] - a['seconds']
        if not 0 < gap <= 12 or sa.shape != sb.shape or np.abs(sa-sb).mean() > 20:
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
    next_row = {}
    margin = max(.025, settings['ambiguity_margin'])
    for cue, left in by_cue.items():
        right = by_cue.get(cue+1, ())
        forward, backward = defaultdict(list), defaultdict(list)
        for a in left:
            for b in right:
                score = transition(a,b)
                if score is not None:
                    forward[a['id']].append((score,b['id']))
                    backward[b['id']].append((score,a['id']))
        for a in left:
            edges = sorted(forward[a['id']], reverse=True)
            if not edges or len(edges)>1 and edges[0][0]-edges[1][0] < margin:
                continue
            _, bid = edges[0]
            rivals = sorted(backward[bid], reverse=True)
            if rivals[0][1] == a['id'] and (len(rivals)==1 or rivals[0][0]-rivals[1][0] >= margin):
                next_row[a['id']] = bid
    lookup = {r['id']: r for r in rows}
    looks = defaultdict(list)
    for start, owner in root.items():
        path = []
        successor = next_row.get(start)
        while successor and successor not in root and len(path) < 4:
            member = lookup[successor]
            if member.get('face_person') is not None:
                break
            path.append(member)
            successor = next_row.get(successor)
        if root.get(successor) != owner or not path:
            continue
        for r in path:
            q = geometry[r['id']][0]
            if q.get('orientation') in ('front','back','side') and not q.get('query_only'):
                looks[owner, q['orientation']].append((start, successor, r))

    # Keep training routes independent by shot, face roots, pixels and cue.
    banks = {}
    for key, tracks in looks.items():
        kept, roots, shots = [], set(), []
        for start, end, row in tracks:
            signature = geometry[start][2]
            if start in roots or end in roots or any(np.abs(signature-s).mean() < 12 for s in shots):
                continue
            if kept and scores.compare([row], [r for _,_,r in kept])[1] < settings['cohesion_threshold']:
                continue
            kept.append((start,end,row))
            roots.update((start,end))
            shots.append(signature)
        witnesses = [r for _,_,r in kept]
        if len(kept)>=2 and len(independent(witnesses, settings)) >= max(2,settings['attach_min_support']):
            banks[key] = (witnesses, [(start,end,r['id']) for start,end,r in kept])

    if not banks:
        return
    owner_rows = defaultdict(list)
    for r in rows:
        p = decisions[r['id']]['person']
        if p is not None:
            owner_rows[p].append(r)
    from .body_identity import copresent
    candidates = []
    for row in rows:
        decision = decisions[row['id']]
        if (row['id'] not in geometry or row.get('face_person') or
                decision['status']=='accepted' and decision.get('reason')!='body_scene_core'):
            continue
        kind = geometry[row['id']][0].get('orientation')
        for (owner, perspective), (bank, proof) in banks.items():
            if perspective != kind or copresent([row],owner_rows[owner]) or not guard.allows(row,bank,owner):
                continue
            mean, low = scores.compare([row],bank)
            if mean < settings['match_threshold'] or low < settings['cohesion_threshold']:
                continue
            # Every foreign owner including provisional scene groups competes.
            runner = max((scores.compare([row],others)[0] for p, others in owner_rows.items()
                          if p not in (owner,decision['person']) and not copresent([row],others)
                          and guard.allows(row,others)),default=-1.)
            rivals = [scores.compare([row], other)[0] for (p,_), (other,_) in banks.items() if p!=owner]
            if mean-max([runner,*rivals]) >= margin:
                candidates.append((row['id'],owner,perspective,proof,mean))
    proposed = {oid: owner for oid, owner, *_ in candidates}
    # An established body fragment moves as a unit or remains available as a rival.
    protected = {p for p, members in owner_rows.items()
                 if any(decisions[r['id']]['reason']=='body_scene_core' for r in members)
                 and (any(r['id'] not in proposed for r in members)
                      or len({proposed.get(r['id']) for r in members}) != 1)}
    for oid,owner,angle,proof,score in candidates:
        if decisions[oid]['person'] in protected:
            continue
        decisions[oid] = dict(person=owner, status='accepted', role='auxiliary',
                              reason='face_rooted_body_view', perspective=angle,
                              source_tracks=proof, score=score)
