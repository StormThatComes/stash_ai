"""Candidate scene splits from retained decisions, without decoding crops or embeddings."""
from collections import defaultdict
import json

from .body_identity import available, policy


def ranked(db,scan,config,limit):
    if not available(db):
        return [],[]
    rows = {}
    cues = defaultdict(list)
    face_owners = set()
    for oid,cue,owner,face,status,raw in db.execute("""SELECT b.id,b.cue,m.person,f.person,m.status,m.proof
            FROM body_observations b JOIN body_members m ON m.observation=b.id
            LEFT JOIN members f ON f.observation=b.face
            WHERE b.scan=? AND m.policy=? ORDER BY b.cue,b.id""",(scan,policy(config))):
        proof = json.loads(raw)
        fragment = owner or proof.get('anonymous_fragment') or oid
        rows[oid] = (cue,fragment,status,proof)
        cues[cue].append(oid)
        if face:
            face_owners.add(face)
    for oid,cue,face in db.execute("""SELECT o.id,o.cue,f.person FROM observations o JOIN members f ON f.observation=o.id
            WHERE o.scan=? AND NOT EXISTS (SELECT 1 FROM body_observations b WHERE b.scan=o.scan AND b.face=o.id
                AND b.rejected IS NULL) ORDER BY o.cue,o.id""",(scan,)):
        face_owners.add(face)
        rows[oid] = (cue,face,'accepted',{})
        cues[cue].append(oid)
    candidates = {}
    positives,negatives = defaultdict(set),defaultdict(set)
    opposed = {}
    def oppose(oid,other,reason):
        cue,fragment,_,_ = rows[oid]
        if other and other!=fragment and other!='new_participant':
            key = tuple(sorted((fragment,other)))
            negatives[key].add(cue)
            opposed.setdefault(key,dict(fragments=list(key),observation=oid,cue=cue,reason=reason))
    def add(oid,other,reason,explicit,peer=None):
        cue,fragment,status,_ = rows[oid]
        if not other or other==fragment or fragment in face_owners and other in face_owners:
            return
        key=tuple(sorted((fragment,other)))
        positives[key].add(cue)
        item=dict(fragments=list(key),kind='face_body' if face_owners.intersection(key) else 'body_body',
                  cue=cue,observation=oid,peer=peer,status=status,reason=reason,recorded=explicit)
        if peer in rows:
            peer_cue = rows[peer][0]
            item.update(previous_cue=min(cue, peer_cue), next_cue=max(cue, peer_cue))
        blocker = rows[oid][3].get('track_blocker') or {}
        for field in ('measured', 'limit', 'rival', 'partner_score'):
            if field in blocker:
                item[field] = blocker[field]
        if key not in candidates or explicit and not candidates[key]['recorded']:
            candidates[key] = item
    ordered = sorted(cues)
    previous = dict(zip(ordered[1:],ordered))
    for oid,(cue,fragment,status,proof) in rows.items():
        blocker = proof.get('track_blocker') or {}
        peer = blocker.get('peer')
        rival = blocker.get('rival')
        oppose(oid,rows[rival][1] if rival in rows else rival,'competing_transition')
        oppose(oid,proof.get('rival'),'competing_owner')
        reason = blocker.get('reason') or proof.get('reason')
        if peer in rows:
            if reason in ('different_face_root','competing_transition','unresolved_partner','weak_face_track_conflict'):
                oppose(oid,rows[peer][1],reason)
            else:
                add(oid,rows[peer][1],reason,True,peer)
        for other in (proof.get('retry_owner'),(proof.get('open_face_root') or {}).get('owner')):
            if other!='new_participant':
                add(oid,other,reason,True)
        # Adjacent solitary records nominate an inspection, not an asserted cause.
        before = previous.get(cue)
        if len(cues[cue])==1 and before is not None and len(cues[before])==1:
            peer=cues[before][0]
            adjacent = cue==before+1
            add(oid,rows[peer][1],blocker.get('reason') or (
                'unverified_adjacent_fragments' if adjacent else 'cue_gap'),False,peer)
    # When one owner persists across a two-person transition, only the other
    # trajectory can explain the split. Two simultaneous changes stay ambiguous.
    for cue, before in previous.items():
        if cue != before+1 or len(cues[cue]) != 2 or len(cues[before]) != 2:
            continue
        left = {rows[i][1]: i for i in cues[before]}
        right = {rows[i][1]: i for i in cues[cue]}
        if len(left) == len(right) == 2 and len(left.keys() & right.keys()) == 1:
            source, = left.keys()-right.keys()
            dest, = right.keys()-left.keys()
            query = right[dest]
            reason = rows[query][3].get('track_blocker', {}).get('reason', 'unverified_partner_continuity')
            add(query, source, reason, False, left[source])
    presence = defaultdict(set)
    for cue,owner,_,_ in rows.values():
        presence[owner].add(cue)
    for key,item in candidates.items():
        shared = presence[key[0]] & presence[key[1]]
        if shared:
            negatives[key].update(shared)
            opposed[key] = dict(fragments=list(key),cue=min(shared),reason='co_present')
        item.update(support_cues=len(positives[key]),opposing_cues=len(negatives[key]))
    for key,item in opposed.items():
        item.update(support_cues=len(positives[key]),opposing_cues=len(negatives[key]))
    likely = [v for key,v in candidates.items() if not presence[key[0]].intersection(presence[key[1]])]
    likely.sort(key=lambda v:(v['opposing_cues']-v['support_cues'],not v['recorded'],v['cue'],v['fragments']))
    rivals = sorted(opposed.values(),key=lambda v:(-v['opposing_cues'],v['cue'],v['fragments']))
    return likely[:limit],rivals[:limit]


def pairs(db,scan,config,limit):
    return ranked(db,scan,config,limit)[0]
