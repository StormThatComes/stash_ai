"""Candidate scene splits from retained decisions, without decoding crops or embeddings."""
from collections import defaultdict
import json

from .body_identity import available, policy


def pairs(db,scan,config,limit):
    if not available(db):
        return []
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
    def add(oid,other,reason,explicit,peer=None):
        cue,fragment,status,_ = rows[oid]
        if not other or other==fragment or fragment in face_owners and other in face_owners:
            return
        key=tuple(sorted((fragment,other)))
        item=dict(fragments=list(key),kind='face_body' if face_owners.intersection(key) else 'body_body',
                  cue=cue,observation=oid,peer=peer,status=status,reason=reason,recorded=explicit)
        if key not in candidates or explicit and not candidates[key]['recorded']:
            candidates[key] = item
    ordered = sorted(cues)
    previous = dict(zip(ordered[1:],ordered))
    for oid,(cue,fragment,status,proof) in rows.items():
        blocker = proof.get('track_blocker') or {}
        peer = blocker.get('peer') or blocker.get('rival')
        reason = blocker.get('reason') or proof.get('reason')
        if peer in rows:
            add(oid,rows[peer][1],reason,True,peer)
        for other in (proof.get('retry_owner'),proof.get('rival'),(proof.get('open_face_root') or {}).get('owner')):
            if other!='new_participant':
                add(oid,other,reason,True)
        # Adjacent solitary records nominate an inspection, not an asserted cause.
        before = previous.get(cue)
        if len(cues[cue])==1 and before is not None and len(cues[before])==1:
            peer=cues[before][0]
            adjacent = cue==before+1
            add(oid,rows[peer][1],blocker.get('reason') or (
                'unverified_adjacent_fragments' if adjacent else 'cue_gap'),False,peer)
    return sorted(candidates.values(),key=lambda x:(not x['recorded'],x['cue'],x['fragments']))[:limit]
