"""Frozen, face-rooted rear appearances from current scans only."""
from collections import defaultdict
import json
import numpy as np

from .body_identity import policy, vector
from .body_observations import contract
from .pose_policy import anchor_sql
from .reference_roles import strict_sql
from .resource_errors import ResourceLimit
from .scan_heads import predicate


def load(db, config, scope, namespace):
    # Read proof addresses first. No general auxiliary membership trains this bank.
    sql = """WITH proofs AS MATERIALIZED (
        SELECT b.id,b.scan,m.person,json_extract(m.proof,'$.reason') AS reason,
            COALESCE(json_extract(m.proof,'$.rooted_view'),json_extract(m.proof,'$.rooted_track')) AS claim
        FROM body_members m JOIN body_observations b ON b.id=m.observation
        JOIN scans s ON s.id=b.scan WHERE s.scope=? AND s.namespace=? AND s.status='complete'
        AND b.contract=? AND b.rejected IS NULL AND m.policy=? AND """ + predicate() + """
        AND COALESCE(json_extract(m.proof,'$.rooted_view.perspective'),
                     json_extract(m.proof,'$.rooted_track.perspective'))='back'),
        certified AS MATERIALIZED (
        SELECT p.id,json_extract(p.claim,'$.owner') AS owner,json_extract(p.claim,'$.roots') AS roots
        FROM proofs p JOIN json_each(p.claim,'$.roots') root
        JOIN body_observations rb ON rb.id=root.value AND rb.scan=p.scan
        JOIN observations o ON o.id=rb.face JOIN members m ON m.observation=o.id
        WHERE p.reason NOT IN ('body_copresence_conflict','different_face_root')
        AND (p.person IS NULL OR p.person=m.person OR p.reason='body_scene_core')
        AND m.person=json_extract(p.claim,'$.owner') AND o.rejected IS NULL
        AND """ + strict_sql() + ' AND ' + anchor_sql('o',config.data['clustering']) + """
        GROUP BY p.id HAVING count(DISTINCT rb.id)=2)
        SELECT b.id,b.scan,b.cue,b.seconds,b.quality,b.embedding,c.owner,c.roots
        FROM certified c JOIN body_observations b ON b.id=c.id ORDER BY b.scan,b.cue,b.id"""
    settings = config.data['sapiensid']['identities']
    limit = config.data['sapiensid']['execution']['reference_bytes'] // 8
    groups = defaultdict(list)
    used = 0
    for raw in db.execute(sql,(scope,namespace,contract(config),policy(config))):
        row = dict(raw)
        owner = row.pop('owner')
        roots = json.loads(row.pop('roots'))
        used += 5*len(row['embedding']) + len(row['quality']) + 2048
        if used>limit:
            raise ResourceLimit('Rear-view gallery exceeds sapiensid.execution.reference_bytes.')
        row['_vector'] = vector(row)
        del row['embedding']
        row['roots'] = roots
        groups[owner,row['scan']].append(row)
    banks = []
    from .body_identity import Scores
    from .rooted_witnesses import modes
    scores = Scores(config)

    def rank(item):
        q = json.loads(item[2]['quality']).get('quality',{})
        return (-q.get('overlap',0),q.get('keypoints',0),q.get('sharpness',0),item[2]['id'])

    for (owner,scan), rows in groups.items():
        tracks = [(*row['roots'],row) for row in rows]
        def local_shot(row):
            shot = json.loads(row['quality']).get('quality',{}).get('view_shot')
            return tuple(row['roots']) if shot is None else shot
        for kept in modes(tracks,settings,scores,rank,local_shot):
            if len(kept)>=max(2,settings['attach_min_support']):
                banks.append((owner,scan,[r for _,_,r in kept]))
    by_owner = defaultdict(list)
    for (owner,scan), rows in groups.items():
        for row in rows:
            by_owner[owner].append((*row['roots'],row))
    for owner, tracks in sorted(by_owner.items()):
        if len({row['scan'] for _,_,row in tracks}) < 2:
            continue
        for kept in modes(tracks,settings,scores,rank,lambda row:row['scan']):
            if len(kept) >= max(2,settings['attach_min_support']):
                banks.append((owner,kept[0][2]['scan'],[row for _,_,row in kept]))
    return Gallery(banks)


def revision(gallery):
    import hashlib
    from .database import encoded
    stamp = hashlib.sha256()
    for owner, source, rows in sorted(gallery.banks, key=lambda bank: bank[:2]):
        stamp.update(encoded([owner,source]).encode())
        for row in rows:
            stamp.update(encoded([row['id'],row['roots'],row['quality']]).encode())
            stamp.update(row['_vector'].tobytes())
    return stamp.hexdigest()


def winner(ranked, settings):
    for candidate in ranked:
        mean, minimum, owner = candidate[:3]
        if mean < settings['match_threshold'] or minimum < settings['cohesion_threshold']:
            continue
        rival = max((max(r[0],r[5]) for r in ranked if r[2]!=owner), default=-1.)
        if mean-rival >= settings['ambiguity_margin']:
            return candidate
    return None


class Gallery:
    def __init__(self,banks):
        from scipy.spatial import cKDTree
        self.banks = banks
        self.addresses = [index for index,(_,_,rows) in enumerate(banks) for row in rows]
        self.tree = cKDTree([row['_vector'] for _,_,rows in banks for row in rows]) if banks else None

    def rank(self, query, scan, settings, pixel):
        # A mean above the floor requires at least one witness above the floor.
        floor = settings['match_threshold'] - settings['ambiguity_margin']
        nearby = self.tree.query_ball_point(query, np.sqrt(2*(1-floor)) + .002)
        candidates = sorted({self.addresses[index] for index in nearby})
        result = []
        for index in candidates:
            owner,source,rows = self.banks[index]
            rows = [row for row in rows if row.get('scan',source)!=scan]
            if not rows:
                continue
            source = rows[0].get('scan',source)
            scored = [(float(np.dot(query,row['_vector'])),row) for row in rows]
            rival_score = sum(score for score,_ in scored)/len(scored)
            independent = [(score,row) for score,row in scored
                           if not pixel or json.loads(row['quality']).get('quality',{}).get('pixel_digest') != pixel]
            if len(independent)>=max(2,settings['attach_min_support']):
                scores, witnesses = zip(*independent)
                result.append((sum(scores)/len(scores), min(scores), owner, source, witnesses, rival_score))
            else:
                # Losing independent support cannot remove a plausible rival.
                result.append((-1., -1., owner, source, (), rival_score))
        return sorted(result,key=lambda r:(-r[0],-r[1],r[2],r[3]))
