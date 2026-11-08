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


def load(db, config, scope, namespace, *, pages=None):
    # Read proof addresses first. No general auxiliary membership trains this bank.
    sql = """WITH proofs AS MATERIALIZED (
        SELECT b.id,b.scan,m.person,json_extract(m.proof,'$.reason') AS reason,
            COALESCE(json_extract(m.proof,'$.rooted_view'),json_extract(m.proof,'$.rooted_track')) AS claim
        FROM body_members m JOIN body_observations b ON b.id=m.observation
        JOIN scans s ON s.id=b.scan WHERE s.scope=? AND s.namespace=? AND s.status='complete'
        AND b.contract=? AND b.rejected IS NULL AND m.policy=? AND """ + predicate() + """
        AND COALESCE(json_extract(m.proof,'$.rooted_view.perspective'),
                     json_extract(m.proof,'$.rooted_track.perspective')) IN ('back','side')),
        certified AS MATERIALIZED (
        SELECT p.id,json_extract(p.claim,'$.owner') AS owner,json_extract(p.claim,'$.roots') AS roots,
            json_extract(p.claim,'$.perspective') AS perspective
        FROM proofs p JOIN json_each(p.claim,'$.roots') root
        JOIN body_observations rb ON rb.id=root.value AND rb.scan=p.scan
        JOIN observations o ON o.id=rb.face JOIN members m ON m.observation=o.id
        WHERE COALESCE(p.reason,'') NOT IN ('body_copresence_conflict','different_face_root')
        AND (p.person IS NULL OR p.person=m.person OR p.reason='body_scene_core')
        AND m.person=json_extract(p.claim,'$.owner') AND o.rejected IS NULL
        AND """ + strict_sql() + ' AND ' + anchor_sql('o',config.data['clustering']) + """
        GROUP BY p.id HAVING count(DISTINCT rb.id)=2)
        SELECT b.id,b.scan,b.cue,b.seconds,b.quality,c.owner,c.roots,c.perspective
        FROM certified c JOIN body_observations b ON b.id=c.id ORDER BY c.owner,c.perspective,b.scan,b.cue,b.id"""
    settings = config.data['sapiensid']['identities']
    limit = config.data['sapiensid']['execution']['reference_bytes'] // 8
    from functools import lru_cache
    from itertools import groupby
    from .body_identity import Scores
    from .rooted_witnesses import modes
    # Witness vectors live in the immutable observation table. A small read
    # cache is shared by local and multi-scan selection for this generation.
    @lru_cache(maxsize=max(1,min(128,limit//(8*16384))))
    def feature(oid):
        raw = db.execute('SELECT embedding FROM body_observations WHERE id=?',(oid,)).fetchone()
        return vector(dict(embedding=raw[0]))

    class Witness(dict):
        def __getitem__(self, key):
            if key=='_vector':
                return feature(self['id'])
            return super().__getitem__(key)

    scores = Scores(config)
    def rank(item):
        q = json.loads(item[2]['quality']).get('quality',{})
        return (-q.get('overlap',0),q.get('keypoints',0),q.get('sharpness',0),item[2]['id'])

    def local_shot(row):
        shot = json.loads(row['quality']).get('quality',{}).get('view_shot')
        return tuple(row['roots']) if shot is None else shot

    def banks():
        cursor = db.execute(sql,(scope,namespace,contract(config),policy(config)))
        for (owner,kind), records in groupby(cursor,lambda raw:(raw['owner'],raw['perspective'])):
            if pages is not None:
                from .paged_tracks import Tracks
                tracks = Tracks(pages, Witness)
                pages.execute('DELETE FROM witnesses')
                multiple_scans = False
                for scan,group in groupby(records,lambda raw:raw['scan']):
                    for raw in group:
                        row = Witness(raw)
                        row['roots'] = json.loads(row['roots'])
                        tracks.add(row, local_shot(row))
                    for kept in tracks.modes(scan,settings,scores,rank,local_shot):
                        if len(kept)>=max(2,settings['attach_min_support']):
                            yield owner,scan,[r for _,_,r in kept]
                    if not multiple_scans:
                        multiple_scans = pages.execute('SELECT 1 FROM witnesses WHERE scan<>? LIMIT 1',(scan,)).fetchone() is not None
                if multiple_scans:
                    for kept in tracks.modes(None,settings,scores,rank,lambda row:row['scan']):
                        if len(kept)>=max(2,settings['attach_min_support']):
                            yield owner,kept[0][2]['scan'],[r for _,_,r in kept]
                continue
            tracks = []
            cost = 0
            for _,group in groupby(records,lambda row:row['scan']):
                local = []
                for raw in group:
                    row = Witness(raw)
                    row['roots'] = json.loads(row['roots'])
                    cost += len(row['quality'])+1024
                    if cost>limit:
                        raise ResourceLimit('Perspective witness metadata for one owner exceeds sapiensid.execution.reference_bytes.')
                    local.append((*row['roots'],row))
                tracks.extend(local)
                for kept in modes(local,settings,scores,rank,local_shot):
                    if len(kept)>=max(2,settings['attach_min_support']):
                        yield owner,kept[0][2]['scan'],[r for _,_,r in kept]
            if tracks and tracks[0][2]['scan']!=tracks[-1][2]['scan']:
                for kept in modes(tracks,settings,scores,rank,lambda row:row['scan']):
                    if len(kept)>=max(2,settings['attach_min_support']):
                        yield owner,kept[0][2]['scan'],[r for _,_,r in kept]
    if pages is not None:
        from .perspective_pages import DiskGallery
        return DiskGallery(pages,banks(),limit)
    retained = []
    used = 0
    for owner,source,rows in banks():
        values = [dict(row,_vector=row['_vector']) for row in rows]
        used += sum(5*r['_vector'].nbytes+len(r['quality'])+2048 for r in values)
        if used>limit:
            raise ResourceLimit('Perspective gallery exceeds sapiensid.execution.reference_bytes in memory mode.')
        retained.append((owner,source,values))
    return Gallery(retained)


def revision(gallery):
    import hashlib
    from .database import encoded
    if hasattr(gallery,'versions'):
        return hashlib.sha256(encoded(gallery.versions).encode()).hexdigest()
    stamp = hashlib.sha256()
    for owner, source, rows in sorted(gallery.banks, key=lambda bank: bank[:2]):
        stamp.update(encoded([owner,source]).encode())
        for row in rows:
            stamp.update(encoded([row['id'],row['roots'],row['quality'],row.get('perspective','back')]).encode())
            stamp.update(row['_vector'].tobytes())
    return stamp.hexdigest()


def winner(ranked, settings, *, contextual=(), bonus=0.):
    for candidate in ranked:
        mean, minimum, owner = candidate[:3]
        threshold = settings['match_threshold'] - (bonus if owner in contextual else 0.)
        if mean < threshold or minimum < settings['cohesion_threshold']:
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

    def owner_at(self,index):
        return self.banks[index][0]

    def nearby(self,query,radius):
        return {self.addresses[i] for i in self.tree.query_ball_point(query,radius)}

    def rank(self, query, scan, settings, pixel, perspective='back'):
        # A mean above the floor requires at least one witness above the floor.
        floor = settings['match_threshold'] - settings['ambiguity_margin']
        candidates = sorted(self.nearby(query,np.sqrt(2*(1-floor)) + .002))
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
            if rows[0].get('perspective','back')==perspective and len(independent)>=max(2,settings['attach_min_support']):
                scores, witnesses = zip(*independent)
                source = witnesses[0].get('scan',source)
                proof = tuple({k:w[k] for k in ('id','scan','roots') if k in w} for w in witnesses)
                result.append((sum(scores)/len(scores), min(scores), owner, source, proof, rival_score))
            else:
                # Losing independent support cannot remove a plausible rival.
                result.append((-1., -1., owner, source, (), rival_score))
        return sorted(result,key=lambda r:(-r[0],-r[1],r[2],r[3]))
