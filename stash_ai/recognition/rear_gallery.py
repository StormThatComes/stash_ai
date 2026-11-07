"""Frozen, face-rooted rear appearances from current scans only."""
from collections import defaultdict
import json
import numpy as np

from .body_identity import policy, vector, independent
from .body_observations import contract
from .pose_policy import anchor_sql
from .reference_roles import strict_sql
from .resource_errors import ResourceLimit
from .scan_heads import predicate


def load(db, config, scope, namespace):
    # Read proof addresses first. No general auxiliary membership trains this bank.
    sql = """SELECT b.id,b.scan,b.cue,b.seconds,b.quality,b.embedding,m.proof,m.person
        FROM body_members m JOIN body_observations b ON b.id=m.observation
        JOIN scans s ON s.id=b.scan WHERE s.scope=? AND s.namespace=? AND s.status='complete'
        AND b.contract=? AND b.rejected IS NULL AND m.policy=?
        AND json_extract(m.proof,'$.rooted_view.perspective')='back' AND """ + predicate()
    settings = config.data['sapiensid']['identities']
    limit = config.data['sapiensid']['execution']['reference_bytes'] // 8
    groups = defaultdict(list)
    used = 0
    for raw in db.execute(sql,(scope,namespace,contract(config),policy(config))):
        row = dict(raw)
        proof = json.loads(row.pop('proof'))['rooted_view']
        owner = proof['owner']
        if row['person'] not in (None,owner):
            continue
        roots = proof['roots']
        # The strict face at each endpoint must still belong to this owner.
        valid = db.execute("""SELECT count(*) FROM body_observations b JOIN observations o ON o.id=b.face
            JOIN members m ON m.observation=o.id WHERE b.id IN (?,?) AND b.scan=?
            AND m.person=? AND o.rejected IS NULL AND """+strict_sql()+' AND '+anchor_sql('o',config.data['clustering']),
            (*roots,row['scan'],owner)).fetchone()[0]
        if valid!=2:
            continue
        used += 5*len(row['embedding']) + len(row['quality']) + 2048
        if used>limit:
            raise ResourceLimit('Rear-view gallery exceeds sapiensid.execution.reference_bytes.')
        row['_vector'] = vector(row)
        del row['embedding']
        row['roots'] = roots
        groups[owner,row['scan']].append(row)
    banks = []
    for (owner,scan), rows in groups.items():
        witnesses = independent(rows,settings)
        if len(witnesses)>=max(2,settings['attach_min_support']):
            banks.append((owner,scan,witnesses))
    return Gallery(banks)


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
            if source == scan:
                continue
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
