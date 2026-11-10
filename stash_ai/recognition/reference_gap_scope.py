"""Localise a missing reference only with independent, current physical exclusions."""
from collections import defaultdict
from functools import lru_cache
import json

from .database import encoded
from .scan_heads import predicate
from .identity_merges import canonical
from .crowd_context import pair


class MissingReferences:
    def __init__(self, db, scope, namespace, pending, *, incompatible=None, exclude_scans=()):
        self.db = db
        self.rivals = defaultdict(set)
        self.unknown = set()
        table = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='body_members'").fetchone()
        if not table:
            self.unknown.update(pending)
            return
        params = (scope,namespace,encoded(pending))
        clause = "s.scope=? AND s.namespace=? AND s.scene IN (SELECT value FROM json_each(?))"
        body = ("SELECT s.scene,m.person FROM body_observations b JOIN body_members m ON m.observation=b.id "
                "JOIN scans s ON s.id=b.scan WHERE b.rejected IS NULL AND "+clause)
        if incompatible is not None:
            body += (" AND m.role='anchor' AND m.status IN ('accepted','provisional') AND "+predicate()+
                     " AND b.contract NOT IN (SELECT value FROM json_each(?))")
            entries = db.execute(body,(*params,encoded(incompatible)))
        else:
            face = ("SELECT s.scene,m.person FROM observations o JOIN members m ON m.observation=o.id "
                    "JOIN scans s ON s.id=o.scan WHERE o.rejected IS NULL AND "+clause)
            entries = db.execute(body+' UNION '+face,(*params,*params))
        for scene,owner in entries:
            if owner is None:
                self.unknown.add(scene)
            else:
                self.rivals[scene].add(canonical(db,owner))
        self.unknown.update(scene for scene in pending if not self.rivals[scene])
        people = {p for group in self.rivals.values() for p in group}
        if not people:
            return
        # Only corroborated face and body anchors of two distinct people may
        # certify an exclusion. A pending scene cannot certify itself.
        self.excluded = set(pending)
        self.excluded_scans = set(exclude_scans)
        self.namespace = (scope,namespace)
        self.peers = lru_cache(maxsize=128)(self._peers)

    def _peers(self, person):
        from .reference_roles import strict_sql
        peers = set()
        sql = ("SELECT a.scan,s.scene,a.geometry,b.geometry,o.geometry,z.geometry,n.person FROM body_members m "
               "JOIN body_observations a ON a.id=m.observation JOIN scans s ON s.id=a.scan "
               "JOIN observations o ON o.id=a.face JOIN members f ON f.observation=o.id "
               "JOIN body_observations b ON b.scan=a.scan AND b.cue=a.cue AND b.id<>a.id "
               "JOIN body_members n ON n.observation=b.id JOIN observations z ON z.id=b.face "
               "JOIN members g ON g.observation=z.id WHERE m.person=? AND n.person<>m.person "
               "AND f.person=m.person AND g.person=n.person AND m.status='accepted' AND n.status='accepted' "
               "AND m.role='anchor' AND n.role='anchor' AND a.rejected IS NULL AND b.rejected IS NULL "
               "AND o.rejected IS NULL AND z.rejected IS NULL AND s.scope=? AND s.namespace=? "
               "AND s.status='complete' AND "+strict_sql('o','f')+" AND "+strict_sql('z','g')+" AND "+predicate())
        for scan,scene,torso,other_torso,face,other_face,other in self.db.execute(sql,(person,*self.namespace)):
            if scene in self.excluded or scan in self.excluded_scans or other in peers:
                continue
            rectangles = [json.loads(raw)[:4] for raw in (face,other_face)]
            if any(len(rect)!=4 or not all(isinstance(v,(int,float)) for v in rect) for rect in rectangles):
                continue
            x,y,w,h = rectangles[0]
            u,v,s,t = rectangles[1]
            if min(w,h,s,t)>0 and (x+w<=u or u+s<=x or y+h<=v or v+t<=y):
                bodies = [json.loads(torso),json.loads(other_torso)]
                if all(len(body.get('bounds',()))==4 for body in bodies) and pair(bodies):
                    peers.add(canonical(self.db,other))
        return peers

    def affected(self, owners):
        people = {canonical(self.db,p) for p in owners}
        return sorted(self.unknown | {scene for scene,rivals in self.rivals.items()
            if not people or any(p in people or not people.issubset(self.peers(p)) for p in rivals)},key=str)

    def protect(self, result, reason):
        for decision in result['decisions']:
            if decision['outcome']=='merge':
                affected = self.affected(decision['people'])
                if affected:
                    decision.update(outcome='blocked',blockers=[reason],required_scenes=affected)
