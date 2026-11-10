"""Verify every physical member before moving a rear-only owner across scans."""
from collections import defaultdict
import json

from .database import encoded
from .scan_heads import predicate
from .rear_snapshot import write


class SpanningClaims:
    def __init__(self, db):
        self.db = db
        self.nominated = set()
        self.verifying = False

    def stage(self, scan, cue, oid, old, target, proof, rivals):
        if not self.verifying:
            self.nominated.update((old,*rivals))
        elif old in self.nominated:
            self.db.execute('INSERT INTO rear_claims VALUES (?,?,?,?,?,?,?)',
                            (oid,scan,cue,old,target,encoded(sorted(rivals)),encoded(dict(proof,spanning_source=old))))

    def verify(self, search, revisit, scope, namespace, config):
        from .assignment_delta import enqueue
        if not self.nominated:
            return
        db = self.db
        self.verifying = True
        db.execute('CREATE TEMP TABLE rear_claims(observation TEXT PRIMARY KEY,scan TEXT,cue INTEGER,source TEXT,target TEXT,rivals TEXT,proof TEXT)')
        try:
            db.execute('CREATE INDEX temp.rear_claims_source ON rear_claims(source)')
            db.execute('CREATE INDEX temp.rear_claims_target ON rear_claims(target,scan,cue)')
            scans = {scan for owner in self.nominated if owner in search.bank
                     for scan,_ in search.bank[owner]['cues']}
            for scan in sorted(scans):
                revisit(scan)
            # A source is indivisible. Require a fresh positive for every current member,
            # including satellites, and refuse transfers of bound or face-rooted owners.
            owners = encoded(sorted(self.nominated))
            valid = list(db.execute("""SELECT m.person,min(c.target) FROM body_members m
                JOIN body_observations b ON b.id=m.observation JOIN scans s ON s.id=b.scan
                JOIN people p ON p.id=m.person LEFT JOIN rear_claims c ON c.observation=b.id
                WHERE m.person IN (SELECT value FROM json_each(?)) AND p.performer IS NULL
                AND s.scope=? AND s.namespace=? AND s.status='complete' AND """+predicate()+"""
                GROUP BY m.person HAVING count(*)=count(c.target) AND count(DISTINCT c.target)=1
                """,(owners,scope,namespace)))
            accepted = {source for source,target in valid if target not in self.nominated}
            destinations = dict(valid)
            dependencies = defaultdict(set)
            for source,rival in db.execute("SELECT source,j.value FROM rear_claims,json_each(rivals) j"):
                dependencies[source].add(rival)
            # Simultaneous independent sources cannot share a destination, even
            # when their claims arrived through different search passes.
            collisions = db.execute("""SELECT c.source FROM rear_claims c JOIN
                (SELECT target,scan,cue FROM rear_claims GROUP BY target,scan,cue HAVING count(*)>1) occupied
                USING(target,scan,cue)""")
            accepted.difference_update(source for source, in collisions)
            dependents = defaultdict(set)
            blocked = set()
            for source,rivals in dependencies.items():
                for rival in rivals:
                    if rival not in accepted or destinations.get(rival)!=destinations.get(source):
                        blocked.add(source)
                    dependents[rival].add(source)
            while blocked:
                source = blocked.pop()
                if source in accepted:
                    accepted.remove(source)
                    blocked.update(dependents[source])
            dirty = set()
            for oid,scan,source,raw,rivals in db.execute('SELECT observation,scan,source,proof,rivals FROM rear_claims'):
                if source in accepted:
                    proof = json.loads(raw)
                    if write(db,oid,proof):
                        dirty.add(scan)
                elif json.loads(rivals):
                    prior = json.loads(raw)['previous']
                    prior['rear_retry'] = dict(reason='competing_owner',rival=json.loads(rivals)[0],
                                              references=search.rear_revision)
                    if write(db,oid,prior):
                        dirty.add(scan)
            # Revoke the entire former assignment when even one member loses support.
            for oid,scan,raw in db.execute("""SELECT p.observation,p.scan,p.original FROM rear_prior p
                    JOIN main.body_members m ON m.observation=p.observation
                    WHERE json_extract(m.proof,'$.spanning_source') IN (SELECT value FROM json_each(?))""",
                    (encoded(sorted(self.nominated-accepted)),)):
                if write(db,oid,json.loads(raw)):
                    dirty.add(scan)
            for scan in sorted(dirty):
                scene = db.execute('SELECT scene FROM scans WHERE id=?',(scan,)).fetchone()[0]
                enqueue(db,scope,namespace,dict(scan=scan,scene=scene),config,True,'rear_view_attachment')
        finally:
            db.execute('DROP TABLE temp.rear_claims')
