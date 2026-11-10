"""Read frozen local evidence while committing only changed rear memberships."""
from contextlib import contextmanager
import json

from .database import encoded
from .scan_heads import predicate


@contextmanager
def frozen(db, scope, namespace):
    db.execute("""CREATE TEMP TABLE rear_prior AS SELECT m.observation,b.scan,s.scene,
        json_extract(m.proof,'$.previous') AS original,json_extract(m.proof,'$.spanning_source') AS spanning_source FROM main.body_members m
        JOIN body_observations b ON b.id=m.observation JOIN scans s ON s.id=b.scan
        WHERE s.scope=? AND s.namespace=? AND json_extract(m.proof,'$.reason')
        IN ('external_rear_view','external_rear_satellite') AND """+predicate(),(scope,namespace))
    try:
        db.execute('CREATE UNIQUE INDEX temp.rear_prior_id ON rear_prior(observation)')
        db.execute('CREATE INDEX temp.rear_prior_scan ON rear_prior(scan)')
        db.execute("""CREATE TEMP VIEW body_members AS SELECT m.observation,
            CASE WHEN p.observation IS NULL THEN m.person ELSE json_extract(p.original,'$.person') END AS person,
            COALESCE(json_extract(p.original,'$.role'),m.role) AS role,
            COALESCE(json_extract(p.original,'$.status'),m.status) AS status,
            m.policy,COALESCE(p.original,m.proof) AS proof
            FROM main.body_members m LEFT JOIN rear_prior p USING(observation)""")
        try:
            yield dict(db.execute('SELECT DISTINCT scan,scene FROM rear_prior'))
        finally:
            db.execute('DROP VIEW temp.body_members')
    finally:
        db.execute('DROP TABLE temp.rear_prior')


def apply(db, scan, decisions):
    for oid,original in db.execute('SELECT observation,original FROM rear_prior WHERE scan=?',(scan,)):
        if oid not in decisions:
            decisions[oid] = json.loads(original)
    changed = False
    for oid,proof in decisions.items():
        changed |= bool(write(db,oid,proof))
    return changed

def write(db, oid, proof):
    values = proof['person'],proof['role'],proof['status'],encoded(proof)
    return db.execute("""UPDATE main.body_members SET person=?,role=?,status=?,proof=? WHERE observation=?
        AND (person IS NOT ? OR role<>? OR status<>? OR proof<>?)""",(*values,oid,*values)).rowcount
