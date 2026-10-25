"""Apply frozen hybrid plans to fully selected identities, preserving face authority."""
import hashlib
import json
import time

from .config import digest, plain
from .database import encoded
from .person_signal import available
from .progress import stage, tracking


def signature(db,selected,config):
    """Exact semantic warm stamp; publication-only fields and pixels are excluded.

    Do not trust a former merge merely because its input embeddings still exist:
    changed membership, candidates, input contracts or rivals invalidate it too.
    """
    from .clustering import matching_policy
    first=db.execute('SELECT scope,namespace FROM scans WHERE id=?',(selected[0]['scan'],)).fetchone()
    if first is None:raise ValueError('Hybrid rebuild requires an existing scan.')
    scope,namespace=first
    h=hashlib.sha256(encoded(plain([selected,matching_policy(config),config.data['sapiensid']['execution']])).encode())
    queries=[
        ('scans', 'SELECT id,scene,input_sha,policy_sha,status,cursor,total FROM scans WHERE scope=? AND namespace=? ORDER BY id'),
        ('observations','SELECT o.id,o.cue,o.seconds,o.geometry,o.quality,o.rejected,o.dimensions,o.embedding FROM observations o JOIN scans s ON s.id=o.scan WHERE s.scope=? AND s.namespace=? ORDER BY o.id'),
        ('people','SELECT id,created,token FROM people WHERE scope=? AND namespace=? ORDER BY id'),
        ('members','SELECT m.observation,m.person FROM members m JOIN people p ON p.id=m.person WHERE p.scope=? AND p.namespace=? ORDER BY m.observation'),
        ('candidates','SELECT c.id,c.policy,c.person,c.status,c.payload FROM candidates c JOIN scans s ON s.id=c.scan WHERE s.scope=? AND s.namespace=? ORDER BY c.id'),
        ('aliases','SELECT m.source,m.target,m.evidence FROM identity_merges m JOIN people p ON p.id=m.source WHERE p.scope=? AND p.namespace=? ORDER BY m.source'),
        ('origins','SELECT i.sample_key,i.person,i.observation FROM identity_origins i JOIN people p ON p.id=i.person WHERE p.scope=? AND p.namespace=? ORDER BY i.sample_key')]
    if available(db):queries.append(('person','SELECT e.observation,e.contract,e.status,e.metadata,e.embedding FROM person_evidence e JOIN observations o ON o.id=e.observation JOIN scans s ON s.id=o.scan WHERE s.scope=? AND s.namespace=? ORDER BY e.observation,e.contract'))
    for name,query in queries:
        h.update(name.encode())
        for row in db.execute(query,(scope,namespace)):
            for value in row:
                data=value if isinstance(value,bytes) else encoded(value).encode()
                h.update(len(data).to_bytes(8,'big'));h.update(data)
    return digest([scope,namespace,selected]),h.hexdigest()


def apply(database,selected,config):
    from .hybrid_evidence import report
    from .clustering import matching_policy
    from .publication_queue import enqueue
    db=database.connection;policy=matching_policy(config)
    scope,namespace=db.execute('SELECT scope,namespace FROM scans WHERE id=?',(selected[0]['scan'],)).fetchone()
    result=report(db,config,scope,namespace)
    selection={x['scan'] for x in selected}
    with db:
        with tracking(result['decisions'],'Apply hybrid identity merges',unit='component') as decisions:
            for decision in decisions:
                if decision['outcome']!='merge':continue
                people=decision['people']
                scans={r[0] for r in db.execute('SELECT DISTINCT o.scan FROM members m JOIN observations o ON o.id=m.observation '
                    'WHERE m.person IN (SELECT value FROM json_each(?))',(encoded(people),))}
                if not scans<=selection:
                    decision.update(outcome='blocked',blockers=['identity_outside_selected_scans'])
                    continue
                rows=list(db.execute('SELECT id,created FROM people WHERE id IN (SELECT value FROM json_each(?)) ORDER BY created,id',
                                     (encoded(people),)))
                target=rows[0]['id']
                proof=dict(decision,policy=policy,scan=next(iter(sorted(scans))),route='sapiensid_hybrid',
                           reference_policy='source_attachment_only',original_cores=people)
                # Current publication reconciles scene links, preserving former
                # profiles. This is a local alias, never a pending destructive merge.
                for source in (r['id'] for r in rows[1:]):
                    db.execute('INSERT INTO identity_merges VALUES (?,?,?,?,?)',(source,target,time.time(),encoded(proof),'local'))
                    db.execute('UPDATE members SET person=? WHERE person=?',(target,source))
                    # Every accepted historical decision for this source is
                    # downgraded. No latest-row lookup may promote it silently.
                    db.execute("UPDATE candidates SET payload=json_set(payload,'$.prior_reason',json_extract(payload,'$.reason'),"
                        "'$.reason','hybrid_identity','$.hybrid_merge',json(?),'$.merged_from',person,"
                        "'$.reference_role','attachment','$.reference_selected',json('false')),person=? WHERE person=?",
                        (encoded(proof),target,source))
                for row in db.execute('SELECT id,scene FROM scans WHERE id IN (SELECT value FROM json_each(?))',(encoded(sorted(scans)),)):
                    enqueue(db,scope,namespace,row['scene'],config,'hybrid_identity_merge')
    return result


def rebuild(database,selected,config,ordinary,results):
    if not selected or not config.data['sapiensid']['enabled']:
        return ordinary()
    db=database.connection
    with stage('Validate hybrid rebuild evidence'):
        key,before=signature(db,selected,config)
    record=db.execute('SELECT value FROM metadata WHERE key=?',('hybrid-rebuild:'+key,)).fetchone()
    if record is not None and record[0]==before:
        return results()
    ordinary()
    with stage('Consolidate hybrid identities'):
        apply(database,selected,config)
    with stage('Checkpoint hybrid identity decisions'):
        _,after=signature(db,selected,config)
        with db:db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',('hybrid-rebuild:'+key,after))
    return results()
