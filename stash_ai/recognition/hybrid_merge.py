"""Apply frozen hybrid plans to fully selected identities, preserving face authority."""
import hashlib
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
    h=hashlib.sha256(encoded(plain([selected,matching_policy(config),config.data['sapiensid']['execution'],
        config.data['clustering'],config.data['execution']])).encode())
    queries=[
        ('scans', 'SELECT rowid,id,scene,input_sha,policy_sha,status,cursor,total FROM scans WHERE scope=? AND namespace=? ORDER BY id'),
        ('observations','SELECT o.rowid,o.id,o.scan,o.cue,o.seconds,o.source,o.rect,o.geometry,o.quality,o.rejected,o.dimensions,o.embedding FROM observations o JOIN scans s ON s.id=o.scan WHERE s.scope=? AND s.namespace=? ORDER BY o.id'),
        ('people','SELECT id,created,token FROM people WHERE scope=? AND namespace=? ORDER BY id'),
        ('members','SELECT m.observation,m.person FROM members m JOIN people p ON p.id=m.person WHERE p.scope=? AND p.namespace=? ORDER BY m.observation'),
        ('candidates','SELECT c.rowid,c.id,c.scan,c.policy,c.person,c.status,c.payload FROM candidates c JOIN scans s ON s.id=c.scan WHERE s.scope=? AND s.namespace=? ORDER BY c.id'),
        ('aliases','SELECT m.source,m.target,m.evidence FROM identity_merges m JOIN people p ON p.id=m.source WHERE p.scope=? AND p.namespace=? ORDER BY m.source'),
        ('origins','SELECT i.sample_key,i.person,i.observation FROM identity_origins i JOIN people p ON p.id=i.person WHERE p.scope=? AND p.namespace=? ORDER BY i.sample_key')]
    if available(db):queries.append(('person','SELECT e.observation,e.contract,e.status,e.metadata,e.embedding FROM person_evidence e JOIN observations o ON o.id=e.observation JOIN scans s ON s.id=o.scan WHERE s.scope=? AND s.namespace=? ORDER BY e.observation,e.contract'))
    from .dependencies import cached, supported
    from .dependency_schema import validate_triggers, EXTRA_TRIGGERS
    installed={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
    cacheable=supported(db) and set(EXTRA_TRIGGERS)<=installed
    if cacheable:validate_triggers(db,required=True)
    def rows_digest(query,args):
        out=hashlib.sha256()
        for row in db.execute(query,args):
            for value in row:
                data=value if isinstance(value,bytes) else encoded(value).encode()
                out.update(len(data).to_bytes(8,'big'));out.update(data)
        return out.hexdigest()
    scans=[r[0] for r in db.execute('SELECT id FROM scans WHERE scope=? AND namespace=? ORDER BY id',(scope,namespace))]
    for name,query in queries:
        h.update(name.encode())
        if name in ('observations','person'):
            before,order=query.rsplit(' ORDER BY ',1)
            for scan in scans:
                compute=lambda:rows_digest(before+' AND o.scan=? ORDER BY '+order,(scope,namespace,scan))
                value=(cached(db,'hybrid-'+name,scan,'v2',compute,dependency='scan',dependency_key=scan,checked=True)
                       if cacheable else compute())
                h.update(encoded([scan,value]).encode())
        else:
            compute=lambda:rows_digest(query,(scope,namespace))
            value=(cached(db,'hybrid-'+name,digest([scope,namespace]),'v2',compute,checked=True)
                   if cacheable else compute())
            h.update(value.encode())
    return digest([scope,namespace,selected]),h.hexdigest()


def apply(database,selected,config):
    from .hybrid_evidence import report
    from .clustering import matching_policy
    from .publication_queue import enqueue
    db=database.connection;policy=matching_policy(config)
    scope,namespace=db.execute('SELECT scope,namespace FROM scans WHERE id=?',(selected[0]['scan'],)).fetchone()
    result=report(db,config,scope,namespace)
    selection={x['scan'] for x in selected}
    stored_bytes=0; pending=[]
    queue_key="hybrid-pending:"+digest([scope,namespace])
    with db:
        with tracking(result['decisions'],'Apply hybrid identity merges',unit='component') as decisions:
            for decision in decisions:
                if decision['outcome']!='merge':continue
                people=decision['people']
                scans={r[0] for r in db.execute('SELECT DISTINCT o.scan FROM members m JOIN observations o ON o.id=m.observation '
                    'WHERE m.person IN (SELECT value FROM json_each(?))',(encoded(people),))}
                if not scans<=selection:
                    pending.append(dict(people=people,required_scans=sorted(scans),
                        missing_scans=sorted(scans-selection),contract=decision['contract']))
                    decision.update(outcome='blocked',blockers=['identity_outside_selected_scans'])
                    continue
                rows=list(db.execute('SELECT id,created FROM people WHERE id IN (SELECT value FROM json_each(?)) ORDER BY created,id',
                                     (encoded(people),)))
                target=rows[0]['id']
                proof=dict(decision,policy=policy,scan=next(iter(sorted(scans))),route='sapiensid_hybrid',
                           reference_policy='source_attachment_only',original_cores=people)
                stored_bytes+=len(encoded(proof).encode())*(len(rows)-1)
                if stored_bytes>config.data['sapiensid']['execution']['plan_bytes']:
                    raise ValueError('SapiensID plan_bytes budget exceeded by alias evidence; transaction rolled back.')
                # Current publication reconciles scene links, preserving former
                # profiles. This is a local alias, never a pending destructive merge.
                for source in (r['id'] for r in rows[1:]):
                    db.execute('INSERT INTO identity_merges VALUES (?,?,?,?,?)',(source,target,time.time(),encoded(proof),'local'))
                    db.execute('UPDATE members SET person=? WHERE person=?',(target,source))
                    # Every accepted historical decision for this source is
                    # downgraded. No latest-row lookup may promote it silently.
                    recorded=dict(route='sapiensid_hybrid',source=source,target=target,
                        contract=proof['contract'],reference_policy=proof['reference_policy'],
                        proof_location='identity_merges',nearest_rival=proof['nearest_rival'])
                    db.execute("UPDATE candidates SET payload=json_set(payload,'$.prior_reason',json_extract(payload,'$.reason'),"
                        "'$.reason','hybrid_identity','$.hybrid_merge',json(?),'$.merged_from',person,"
                        "'$.reference_role','attachment','$.reference_selected',json('false')),person=? WHERE person=?",
                        (encoded(recorded),target,source))
                for row in db.execute('SELECT id,scene FROM scans WHERE id IN (SELECT value FROM json_each(?))',(encoded(sorted(scans)),)):
                    enqueue(db,scope,namespace,row['scene'],config,'hybrid_identity_merge')
        # A journal entry is a request for revalidation, never an executable
        # stale plan. Subsequent runs automatically recompute eligible decisions
        # once their selected scope includes every required scan.
        payload=encoded(dict(format=1,scope=scope,namespace=namespace,pending=pending))
        if len(payload.encode())>config.data['sapiensid']['execution']['plan_bytes']:
            raise ValueError('SapiensID pending consolidation plan_bytes budget exceeded.')
        if pending:db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',(queue_key,payload))
        else:db.execute('DELETE FROM metadata WHERE key=?',(queue_key,))
    result['pending_consolidations']=pending
    return result


def rebuild(database,selected,config,ordinary,results):
    if not selected or not config.data['sapiensid']['enabled']:
        return ordinary()
    db=database.connection
    from .dependencies import read_only
    with stage('Validate hybrid rebuild evidence'), read_only():
        key,before=signature(db,selected,config)
    record=db.execute('SELECT value FROM metadata WHERE key=?',('hybrid-rebuild:'+key,)).fetchone()
    if record is not None and record[0]==before:
        return results()
    existing={row[0] for row in db.execute('SELECT id FROM people')}
    ordinary()
    births={row[0] for row in db.execute('SELECT id FROM people')}-existing
    with stage('Consolidate hybrid identities'):
        from .hybrid_queries import recover
        if births:recover(database,selected,config,births=births)
        apply(database,selected,config)
        recover(database,selected,config)
    with stage('Checkpoint hybrid identity decisions'):
        _,after=signature(db,selected,config)
        with db:db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',('hybrid-rebuild:'+key,after))
    return results()
