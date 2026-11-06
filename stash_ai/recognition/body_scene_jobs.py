"""Only isolated body plans run on workers. The coordinator owns all SQLite I/O."""
from .compute import Work, current as compute
from .derived_cache import current
from .scene_artifacts import body_key
from stash_ai.runtime.cleanup import closing


def _worker(payload):
    from .body_identity import _plan
    config, frozen = payload
    return _plan(None,config,*frozen)


def _witnesses(db, owners, limit):
    from .scan_heads import predicate
    where=(' FROM observations o JOIN members m ON m.observation=o.id JOIN scans s ON s.id=o.scan '
           "WHERE m.person=? AND o.rejected IS NULL AND s.status='complete' AND "+predicate())
    frozen={}
    used=0
    for person in sorted(owners):
        charge=db.execute('SELECT COALESCE(SUM(3*COALESCE(length(o.embedding),0)+512),0)'+where,
                          (person,)).fetchone()[0]+512
        used+=charge
        if used>limit:return None,0
        # Keep raw values. Validate in traversal order on demand, just like the
        # serial guard; a veto may stop before a later corrupt witness.
        frozen[person]=tuple(dict(r) for r in db.execute('SELECT o.dimensions,o.embedding'+where,(person,)))
    return frozen,used


def planned(db, items, config):
    from .body_identity import scene_plan, _load_scene, _plan
    from .hybrid_query_observer import _observer
    store=current() if _observer.get() is None else None
    pool=compute()
    batch=[]
    def drain():
        if batch:
            jobs=(Work(i,key,(config,inputs),size) for i,(key,inputs,size) in enumerate(batch))
            with closing(pool.ordered(_worker,jobs),name='finish body scene plans') as results:
                for result in results:
                    record,rows,decisions=result.value
                    if store:store.put('body-scene',result.generation,decisions)
                    yield record,rows,decisions
            batch.clear()

    for item in items:
        scan=item['scan']
        key=body_key(db,scan,config) if store else None
        cached=store.get('body-scene',key) if store else None
        if cached is not None:
            yield from drain()
            yield db.execute('SELECT * FROM scans WHERE id=?',(scan,)).fetchone(),[{'id':i} for i in cached],cached
            continue
        size=0
        if _observer.get() is None and pool and pool.workers>1:
            # Budget for packed input, local scoring banks, output and guard cache.
            raw=db.execute('SELECT COALESCE(SUM(length(embedding)+length(quality)+length(geometry)+8192),0) '
                           'FROM body_observations WHERE scan=?',(scan,)).fetchone()[0]
            size=131072+raw*12
        if size and size <= pool.limit//max(2,pool.workers):
            frozen=_load_scene(db,scan,config)
            rows=frozen[1]
            faces,charge={},0
            if any(r['face_person'] is None and r['_face_vector'] is not None for r in rows):
                from .body_identity import anchor
                settings=config.data['sapiensid']['identities']
                owners={frozen[2][r['face_person']] for r in rows if r['face_person'] is not None
                        and r['face_anchor'] and anchor(r,settings)}
                room=min(pool.limit//max(2,pool.workers)-size,config.data['sapiensid']['execution']['reference_bytes']//16)
                faces,charge=_witnesses(db,owners,room)
            if faces is not None:
                batch.append((key,(*frozen,faces),size+charge))
                if len(batch)>=pool.workers:
                    yield from drain()
                continue
            yield from drain()
            plan=_plan(db,config,*frozen)
        else:
            yield from drain()
            plan=scene_plan(db,scan,config)
        if store:store.put('body-scene',key,plan[2])
        yield plan
    yield from drain()
