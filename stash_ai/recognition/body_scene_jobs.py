"""Only isolated body plans run on workers. The coordinator owns all SQLite I/O."""
from .compute import Work, current as compute
from .derived_cache import current
from .scene_artifacts import body_key
from stash_ai.runtime.cleanup import closing


def _worker(payload):
    from .body_identity import _plan, _prepare_vectors
    config, frozen, wanted, compact = payload
    from .shared_payload import Rows, unpack
    if isinstance(frozen[1], Rows):
        with unpack(frozen[1]) as packed:
            rows = [dict(metadata, embedding=vector) for metadata, vector in packed]
        frozen = (frozen[0], rows, *frozen[2:])
    _prepare_vectors(frozen[1])
    hints = []
    plan = _plan(None, config, *frozen,
                 capture_hints=lambda groups, seeds, modes: hints.append((groups, seeds, modes)) if wanted else None)
    if compact and not hints:
        record, rows, decisions = plan
        plan = record, [{'id': row['id']} for row in rows], decisions
    return plan, hints


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


def planned(db, items, config, *, ids_only=False):
    from .body_identity import scene_plan, _load_scene, _plan, _prepare_vectors
    from .hybrid_query_observer import _observer
    store=current() if _observer.get() is None else None
    from .stage_checkpoints import current as stage_current
    checkpoints=stage_current() if _observer.get() is None else None
    def save(key,decisions):
        if store:store.put('body-scene',key,decisions)
        if checkpoints:checkpoints.put('body-scene',key,decisions)
    observer = _observer.get()
    pool=compute()
    from types import SimpleNamespace
    from .config import plain
    backend = config.data['execution']['rebuild_cohort_backend'] if observer is None and ids_only else 'threads'
    worker_config = SimpleNamespace(data=plain(config.data)) if backend == 'processes' else config
    batch=[]
    def jobs():
        for ordinal, (key, inputs, size, wanted) in enumerate(batch):
            if backend == 'processes':
                from .shared_payload import pack
                record, rows, *rest = inputs
                packed = [(dict((k,v) for k,v in row.items() if k != 'embedding'), row['embedding']) for row in rows]
                try:
                    packed, segment = pack(packed)
                except OSError:
                    segment = None
                if segment is not None:
                    pool.shared_inputs[ordinal] = segment
                    inputs = (record, packed, *rest)
            yield Work(ordinal, key, (worker_config, inputs, wanted, ids_only), size)

    def drain():
        if batch:
            with closing(pool.ordered(_worker, jobs(), backend=backend),name='finish body scene plans') as results:
                for result in results:
                    (record, rows, decisions), hints = result.value
                    if observer is not None and hasattr(observer, 'local_budget'):
                        observer.local_budget.start(observer.visits)
                        from .local_body_hints import observe
                        from .body_guards import FaceGuard
                        for groups, seeds, modes in hints:
                            observe(rows, groups, seeds, None, FaceGuard(db, config), config, frozen=modes)
                    save(result.generation,decisions)
                    yield record,rows,decisions
            batch.clear()

    for item in items:
        scan=item['scan']
        key=body_key(db,scan,config) if store or checkpoints else None
        cached=store.get('body-scene',key) if store else None
        if cached is None and checkpoints:cached=checkpoints.get('body-scene',key)
        if cached is not None:
            yield from drain()
            yield db.execute('SELECT * FROM scans WHERE id=?',(scan,)).fetchone(),[{'id':i} for i in cached],cached
            continue
        size=0
        if pool and pool.workers > 1 and (observer is None or hasattr(observer, 'local_budget')):
            # Budget for packed input, local scoring banks, output and guard cache.
            raw=db.execute('SELECT COALESCE(SUM(COALESCE(length(embedding),0)+length(quality)+length(geometry)+8192),0) '
                           'FROM body_observations WHERE scan=?',(scan,)).fetchone()[0]
            size=131072+raw*12
        if size and size <= pool.limit:
            if batch and sum(job[2] for job in batch) + size > pool.limit:
                yield from drain()
            frozen=_load_scene(db,scan,config,prepare_vectors=False)
            rows=frozen[1]
            # Fully face-linked scenes have no body queries to compare. Avoid
            # transferring their rows and retaining idle numerical work slots.
            if all(r['face_person'] is not None for r in rows):
                yield from drain()
                _prepare_vectors(rows)
                plan = _plan(db, config, *frozen)
                save(key, plan[2])
                yield plan
                continue
            faces,charge={},0
            if any(r['face_person'] is None and r['face_embedding'] is not None and r['face_rejected'] is None for r in rows):
                from .body_identity import anchor
                settings=config.data['sapiensid']['identities']
                owners={frozen[2][r['face_person']] for r in rows if r['face_person'] is not None
                        and r['face_anchor'] and anchor(r,settings)}
                room=min(pool.limit-size-sum(job[2] for job in batch),config.data['sapiensid']['execution']['reference_bytes']//16)
                faces,charge=_witnesses(db,owners,room)
            if faces is not None:
                wanted = observer is not None and any(
                    r['id'] in observer.sets or r['face'] in observer.sets for r in rows)
                batch.append((key, (*frozen, faces), size+charge, wanted))
                if len(batch)>=pool.workers:
                    yield from drain()
                continue
            yield from drain()
            _prepare_vectors(rows)
            plan=_plan(db,config,*frozen)
        else:
            yield from drain()
            plan=scene_plan(db,scan,config)
        save(key,plan[2])
        yield plan
    yield from drain()
