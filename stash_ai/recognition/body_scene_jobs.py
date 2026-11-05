"""Only isolated body plans run on workers. The coordinator owns all SQLite I/O."""
from .compute import Work, current as compute
from .derived_cache import current
from .scene_artifacts import body_key
from stash_ai.runtime.cleanup import closing


def _worker(payload):
    from .body_identity import _plan
    config, frozen = payload
    return _plan(None,config,*frozen)


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
                    store.put('body-scene',result.generation,decisions)
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
        if store and pool and pool.workers>1:
            # Budget for packed input, local scoring banks, output and guard cache.
            raw=db.execute('SELECT COALESCE(SUM(length(embedding)+length(quality)+length(geometry)+8192),0) '
                           'FROM body_observations WHERE scan=?',(scan,)).fetchone()[0]
            size=131072+raw*12
        if size and size <= pool.limit//max(2,pool.workers):
            frozen=_load_scene(db,scan,config)
            rows=frozen[1]
            # Unowned observations with a face can require the complete owner's
            # off-scene veto bank. Keep this case on the SQLite coordinator.
            if not frozen[2] or not any(r['face_person'] is None and r['_face_vector'] is not None for r in rows):
                batch.append((key,frozen,size))
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
