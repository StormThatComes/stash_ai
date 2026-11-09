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


from dataclasses import dataclass
from functools import cached_property


@dataclass(frozen=True)
class Witness:
    dimensions: int
    embedding: bytes
    strict: bool = True
    scan: str = ''
    seconds: float = 0.
    quality: str = '{}'

    @cached_property
    def validated(self):
        from .clustering import vector
        return vector({'dimensions': self.dimensions, 'embedding': self.embedding})


def face_witness_query():
    from .reference_roles import strict_sql
    return "SELECT o.dimensions,o.embedding,"+strict_sql()+",o.scan,o.seconds,json_object(" + (
        "'evidence_pixel_sha256',json_extract(o.quality,'$.evidence_pixel_sha256'),"
        "'evidence_crop_sha256',json_extract(o.quality,'$.evidence_crop_sha256'))") + _bank_source()


def _bank_source():
    from .scan_heads import predicate
    return (' FROM observations o JOIN members m ON m.observation=o.id JOIN scans s ON s.id=o.scan '
            "WHERE m.person=? AND o.rejected IS NULL AND s.status='complete' AND "+predicate())


def _witnesses(db, owners, limit, cache, generation, charges):
    frozen={}
    used=0
    for person in sorted(owners):
        from .run_cache import MISSING
        key = ('body-veto-bank', generation, person)
        hit = cache.get(key) if cache is not None else MISSING
        if hit is not MISSING:
            rows, charge = hit
            used += charge
            if used > limit:return None,0
            frozen[person] = rows
            continue
        charge = charges(person)
        used+=charge
        if used>limit:return None,0
        # Keep raw values. Validate in traversal order on demand, just like the
        # serial guard; a veto may stop before a later corrupt witness.
        frozen[person]=tuple(Witness(*r) for r in db.execute(face_witness_query(),(person,)))
        if cache is not None:
            cache.put(key, (frozen[person], charge), charge)
    return frozen,used


def planned(db, items, config, *, ids_only=False):
    from .body_identity import scene_plan, _load_scene, _plan, _prepare_vectors
    from .hybrid_query_observer import _observer
    from functools import partial
    from types import SimpleNamespace
    from .config import plain
    from .database import encoded
    from .body_observations import contract
    from .run_cache import current as numerical_cache, MISSING
    from .stage_checkpoints import current as stage_current

    observer = _observer.get()
    pool = compute()
    store, checkpoints, cache = current(), stage_current(), numerical_cache()
    target_scans = None
    if observer is not None and hasattr(observer, 'local_budget'):
        addresses = encoded(sorted(observer.sets))
        target_scans = {s for s, in db.execute(
            'SELECT scan FROM body_observations WHERE id IN (SELECT value FROM json_each(?)) UNION '
            'SELECT scan FROM body_observations WHERE face IN (SELECT value FROM json_each(?))', (addresses,addresses))}
        observer.local_budget.start(observer.visits)
    backend = config.data['execution']['rebuild_cohort_backend'] if observer is None and ids_only else 'threads'
    worker_config = SimpleNamespace(data=plain(config.data)) if backend == 'processes' else config
    expected = contract(config)
    settings = config.data['sapiensid']['identities']
    # Every scene in this planning pass reads the same frozen memberships.
    witness_generation = object()
    from functools import lru_cache
    from .identity_merges import canonical

    @lru_cache(maxsize=1024)
    def charge(person):
        return db.execute('SELECT COALESCE(SUM(3*COALESCE(length(o.embedding),0)+512),0)'
                          + _bank_source(), (person,)).fetchone()[0]+512

    def guard_room(scan, cap):
        owners = {canonical(db, row[0]) for row in db.execute(
            'SELECT DISTINCT m.person FROM body_observations b JOIN members m ON m.observation=b.face '
            'WHERE b.scan=? AND b.contract=? AND b.rejected IS NULL AND b.embedding IS NOT NULL', (scan,expected))}
        return min(cap, sum(charge(owner) for owner in owners))

    def job(ordinal, key, size, frozen, faces, wanted):
        if backend == 'processes':
            from .shared_payload import pack
            record, rows, *rest = frozen
            data = [(dict((k,v) for k,v in row.items() if k != 'embedding'), row['embedding']) for row in rows]
            packed, segment = pack(data)
            if segment is not None:
                pool.shared_inputs[ordinal] = segment
                frozen = (record, packed, *rest)
        return Work(ordinal, key, (worker_config, (*frozen,faces), wanted, ids_only), size)

    def prepare(ordinal, scan, key, size, room, wanted):
        if size > pool.limit:
            return Work(ordinal,key,(scene_plan(db,scan,config),[]),size,ready=True)
        frozen = _load_scene(db,scan,config,prepare_vectors=False)
        rows = frozen[1]
        faces, admitted = {}, 0
        if room:
            from .body_identity import anchor
            owners = {frozen[2][row['face_person']] for row in rows if row['face_person'] is not None
                      and row['face_anchor'] and anchor(row,settings)}
            faces, admitted = _witnesses(db, owners, room, cache, witness_generation, charge)
            if faces is None:
                _prepare_vectors(rows)
                return Work(ordinal,key,(_plan(db,config,*frozen),[]),size,ready=True)
        return job(ordinal,key,size-room+admitted,frozen,faces,wanted)

    def jobs():
        for ordinal,item in enumerate(items):
            scan = item['scan']
            wanted = observer is not None and (target_scans is None or scan in target_scans)
            key = body_key(db,scan,config) if not wanted and (store or checkpoints or (observer and cache)) else None
            if key:
                cached = store.get('body-scene',key) if store else None
                if cached is None and checkpoints:cached = checkpoints.get('body-scene',key)
                if cached is None and observer and cache:
                    raw = cache.get(('body-plan',key))
                    if raw is not MISSING:
                        import json
                        cached = json.loads(raw)
                if cached is not None:
                    record = db.execute('SELECT * FROM scans WHERE id=?',(scan,)).fetchone()
                    plan = record,[dict(id=i) for i in cached],cached
                    from .memory_size import size as memory_size
                    yield Work(ordinal,None,(plan,[]),max(1,memory_size(cached)*2+4096),ready=True)
                    continue
            if pool is None or pool.workers == 1 or (observer and target_scans is None):
                yield Work(ordinal,key,(scene_plan(db,scan,config),[]),1,ready=True)
                continue
            # Include native vector copies, packed IPC input, Python geometry,
            # score keys, and the decision output. Excluded rows are never loaded.
            raw, needs_guard = db.execute(
                'SELECT COALESCE(SUM(8*length(b.embedding)+8*(length(b.quality)+length(b.geometry))'
                '+4*COALESCE(length(o.embedding),0)+8192),0), '
                'MAX(o.embedding IS NOT NULL AND o.rejected IS NULL) '
                'FROM body_observations b LEFT JOIN observations o ON o.id=b.face '
                'LEFT JOIN members m ON m.observation=o.id WHERE b.scan=? AND b.contract=? '
                'AND b.rejected IS NULL AND b.embedding IS NOT NULL',(scan,expected)).fetchone()
            size = raw + 131072
            cap = min(max(0,pool.limit-size), config.data['sapiensid']['execution']['reference_bytes']//16)
            room = guard_room(scan,cap) if needs_guard else 0
            size = pool.limit+1 if needs_guard and cap == 0 else size+room
            yield Work(ordinal,key,None,size,ready=size>pool.limit,
                       prepare=partial(prepare,ordinal,scan,key,size,room,wanted))

    if pool is None:
        results = ((work.generation, work.payload) for work in jobs())
    else:
        results = ((result.generation,result.value) for result in pool.independent_scenes(_worker,jobs(),backend=backend))
    with closing(results, name='finish body scene plans') as ordered:
        for key, (plan, hints) in ordered:
            record, rows, decisions = plan
            if hints:
                from .local_body_hints import observe
                from .body_guards import FaceGuard
                for groups, seeds, modes in hints:
                    observe(rows,groups,seeds,None,FaceGuard(db,config),config,frozen=modes)
            if key:
                if store:store.put('body-scene',key,decisions)
                if checkpoints:checkpoints.put('body-scene',key,decisions)
                if observer and cache:
                    raw = encoded(decisions)
                    cache.put(('body-plan',key),raw,len(raw.encode())+512)
            yield record,rows,decisions
