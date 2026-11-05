"""Pure bounded hybrid jobs; SQLite and decisions stay on coordinator."""
from .profiling import measured,count,timed_stream
from .compute import current, Work
from .disk_body_modes import Matrix,resident_bytes
from stash_ai.runtime.cleanup import closing


def products(payload):
    a,b,c,d=payload
    return a @ b.T,c @ d.T


def exact(payload):
    from .hybrid_evidence import pair
    left,right,settings,copresent,threshold=payload
    return pair(left,right,settings,copresent=copresent,reflection_threshold=threshold)


@measured('hybrid_copresence')
def copresence(db,pairs,config):
    """One bounded join per nomination block, preserving physical-duplicate rules."""
    from .global_views import _cooccurs
    if db is None:return {pair:_cooccurs(db,*pair,config) for pair in pairs}
    from .database import encoded
    from .physical_faces import duplicate
    enabled=bool(config.data['clustering'].get('deduplicate_detections'))
    if not enabled:
        # EXISTS stops as soon as one real conflicting observation is found.
        rows=db.execute("WITH pairs AS (SELECT json_extract(value,'$[0]') a,json_extract(value,'$[1]') b FROM json_each(?)) "
            'SELECT pairs.a,pairs.b,EXISTS(SELECT 1 FROM members a JOIN observations x ON x.id=a.observation '
            'JOIN observations y ON y.scan=x.scan AND y.cue=x.cue '
            'JOIN members b ON b.observation=y.id WHERE a.person=pairs.a AND b.person=pairs.b '
            'AND x.id<>y.id AND x.rejected IS NULL AND y.rejected IS NULL) FROM pairs',(encoded(pairs),))
        return {(r[0],r[1]):bool(r[2]) for r in rows}
    result=dict.fromkeys(pairs,False)
    rows=db.execute("WITH pairs(a,b) AS (SELECT json_extract(value,'$[0]'),json_extract(value,'$[1]') "
        "FROM json_each(?)) SELECT pairs.a,pairs.b,x.geometry,x.embedding,y.geometry,y.embedding "
        "FROM pairs JOIN members a ON a.person=pairs.a JOIN observations x ON x.id=a.observation "
        "JOIN observations y ON y.scan=x.scan AND y.cue=x.cue "
        "JOIN members b ON b.observation=y.id AND b.person=pairs.b "
        "WHERE x.id<>y.id AND x.rejected IS NULL AND y.rejected IS NULL",(encoded(pairs),))
    for a,b,gx,ex,gy,ey in rows:
        if not result[a,b]:
            result[a,b]=not duplicate(dict(geometry=gx,embedding=ex),dict(geometry=gy,embedding=ey),True)
    return result


def cue_index(db,owned,budget):
    """An overbudget optional index falls back to exact batched SQL."""
    if db is None or len(owned)<64:return None
    by_owner={}
    used=0
    rows=db.execute('SELECT m.person,o.scan,o.cue FROM members m JOIN observations o ON m.observation=o.id '
                    'WHERE o.rejected IS NULL GROUP BY m.person,o.scan,o.cue')
    try:
        for owner,scan,cue in rows:
            if owner not in owned:continue
            used+=256+len(scan)*4
            if used>budget:return None
            by_owner.setdefault(owner,set()).add((scan,cue))
    finally:
        rows.close()
    return by_owner


def proofs(db,owned,nominations,config,*,compact=False):
    pool=current();settings=dict(config.data['sapiensid']['matching'])
    threshold=config.data['recognition']['cooccurrence_merge_threshold']
    from .run_cache import current as current_cache,MISSING
    from .database import encoded
    from copy import deepcopy
    import hashlib,pickle
    from .derived_cache import current as derived
    from .config import digest
    from .hybrid_delta import endpoint_version
    cache=current_cache()
    store=derived() if compact else None
    from itertools import islice
    from .profiling import count
    signatures={}
    policy=encoded(settings)
    def signature(identity):
        if identity.person not in signatures:
            version=(endpoint_version(identity) if store else hashlib.sha256(pickle.dumps(identity,protocol=5)).hexdigest())
            signatures[identity.person]=version
            if store and version not in store.previous_endpoints:
                store.put('hybrid-endpoint',version,True)
        return signatures[identity.person]
    def cache_key(payload):
        left,right,_,present,threshold=payload
        return ('hybrid-proof-v2',signature(left),signature(right),policy,present,threshold)
    def remember(payload,result):
        key=cache_key(payload) if cache is not None or store is not None else None
        if cache is not None:cache.put(key,deepcopy(result),2*len(encoded(result).encode())+512)
        if store is not None:
            saved=dict(blockers=result['blockers']) if result['blockers'] else result
            store.put('hybrid-pair',digest(key),saved)
            count('hybrid_pair_proofs_recomputed')
            if key[1] in store.previous_endpoints and key[2] in store.previous_endpoints:
                count('hybrid_unchanged_pair_proofs_recomputed')
        return result
    def drain(jobs,keys):
        if not jobs:return
        if pool is None or pool.workers==1:
            for key,job in zip(keys,jobs):yield *key,remember(job.payload,exact(job.payload))
        else:
            with closing(pool.ordered(exact,jobs),name='close hybrid proof workers') as stream:
                for key,job,result in zip(keys,jobs,stream):yield *key,remember(job.payload,result.value)
    cues=cue_index(db,owned,min(16*1024**2,config.data['sapiensid']['execution']['reference_bytes']//16))
    pending=timed_stream('hybrid_nomination',nominations)
    start=0
    while block:=list(islice(pending,64)):
        if compact:
            from .hybrid_evidence import preliminary_blocker
            remaining=[]
            for a,b in block:
                blocker=preliminary_blocker(owned[a],owned[b],settings)
                if blocker:
                    count('hybrid_'+blocker)
                    yield a,b,dict(blockers=[blocker])
                else:
                    remaining.append((a,b))
            block=remaining
        if not block:continue
        overlap=(block if cues is None else [(a,b) for a,b in block
                 if not cues.get(a,set()).isdisjoint(cues.get(b,set()))])
        present=copresence(db,overlap,config) if overlap else {}
        count('hybrid_copresence_pairs',len(overlap))
        count('hybrid_exact_pairs',len(block))
        jobs=[];keys=[]
        for offset,(a,b) in enumerate(block):
            left,right=owned[a],owned[b]
            size=16384+8*sum(resident_bytes(v) for i in (left,right) for v in
                (i.faces,i.persons,i.guard_faces,i.guard_persons,*i.body_modes) if v is not None)
            if any(isinstance(m,Matrix) for i in (left,right) for m in i.body_modes):size+=4*1024**2
            payload=left,right,settings,present.get((a,b),False),threshold
            hit=cache.get(cache_key(payload)) if cache is not None else MISSING
            if hit is MISSING and store is not None:
                saved=store.get('hybrid-pair',digest(cache_key(payload)))
                if saved is not None:hit=saved
            if hit is not MISSING:
                yield from timed_stream('hybrid_proof_scoring',drain(jobs,keys));jobs=[];keys=[]
                yield a,b,deepcopy(hit)
                continue
            if pool is not None and size>pool.limit:
                yield from timed_stream('hybrid_proof_scoring',drain(jobs,keys));jobs=[];keys=[]
                pool.fallback('hybrid_proof_admission')
                yield a,b,remember(payload,exact(payload))
            else:
                jobs.append(Work(start+offset,'hybrid-frozen',payload,size));keys.append((a,b))
        yield from timed_stream('hybrid_proof_scoring',drain(jobs,keys))
        start+=len(block)
