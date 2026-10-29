"""Pure bounded hybrid jobs; SQLite and decisions stay on coordinator."""
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
    # Duplicate verification stays in Python, but close each pair's cursor at
    # its first conflict instead of exhausting the full co-occurrence join.
    result={}
    for a,b in pairs:
        rows=db.execute('SELECT x.geometry,x.embedding,y.geometry,y.embedding FROM members a '
            'JOIN observations x ON x.id=a.observation JOIN observations y ON y.scan=x.scan AND y.cue=x.cue '
            'JOIN members b ON b.observation=y.id WHERE a.person=? AND b.person=? '
            'AND x.id<>y.id AND x.rejected IS NULL AND y.rejected IS NULL',(a,b))
        try:
            result[a,b]=any(not duplicate(dict(geometry=r[0],embedding=r[1]),
                dict(geometry=r[2],embedding=r[3]),True) for r in rows)
        finally:rows.close()
    return result


def proofs(db,owned,nominations,config):
    pool=current();settings=dict(config.data['sapiensid']['matching'])
    threshold=config.data['recognition']['cooccurrence_merge_threshold']
    from .run_cache import current as current_cache,MISSING
    from .database import encoded
    from copy import deepcopy
    import hashlib,pickle
    cache=current_cache()
    def cache_key(payload):
        return ('hybrid-proof',hashlib.sha256(pickle.dumps(payload,protocol=5)).digest())
    def remember(payload,result):
        if cache is not None:cache.put(cache_key(payload),deepcopy(result),2*len(encoded(result).encode())+512)
        return result
    def drain(jobs,keys):
        if not jobs:return
        if pool is None or pool.workers==1:
            for key,job in zip(keys,jobs):yield *key,remember(job.payload,exact(job.payload))
        else:
            with closing(pool.ordered(exact,jobs),name='close hybrid proof workers') as stream:
                for key,job,result in zip(keys,jobs,stream):yield *key,remember(job.payload,result.value)
    for start in range(0,len(nominations),64):
        block=nominations[start:start+64];present=copresence(db,block,config)
        jobs=[];keys=[]
        for offset,(a,b) in enumerate(block):
            left,right=owned[a],owned[b]
            size=16384+8*sum(resident_bytes(v) for i in (left,right) for v in
                (i.faces,i.persons,i.guard_faces,i.guard_persons,*i.body_modes) if v is not None)
            if any(isinstance(m,Matrix) for i in (left,right) for m in i.body_modes):size+=4*1024**2
            payload=left,right,settings,present[a,b],threshold
            hit=cache.get(cache_key(payload)) if cache is not None else MISSING
            if hit is not MISSING:
                yield from drain(jobs,keys);jobs=[];keys=[]
                yield a,b,deepcopy(hit)
                continue
            if pool is not None and size>pool.limit:
                yield from drain(jobs,keys);jobs=[];keys=[]
                pool.fallback('hybrid_proof_admission')
                yield a,b,remember(payload,exact(payload))
            else:
                jobs.append(Work(start+offset,'hybrid-frozen',payload,size));keys.append((a,b))
        yield from drain(jobs,keys)
