"""Pure bounded hybrid jobs; SQLite and decisions stay on coordinator."""
from .compute import current, Work
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
    fields=',x.geometry,x.embedding,y.geometry,y.embedding' if enabled else ''
    rows=db.execute("WITH pairs AS (SELECT json_extract(value,'$[0]') a,json_extract(value,'$[1]') b FROM json_each(?)) "
        'SELECT pairs.a,pairs.b'+fields+' FROM pairs JOIN members a ON a.person=pairs.a '
        'JOIN observations x ON x.id=a.observation JOIN observations y ON y.scan=x.scan AND y.cue=x.cue '
        'JOIN members b ON b.observation=y.id AND b.person=pairs.b '
        'WHERE x.id<>y.id AND x.rejected IS NULL AND y.rejected IS NULL',(encoded(pairs),))
    found=set()
    for row in rows:
        key=tuple(row[:2])
        if key in found:continue
        if not enabled or not duplicate(dict(geometry=row[2],embedding=row[3]),dict(geometry=row[4],embedding=row[5]),True):
            found.add(key)
    return {pair:pair in found for pair in pairs}


def proofs(db,owned,nominations,config):
    pool=current();settings=dict(config.data['sapiensid']['matching'])
    threshold=config.data['recognition']['cooccurrence_merge_threshold']
    def drain(jobs,keys):
        if not jobs:return
        if pool is None or pool.workers==1:
            for key,job in zip(keys,jobs):yield *key,exact(job.payload)
        else:
            with closing(pool.ordered(exact,jobs),name='close hybrid proof workers') as stream:
                for key,result in zip(keys,stream):yield *key,result.value
    for start in range(0,len(nominations),64):
        block=nominations[start:start+64];present=copresence(db,block,config)
        jobs=[];keys=[]
        for offset,(a,b) in enumerate(block):
            left,right=owned[a],owned[b]
            size=16384+8*sum(v.nbytes for i in (left,right) for v in
                (i.faces,i.persons,i.guard_faces,i.guard_persons,*i.body_modes) if v is not None)
            payload=left,right,settings,present[a,b],threshold
            if pool is not None and size>pool.limit:
                yield from drain(jobs,keys);jobs=[];keys=[]
                pool.fallback('hybrid_proof_admission')
                yield a,b,exact(payload)
            else:
                jobs.append(Work(start+offset,'hybrid-frozen',payload,size));keys.append((a,b))
        yield from drain(jobs,keys)
