"""Pure bounded hybrid numerical jobs; SQLite and decisions stay on coordinator."""
from .compute import current, Work
from stash_ai.runtime.cleanup import closing


def products(payload):
    a,b,c,d=payload
    return a @ b.T,c @ d.T


def exact(payload):
    from .hybrid_evidence import pair
    left,right,settings,copresent,threshold=payload
    return pair(left,right,settings,copresent=copresent,reflection_threshold=threshold)


def proofs(db,owned,nominations,config):
    from .global_views import _cooccurs
    pool=current();settings=dict(config.data['sapiensid']['matching'])
    threshold=config.data['recognition']['cooccurrence_merge_threshold']
    def items():
        for ordinal,(a,b) in enumerate(nominations):
            left,right=owned[a],owned[b]
            size=16384+8*sum(v.nbytes for i in (left,right) for v in
                (i.faces,i.persons,i.guard_faces,i.guard_persons) if v is not None)
            yield ordinal,a,b,(left,right,settings,_cooccurs(db,a,b,config),threshold),size
    if pool is None or pool.workers==1:
        for _,a,b,payload,_ in items():yield a,b,exact(payload)
        return
    # Oversized individual proofs use the ordinary bounded scalar path.
    if any(16384+8*sum(v.nbytes for i in (owned[a],owned[b]) for v in
        (i.faces,i.persons,i.guard_faces,i.guard_persons) if v is not None)>pool.limit for a,b in nominations):
        pool.fallback('hybrid_proof_admission')
        for _,a,b,payload,_ in items():yield a,b,exact(payload)
        return
    jobs=(Work(n,'hybrid-frozen',payload,size) for n,_,_,payload,size in items())
    with closing(pool.ordered(exact,jobs),name='close hybrid proof workers') as stream:
        for (a,b),result in zip(nominations,stream):yield a,b,result.value
