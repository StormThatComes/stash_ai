import numpy as np
from types import SimpleNamespace
from unittest.mock import patch
from stash_ai.recognition.hybrid_evidence import Identity,plan
from stash_ai.recognition.config import defaults
from stash_ai.recognition.joint_claims import solve


def test_face_contradiction_cannot_be_body_only_competitor():
    v=np.array([[1.,0.]],dtype='float32')
    identities=[Identity(n,(n,),v if n!='c' else -v,v,(1,2,3)) for n in 'abc']
    with patch('stash_ai.recognition.global_views._cooccurs',return_value=False):
        result=plan(None,SimpleNamespace(data=defaults()),identities)
    assert any(d['people']==['a','b'] and d['outcome']=='merge' for d in result['decisions'])


def test_joint_solver_prunes_impossible_large_product_without_dropping_rivals():
    domains=[[(str(j),1. if j==i else 0.,1.) for j in range(8)] for i in range(8)]
    solution=solve(domains,lambda i,a,j,b:a[0]==str(i) and b[0]==str(j),.1)
    assert solution is not None and [v[0] for v in solution]==list(map(str,range(8)))
    assert solve(domains,lambda *a:True,.1,maximum=10) is None
