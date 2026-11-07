from contextlib import closing
import numpy as np
from stash_ai.recognition.hybrid_query_index import QueryIndex
from stash_ai.recognition.hybrid_evidence import scores


def test_disk_and_ram_nominate_same_banks_including_floor_ties():
    q=np.array([1.,0.],dtype='f4');z=np.empty((0,2))
    banks=[(i,None,z,np.array([v,v],dtype='f4')) for i,v in enumerate(([1,0],[0,1],[.5,.5],[-1,0]))]
    for budget in (0,1024**2):
        with closing(QueryIndex(banks,budget)) as idx:
            assert (idx.disk is not None)==(budget==0)
            result=[(row,[item[0] for item in gen]) for row,_,gen in idx.batch([(1,q),(2,q)],.5)]
            assert result==[(1,[0,2]),(2,[0,2])]


def test_tiled_exact_scores_match_full_cross_product():
    rng=np.random.default_rng(47)
    a,b=(rng.normal(size=(n,16)) for n in (19,145))
    a/=np.linalg.norm(a,axis=1,keepdims=True);b/=np.linalg.norm(b,axis=1,keepdims=True)
    expected=a@b.T;actual=scores(a,b)
    np.testing.assert_allclose([actual['minimum'],actual['mean']],[expected.min(),expected.mean()],atol=1e-12)
