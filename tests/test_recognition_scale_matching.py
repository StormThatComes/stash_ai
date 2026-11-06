from contextlib import ExitStack
from types import SimpleNamespace
import tempfile
import numpy as np


def test_spatial_tree_keeps_every_threshold_pair_including_reversed_owners():
    from stash_ai.recognition.body_global_search import Search,ROUNDING
    from stash_ai.recognition.centroid_tree import reorder,Tree
    from stash_ai.recognition.config import defaults
    c=SimpleNamespace(data=defaults());count=300
    rng=np.random.default_rng(2)
    matrix=np.zeros((count,4096),dtype='<f8');matrix[:,:8]=rng.normal(size=(count,8));matrix/=np.linalg.norm(matrix,axis=1)[:,None]
    with ExitStack() as stack:
        search=Search(None,c,{str(i):{} for i in range(count)})
        search.stack=stack;search.index=stack.enter_context(tempfile.TemporaryFile());search.index.write(matrix.tobytes())
        search.entries=[(i,[]) for i in range(count)];search.block=16
        reorder(search)
        def read(start):
            search.index.seek(start*32768)
            return np.frombuffer(search.index.read(min(16,count-start)*32768),dtype='<f8').reshape(-1,4096)
        tree=Tree(search,read);found=set();threshold=.9
        for i,j in tree.pairs(floor=threshold):
            values=read(i)@read(j).T
            for a,b in zip(*np.nonzero(values+ROUNDING>=threshold)):
                if i+a>=j+b:continue
                found.add(tuple(sorted((search.entries[i+a][0],search.entries[j+b][0]))))
        expected={(i,j) for i in range(count) for j in range(i+1,count) if float(matrix[i]@matrix[j])+ROUNDING>=threshold}
        assert found==expected
        active=np.zeros(count,dtype=bool);active[17]=True
        assert all((i<=17<i+16 or j<=17<j+16) for i,j in tree.pairs(active=active))


def test_block_certificate_rejects_only_after_scalar_witness():
    from stash_ai.recognition.body_certificates import contradiction
    v=np.zeros(4096,dtype='<f4');v[0]=1
    class Evidence:
        stats={}
        charged=0
        def feature(self,row,kind):return row
        def charge(self,n,kind):self.charged+=n
    evidence=Evidence()
    assert contradiction(evidence,[v]*100,[-v]*100,.8,'body')==-1
    assert evidence.charged==1
    assert contradiction(evidence,[v]*10,[v]*10,1.,'body') is None
