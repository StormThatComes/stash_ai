from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest

from stash_ai.recognition.config import defaults
from stash_ai.recognition.hybrid_evidence import Identity,plan,scores
from stash_ai.recognition.hybrid_lookup import Lookup
from tests.test_recognition_hybrid_evidence import identity


def test_blocked_means_match_all_witness_pairs_across_block_boundaries():
    rng=np.random.default_rng(8);bank=[]
    for i in range(131):
        face=rng.normal(size=(3,8)).astype('float32');face/=np.linalg.norm(face,axis=1)[:,None]
        body=rng.normal(size=(4,32)).astype('float32');body/=np.linalg.norm(body,axis=1)[:,None]
        bank.append(Identity(str(i),(),face,body,()))
    found=list(Lookup(bank,10**7).pairs())
    assert len(found)==131*130//2
    assert len({(a,b) for a,b,_,_ in found})==len(found)
    for a,b,face,body in found:
        left,right=bank[int(a)],bank[int(b)]
        assert face==pytest.approx(scores(left.faces,right.faces)['mean'],abs=1e-12)
        assert body==pytest.approx(scores(left.persons,right.persons)['mean'],abs=1e-12)


@pytest.mark.parametrize('key,value,message',[
    ('reference_bytes',1,'reference_bytes'),('plan_bytes',1,'plan_bytes'),
    ('max_candidate_pairs',1,'max_candidate_pairs')])
def test_limits_fail_explicitly_without_dropping_rivals(monkeypatch,key,value,message):
    monkeypatch.setattr('stash_ai.recognition.global_views._cooccurs',lambda *a:False)
    config=SimpleNamespace(data=defaults());config.data['sapiensid']['execution'][key]=value
    with pytest.raises(ValueError,match=message):
        plan(None,config,[identity('a'),identity('b'),identity('c')])


def test_smallest_budget_error_never_partially_applies_merge(tmp_path):
    from tests.test_recognition_hybrid_merge import prepared
    from stash_ai.recognition.hybrid_merge import apply
    from stash_ai.recognition.database import FaceDatabase
    config,_=prepared(tmp_path);config.data['sapiensid']['execution']['max_candidate_pairs']=1
    config.data['sapiensid']['execution']['plan_bytes']=1
    with FaceDatabase(config.database) as state:
        db=state.connection;before=list(db.iterdump())
        selected=[dict(scene=r['scene'],scan=r['id']) for r in db.execute('SELECT * FROM scans')]
        with pytest.raises(ValueError,match='plan_bytes'):apply(state,selected,config)
        assert list(db.iterdump())==before
