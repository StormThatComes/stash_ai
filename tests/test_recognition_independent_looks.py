import math
from types import SimpleNamespace
import numpy as np
from stash_ai.recognition.appearance_index import canonical_buckets
from stash_ai.recognition.body_identity import Scores
from stash_ai.recognition.body_seed_modes import FrozenSeeds
from tests.test_contact_body_claims import body
from tests.test_recognition_body_identity import configuration


def test_cross_look_support_requires_independent_strict_face_roots(tmp_path):
    cfg=configuration(tmp_path)
    settings=cfg.data['sapiensid']['identities']
    a,b,q=[body(str(i),i,30,'A') for i in range(3)]
    b['_vector']=np.array([.83,math.sqrt(1-.83**2)])
    q['_vector']=(a['_vector']+b['_vector'])/np.linalg.norm(a['_vector']+b['_vector'])
    guard=SimpleNamespace(allows=lambda *_:True)
    for strict in (True,False):
        b['face_anchor']=strict
        bank=FrozenSeeds({'A':[a,b]},Scores(cfg),settings)
        assert bool(bank.rank(q,guard)[0][4])==strict


def test_bridge_hash_does_not_consume_both_endpoints():
    buckets=canonical_buckets({0,7,63},4)
    assert len(set(buckets.values()))==2
