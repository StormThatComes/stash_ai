import json
import numpy as np
from stash_ai.recognition.independent_evidence import unit
from stash_ai.recognition.query_support import cohorts
from tests.test_recognition_clustering import setup


def test_repeated_pixels_cannot_supply_multiple_query_votes(tmp_path):
    config=setup(tmp_path)
    rows=[dict(id=str(i),scan='s',seconds=i,dimensions=4,
        embedding=np.array([1,0,0,0],dtype='<f4').tobytes(),
        quality=json.dumps(dict(evidence_crop_sha256='a'*64))) for i in range(4)]
    assert len({unit(r) for r in rows})==1
    assert not list(cohorts(rows,config,key=lambda r:'owner',prototype=lambda r:'view',minimum=lambda _:2))
    rows[-1]['quality']=json.dumps(dict(evidence_crop_sha256='b'*64))
    assert list(cohorts(rows,config,key=lambda r:'owner',prototype=lambda r:'view',minimum=lambda _:2))


def test_equal_vectors_without_pixel_proof_remain_independent():
    assert unit(dict(scan='s',seconds=1)) != unit(dict(scan='s',seconds=2))
    assert unit(dict(scan='s',seconds=1,quality='{}')) == unit(dict(scan='s',seconds=1))


def test_votes_require_both_distinct_times_and_pixels_in_every_order():
    from itertools import permutations
    from stash_ai.recognition.independent_evidence import count
    rows = [dict(scan='s',seconds=t,quality=json.dumps({'evidence_crop_sha256':p*64}))
            for t,p in [(1,'a'),(2,'a'),(3,'b'),(3,'c')]]
    assert {count(order) for order in permutations(rows)} == {2}
    assert count([dict(scan='s',seconds=1),dict(scan='s',seconds=1)]) == 1


def test_duplicate_can_improve_representative_without_extra_vote():
    from stash_ai.recognition.view_prototypes import _Cluster
    slot=_Cluster('other','first')
    sample=np.asarray([1.,0.],dtype='<f4')
    base=dict(scan='s',seconds=1,quality=json.dumps(dict(pixels=32,sharpness=1)))
    slot.add(dict(base,id='first'),sample)
    slot.add(dict(base,id='better',quality=json.dumps(dict(pixels=256,sharpness=100))),sample)
    assert slot.observations == ['better']
    assert len(slot.cues) == 1
    assert np.array_equal(slot.total,sample)
