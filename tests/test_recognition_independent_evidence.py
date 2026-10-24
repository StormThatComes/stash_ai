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
