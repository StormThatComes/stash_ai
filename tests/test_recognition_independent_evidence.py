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


def test_raw_fingerprint_deduplicates_copies_across_scenes():
    from stash_ai.recognition.independent_evidence import count
    rows = [dict(scan=s, seconds=t, quality=json.dumps(dict(evidence_pixel_sha256='a'*64)))
            for s,t in [('first', 0), ('copy', 30)]]
    assert count(rows) == 1


def test_near_copy_requires_pixels_aspect_texture_and_embedding():
    from stash_ai.recognition.evidence_pixels import verified
    rng = np.random.default_rng(7)
    pixels = rng.integers(10, 245, 3072, dtype='uint8')
    a = np.array([1.,0.],dtype='<f4')
    assert verified(pixels, pixels+1, a, a, 1., 1.)
    assert not verified(pixels, pixels+10, a, a, 1., 1.)
    assert not verified(pixels, pixels, a, a[::-1], 1., 1.)
    assert not verified(pixels, pixels, a, a, 1., 0.)
    assert not verified(np.zeros(3072), np.zeros(3072), a, a, 1., 1.)


def test_pixel_fingerprint_does_not_depend_on_jpeg_budget():
    from PIL import Image
    from stash_ai.recognition.quality import crop_bytes
    from types import SimpleNamespace
    image = Image.fromarray(np.random.default_rng(3).integers(0,256,(128,128,3),dtype='uint8'))
    first, second = {}, {}
    face = SimpleNamespace(box=(10,10,100,100))
    try:
        a,_ = crop_bytes(image,face,limit=100000,details=first)
        b,_ = crop_bytes(image,face,limit=1000,details=second)
        assert a != b
        assert first['evidence_pixel_sha256'] == second['evidence_pixel_sha256']
    finally:
        image.close()
