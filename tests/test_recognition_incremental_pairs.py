from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from stash_ai.recognition.derived_cache import scope
from stash_ai.recognition.config import defaults
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.hybrid_evidence import plan
from tests.test_recognition_hybrid_evidence import identity


def test_new_competitor_invalidates_old_merge_without_rescoring_its_proof(tmp_path,monkeypatch):
    from stash_ai.recognition import hybrid_jobs
    monkeypatch.setattr('stash_ai.recognition.global_views._cooccurs',lambda *a:False)
    config=SimpleNamespace(data=defaults())
    original=hybrid_jobs.exact
    calls=[]
    def measured(payload):
        calls.append(tuple(item.person for item in payload[:2]))
        return original(payload)
    monkeypatch.setattr(hybrid_jobs,'exact',measured)
    pair=[identity('a'),identity('b',.1)]
    path=tmp_path/'faces.sqlite'
    with FaceDatabase(path) as db,scope(db,config):
        result=plan(None,config,pair,compact=True)
        assert any(d['outcome']=='merge' for d in result['decisions'])
        db.connection.commit()
    rival=replace(identity('c'),persons=np.empty((0,2)),independent=())
    calls.clear()
    with FaceDatabase(path) as db,scope(db,config):
        result=plan(None,config,pair+[rival],compact=True)
        assert ('a','b') not in calls
        decision=next(d for d in result['decisions'] if d['people']==['a','b'])
        assert decision['blockers']==['competing_identity']
    assert result['decisions']==plan(None,config,pair+[rival])['decisions']



def test_block_bound_and_cpu_nominations_match_unpruned_lookup():
    from stash_ai.recognition.hybrid_lookup import Lookup
    from stash_ai.recognition.hybrid_delta import nominations
    from stash_ai.recognition.nomination_bounds import ball,upper
    bank=[identity(str(i),i*.071) for i in range(180)]
    lookup=Lookup(bank,2**26)
    expected=sorted(lookup.pairs(face_floor=.8,person_floor=.95))
    found=sorted(nominations(lookup,None,.8,.95))
    assert [p[:2] for p in found]==[p[:2] for p in expected]
    assert np.allclose([p[2:] for p in found],[p[2:] for p in expected],rtol=0,atol=1e-12)
    a=lookup.faces[:70];b=lookup.faces[109:]
    assert upper(ball(a),ball(b))>=(a@b.T).max()


def test_cuda_tiles_fall_back_at_boundary_or_device_error():
    from stash_ai.recognition.nomination_cuda import Resident
    device=Resident.__new__(Resident)
    device.active=True
    device.scores=lambda *_:(np.array([[.75-1e-6]]),np.array([[.5]]))
    assert device.measured_tile(None,None,.75,.96) is None
    def offline(*args):raise RuntimeError("device unavailable")
    device.scores=offline
    assert device.measured_tile(None,None,.75,.96) is None
    assert not device.active

def test_real_sapiens_buffers_have_stable_content_versions():
    from stash_ai.recognition.hybrid_delta import endpoint_version
    owner=replace(identity('a'),persons=np.zeros((8,4096),dtype='<f4'))
    version=endpoint_version(owner)
    assert endpoint_version(replace(owner,persons=owner.persons.copy()))==version
    owner.persons[0,0]=.001
    assert endpoint_version(owner)!=version
