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


def test_sparse_incident_rows_preserve_order_and_visit_only_active_cells(monkeypatch):
    from stash_ai.recognition.hybrid_lookup import Lookup
    from stash_ai.recognition import parallel_numeric
    bank=[identity(str(i),i*.01) for i in range(256)]
    lookup=Lookup(bank,2**26)
    dense=[p for p in lookup.pairs() if '0' in p[:2]]
    original=parallel_numeric.matrix_tiles
    cells=[]
    def measure(tiles,**options):
        def counted():
            for meta,pairs in tiles:
                cells.append(len(pairs[0][0])*len(pairs[0][1]))
                yield meta,pairs
        yield from original(counted(),**options)
    monkeypatch.setattr(parallel_numeric,'matrix_tiles',measure)
    sparse=list(lookup.pairs(incident={'0'}))
    assert [p[:2] for p in sparse]==[p[:2] for p in dense]
    assert np.allclose([p[2:] for p in sparse],[p[2:] for p in dense])
    assert sum(cells)==255


def test_sparse_catalogue_skips_unchanged_empty_block_pairs(tmp_path,monkeypatch):
    from collections import Counter
    from stash_ai.recognition import profiling
    from stash_ai.recognition.hybrid_lookup import Lookup
    from stash_ai.recognition.hybrid_delta import nominations
    config=SimpleNamespace(data=defaults())
    bank=[identity(str(i),i*.07) for i in range(256)]
    stats=Counter()
    monkeypatch.setattr(profiling,'count',lambda key,n=1:stats.update({key:n}))
    with FaceDatabase(tmp_path/'blocks.sqlite') as db,scope(db,config):
        from stash_ai.recognition.derived_cache import current
        store=current()
        assert not list(nominations(Lookup(bank,2**26),store,2.,2.))
        cold=stats['hybrid_block_pairs_visited']
        stats.clear()
        assert not list(nominations(Lookup(bank,2**26),store,2.,2.))
        assert stats['hybrid_block_pairs_visited']==0
        bank.append(identity('new',.2))
        assert not list(nominations(Lookup(bank,2**26),store,2.,2.))
        assert 0<stats['hybrid_block_pairs_visited']<cold


def test_face_bounds_keep_weak_boundary_and_small_catalogue():
    from stash_ai.recognition.scoring import FrozenFaceBlocks
    from stash_ai.recognition.clustering import Group
    query=np.array([1,0],dtype='float32')
    groups=[Group(str(i),(-query if i<64 else query,),True) for i in range(128)]
    blocks=FrozenFaceBlocks(groups,2**20)
    assert list(blocks.candidates(query,.999999))==groups[64:]
    assert list(FrozenFaceBlocks(groups[:8],2**20).candidates(query,.9))==groups[:8]
