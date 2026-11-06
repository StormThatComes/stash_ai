from dataclasses import replace
import numpy as np


def test_absent_body_never_nominates_at_negative_body_floor():
    from tests.test_recognition_hybrid_evidence import identity
    from stash_ai.recognition.hybrid_lookup import Lookup
    from stash_ai.recognition.hybrid_delta import nominations
    bank=[replace(identity(str(i),0),faces=np.tile(np.eye(80,dtype='float32')[i],(3,1)),
                  persons=np.empty((0,2),dtype='float32')) for i in range(80)]
    lookup=Lookup(bank,2**25)
    assert list(lookup.pairs(face_floor=.8,person_floor=-.01))==[]
    assert list(nominations(lookup,None,.8,-.01))==[]


def test_one_incident_owner_visits_only_active_tiles(monkeypatch):
    from tests.test_recognition_hybrid_evidence import identity
    from stash_ai.recognition.hybrid_lookup import Lookup
    original=np.split;calls=[]
    def counted(*a,**kw):calls.append(1);return original(*a,**kw)
    monkeypatch.setattr(np,'split',counted)
    lookup=Lookup([identity(str(i),i*.001) for i in range(1024)],2**28)
    pairs=list(lookup.pairs(incident={'0'}))
    assert len(pairs)==1023
    assert len(calls)<=16


def test_body_workers_run_without_disk_cache(tmp_path,monkeypatch):
    from tests.test_recognition_body_global import configured,scene
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.derived_cache import scope
    from stash_ai.recognition.compute import compute_scope
    from stash_ai.recognition.body_scene_jobs import planned
    from stash_ai.recognition.body_identity import scene_plan
    monkeypatch.setattr('stash_ai.recognition.compute.available_cpus',lambda:4)
    config=configured(tmp_path);config.data['execution'].update(rebuild_workers=2,rebuild_cache_bytes=0)
    with FaceDatabase(config.database) as state:
        selected=[scene(state.connection,config,str(i))[0] for i in range(1,5)]
        expected=[scene_plan(state.connection,s['scan'],config)[2] for s in selected]
        with scope(state,config),compute_scope(config) as pool:
            actual=[r[2] for r in planned(state.connection,selected,config)]
            assert pool.submitted==4
        assert actual==expected
