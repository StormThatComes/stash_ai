from dataclasses import replace
import json
import math

import numpy as np
import pytest

from stash_ai.recognition.clustering import matching_policy
from stash_ai.recognition.config import plain
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.rebuild import rebuild_selected
from stash_ai.recognition.rebuild_cache import validate_vectors,admissible,resource_limits
from tests.test_recognition_clustering import setup,scan


def changed(config,**clustering):
    data=plain(config.data);data['clustering'].update(clustering)
    return replace(config,data=data)


def test_increased_capacity_reuses_but_reduction_and_semantic_change_replay(tmp_path,monkeypatch):
    from stash_ai.recognition import rebuild
    config=setup(tmp_path)
    with FaceDatabase(config.database) as state:
        identifier,_=scan(state,config,'1')
        selected=[dict(scene='1',scan=identifier)]
        expected=rebuild_selected(state,selected,config)
        increased=changed(config,max_reference_bytes=268435456,max_people_per_lookup=40000)
        assert matching_policy(increased)==matching_policy(config)
        with monkeypatch.context() as patch:
            patch.setattr(rebuild,'cluster_scene',lambda *a,**k:pytest.fail('capacity increase replayed'))
            assert rebuild_selected(state,selected,increased)==expected
        calls=[];original=rebuild.cluster_scene
        def cluster(*a,**k):
            calls.append(1);return original(*a,**k)
        monkeypatch.setattr(rebuild,'cluster_scene',cluster)
        reduced=changed(config,max_reference_bytes=67108864)
        rebuild_selected(state,selected,reduced)
        assert calls
        assert matching_policy(changed(config,max_samples_per_person=4))!=matching_policy(config)


def test_multiple_selection_descriptors_require_current_exact_candidates(tmp_path,monkeypatch):
    from stash_ai.recognition import rebuild
    config=setup(tmp_path)
    with FaceDatabase(config.database) as state:
        selected=[]
        for i in range(2):
            identifier,_=scan(state,config,str(i+1))
            selected.append(dict(scene=str(i+1),scan=identifier))
        expected=rebuild_selected(state,selected,config)
        rebuild_selected(state,selected[:1],config)
        with monkeypatch.context() as patch:
            patch.setattr(rebuild,'cluster_scene',lambda *a,**k:pytest.fail('valid earlier selection replayed'))
            assert rebuild_selected(state,selected,config)==expected
        state.connection.execute('DELETE FROM candidates WHERE id=(SELECT id FROM candidates LIMIT 1)')
        state.connection.commit()
        calls=[];original=rebuild.cluster_scene
        def cluster(*a,**k):calls.append(1);return original(*a,**k)
        monkeypatch.setattr(rebuild,'cluster_scene',cluster)
        rebuild_selected(state,selected,config)
        assert len(calls)==2


@pytest.mark.parametrize('mutation',[
    "UPDATE candidates SET payload='bad-json'",
    "UPDATE candidates SET id='wrong'",
    "UPDATE candidates SET status='withheld'",
    "UPDATE observations SET geometry='repaired'",
    "UPDATE scans SET input_sha='repaired'",
])
def test_bad_derived_cache_or_repaired_inputs_force_retained_evidence_replay(tmp_path,monkeypatch,mutation):
    from stash_ai.recognition import rebuild
    config=setup(tmp_path)
    with FaceDatabase(config.database) as state:
        identifier,_=scan(state,config,'1')
        selected=[dict(scene='1',scan=identifier)]
        rebuild_selected(state,selected,config)
        if "id='wrong'" in mutation:
            mutation+=" WHERE id=(SELECT id FROM candidates LIMIT 1)"
        state.connection.execute(mutation);state.connection.commit()
        calls=[];original=rebuild.cluster_scene
        def cluster(*a,**k):calls.append(1);return original(*a,**k)
        monkeypatch.setattr(rebuild,'cluster_scene',cluster)
        rebuild_selected(state,selected,config)
        assert calls==[1]


@pytest.mark.parametrize('dimensions',[4,128,512,1024])
def test_batched_validation_matches_scalar_at_norm_and_finite_boundaries(dimensions):
    from stash_ai.recognition.clustering import vector
    for length in [1.,.999,1.001,1.002,float('nan'),float('inf')]:
        v=np.zeros(dimensions,dtype='<f4');v[0]=length
        row=dict(dimensions=dimensions,embedding=v.tobytes())
        try:
            vector(row)
        except ValueError as error:
            with pytest.raises(ValueError,match=str(error).replace('(','\\(').replace(')','\\)')):
                validate_vectors([row]*10)
        else:
            validate_vectors([row]*10)


def test_binding_budget_fallback_cannot_be_reused_under_larger_limits(tmp_path):
    config=setup(tmp_path)
    state=dict(format=2,limits=resource_limits(config),resource_reuse=False)
    assert admissible(state,config)
    assert not admissible(state,changed(config,max_people_per_lookup=40000))


def test_warm_candidate_sql_and_vectors_use_bounded_scan_blocks(tmp_path,monkeypatch):
    from benchmarks.benchmark_recognition_replay import fixture
    from stash_ai.recognition import rebuild
    config=setup(tmp_path)
    data=plain(config.data);data['execution']['profile_rebuild']=True
    config=replace(config,data=data)
    with FaceDatabase(config.database) as state:
        selected=fixture(state,260)
        expected=rebuild_selected(state,selected,config)
        statements=[]
        state.connection.set_trace_callback(statements.append)
        monkeypatch.setattr(rebuild,'cluster_scene',lambda *a,**k:pytest.fail('warm replay'))
        assert rebuild_selected(state,selected,config)==expected
        state.connection.set_trace_callback(None)
        checks=[sql for sql in statements if 'AS actual_scan' in sql]
        assert len(checks)==2
        assert state.last_rebuild_profile['counts']['batch_validated_vectors']==780


def test_completed_descriptor_catalog_is_bounded_and_corruption_recoverable(tmp_path):
    from stash_ai.recognition import rebuild_cache
    config=setup(tmp_path)
    with FaceDatabase(config.database) as state:
        for i in range(12):
            rebuild_cache.save(state.connection,'scope','ns',dict(fingerprint=str(i),format=2))
        assert state.connection.execute("SELECT COUNT(*) FROM metadata WHERE key LIKE 'identity-rebuild-cache:%'").fetchone()[0]==8
        assert rebuild_cache.lookup(state.connection,'scope','ns','0') is None
        assert rebuild_cache.lookup(state.connection,'scope','ns','4')['fingerprint']=='4'
        state.connection.execute("UPDATE metadata SET value='corrupt' WHERE key LIKE 'identity-rebuild-cache:%'")
        assert rebuild_cache.lookup(state.connection,'scope','ns','4') is None
