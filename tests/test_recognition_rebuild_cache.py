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
        assert not checks
        assert state.last_rebuild_profile['counts']['certified_warm_rebuilds']==1
        assert not state.last_rebuild_profile['counts'].get('validated_vectors',0)
        # Lost certification still uses the bounded full integrity fallback.
        state.connection.execute("DELETE FROM evidence_fingerprints WHERE kind='completed'");state.connection.commit()
        statements.clear();state.connection.set_trace_callback(statements.append)
        assert rebuild_selected(state,selected,config)==expected
        state.connection.set_trace_callback(None)
        assert len([sql for sql in statements if 'AS actual_scan' in sql])==2
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


def assignment_fixture(state, *, scenes=3, candidates=600):
    from benchmarks.benchmark_recognition_replay import fixture
    from stash_ai.recognition.database import encoded
    selected = fixture(state, scenes, dimensions=4, people=4)
    observations = list(state.connection.execute('SELECT id,scan FROM observations ORDER BY id'))
    rows = []
    for i in range(candidates):
        row = observations[i % len(observations)]
        # Include mismatched scan fields and different policies deliberately:
        # the newest addressed candidate, rather than its scan/policy, decides.
        payload = encoded(dict(observation=row['id'], reason=f'reason-{i}',
                               strict_confirmation=dict(source=i, roots=[i % 7])))
        if i % 47 == 0:
            payload = 'invalid-json'
        elif i % 43 == 0:
            payload = encoded(dict(reason='no observation'))
        rows.append((f'candidate-{i}', selected[-1]['scan'], f'policy-{i % 3}',
                     f'person-{i % 5}', 'accepted' if i % 2 else 'withheld', payload))
    state.connection.executemany('INSERT INTO candidates VALUES (?,?,?,?,?,?)', rows)
    state.connection.commit()
    return selected[:-1]


def legacy_assignment_rows(db, scans):
    from stash_ai.recognition.database import encoded
    return db.execute('SELECT o.id,c.person,c.status,CASE WHEN json_valid(c.payload) THEN '
        "json_extract(c.payload,'$.reason','$.strict_confirmation') END FROM observations o "
        'LEFT JOIN candidates c ON c.rowid=(SELECT z.rowid FROM candidates z WHERE '
        "CASE WHEN json_valid(z.payload) THEN json_extract(z.payload,'$.observation') END=o.id "
        'ORDER BY z.rowid DESC LIMIT 1) WHERE o.scan IN (SELECT value FROM json_each(?)) ORDER BY o.id',
        (encoded(scans),))


@pytest.mark.parametrize('pending', [False, True])
def test_assignment_index_preserves_latest_global_address_and_caller_transaction(tmp_path, pending):
    from stash_ai.recognition.rebuild_cache import _assignment_index
    from stash_ai.recognition.database import encoded
    config = setup(tmp_path)
    with FaceDatabase(config.database) as state:
        selected = assignment_fixture(state)
        db = state.connection
        if pending:
            db.execute("INSERT INTO metadata VALUES ('caller-pending','keep')")
        before = list(db.iterdump())
        scans = [s['scan'] for s in selected]
        expected = [tuple(r) for r in legacy_assignment_rows(db, scans)]
        with _assignment_index(db, selected) as name:
            actual = [tuple(r) for r in db.execute(
                'SELECT o.id,c.person,c.status,CASE WHEN json_valid(c.payload) THEN '
                "json_extract(c.payload,'$.reason','$.strict_confirmation') END FROM observations o "
                f'LEFT JOIN {name} a ON a.observation=o.id LEFT JOIN candidates c ON c.rowid=a.candidate '
                'WHERE o.scan IN (SELECT value FROM json_each(?)) ORDER BY o.id', (encoded(scans),))]
            assert db.execute(f'SELECT COUNT(*) FROM {name}').fetchone()[0] == 6
        assert actual == expected and any(r[1] is not None for r in actual)
        assert db.in_transaction == pending
        assert list(db.iterdump()) == before
        assert not db.execute("SELECT 1 FROM sqlite_temp_master WHERE name LIKE 'rebuild_assignments_%'").fetchone()
        if pending:
            db.rollback()
            assert not db.execute("SELECT 1 FROM metadata WHERE key='caller-pending'").fetchone()


def test_assignment_catalogue_keeps_frozen_digest_with_multiple_candidate_policies(tmp_path):
    from stash_ai.recognition.rebuild_cache import input_context
    config = changed(setup(tmp_path), anchor_view_matching=True)
    with FaceDatabase(config.database) as state:
        selected = assignment_fixture(state)
        assert input_context(state.connection, selected, 'scope', 'ns', config) == 'b870fd8dd4755791c6c768c44ed6d1f7b8c4e1ea857aab48244affc799ef24eb'


def test_assignment_lookup_never_scans_the_global_candidate_table(tmp_path):
    from stash_ai.recognition.rebuild_cache import input_context
    config = changed(setup(tmp_path), anchor_view_matching=True)
    with FaceDatabase(config.database) as state:
        selected = assignment_fixture(state, scenes=259, candidates=1800)
        statements = []
        state.connection.set_trace_callback(statements.append)
        assert input_context(state.connection, selected, 'scope', 'ns', config)
        scans = [sql for sql in statements if sql.startswith('SELECT c.rowid,o.id,o.scan')]
        assert not scans
        assert sum('SELECT z.rowid' in sql for sql in statements)==len(selected)
        statements.clear()
        assert input_context(state.connection,selected,'scope','ns',config)
        assert not any('SELECT z.rowid' in sql for sql in statements)


def test_assignment_index_bounds_sql_work_before_validation(tmp_path):
    from stash_ai.recognition.rebuild_cache import input_context
    config = changed(setup(tmp_path), anchor_view_matching=True)
    with FaceDatabase(config.database) as state:
        selected = assignment_fixture(state, scenes=251, candidates=6000)
        calls = [0]
        def budget():
            calls[0] += 1
            return calls[0] > 1000  # One million VM instructions, independent of wall time.
        state.connection.set_progress_handler(budget, 1000)
        try:
            assert input_context(state.connection, selected, 'scope', 'ns', config)
        finally:
            state.connection.set_progress_handler(None, 0)
        assert calls[0] < 1000
