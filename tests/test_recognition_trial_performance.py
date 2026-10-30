"""Exact replay parity with bounded diagnostics and shared immutable trial inputs."""
from contextlib import closing
from dataclasses import replace
from types import SimpleNamespace
import pickle
import sqlite3
import numpy as np
import pytest

from benchmarks.benchmark_merge_solver import fixture
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.fragmentation_report import _open_readonly
from stash_ai.recognition.models import namespace
from stash_ai.recognition.offline_evaluation import _ram
from stash_ai.recognition.replay_simulation import prepared_history, replay
from stash_ai.recognition.trial_inputs import prepare


@pytest.mark.parametrize('case',['face','query','body_modes'])
def test_shared_inputs_match_full_backup_replay(tmp_path,monkeypatch,case):
    config,_=fixture(tmp_path,case)
    before=config.database.read_bytes()
    outputs=[]
    def run():
        result={}
        replay(config,{'second':config},consume=lambda n,o,r,p:result.update({n:p}) or {})
        outputs.append(result)
    run()
    monkeypatch.setattr('stash_ai.recognition.trial_inputs.prepare',lambda target,seed:seed.backup(target))
    run()
    assert outputs[0]==outputs[1]
    assert config.database.read_bytes()==before


def test_trial_keeps_embeddings_on_shared_disk_and_derived_state_private(tmp_path):
    config,goals=fixture(tmp_path,'body_modes')
    with _open_readonly(config.database) as source:
        installation=source.execute("SELECT value FROM metadata WHERE key='installation'").fetchone()[0]
        with prepared_history(source,config,goals['scope'],namespace(config),installation) as seed:
            with closing(_ram(config)) as a, closing(_ram(config)) as b:
                prepare(a,seed);prepare(b,seed)
                for table in ('observations','body_observations'):
                    assert a.execute(f'SELECT COUNT(*) FROM main.{table} WHERE embedding IS NOT NULL').fetchone()[0]==0
                    assert a.execute(f'SELECT COUNT(*) FROM {table} WHERE embedding IS NOT NULL').fetchone()[0]==seed.execute(f'SELECT COUNT(*) FROM {table} WHERE embedding IS NOT NULL').fetchone()[0]
                assert a.execute('SELECT COUNT(*) FROM main.person_evidence').fetchone()[0]==0
                before=list(b.execute('SELECT * FROM members'))
                a.execute('DELETE FROM members');a.commit()
                assert [tuple(r) for r in b.execute('SELECT * FROM members')]==[tuple(r) for r in before]
                with pytest.raises(sqlite3.OperationalError): a.execute('UPDATE observations SET embedding=NULL')
                with pytest.raises(sqlite3.OperationalError): a.execute('UPDATE retained.observations SET embedding=NULL')
                assert not a.execute('PRAGMA foreign_key_check').fetchall()


def test_diagnostic_budget_reports_incomplete_coverage_without_quadratic_walk(tmp_path):
    from stash_ai.recognition.merge_solver_measurements import diagnostic_pairs
    config,_=fixture(tmp_path,'face')
    config.data['sapiensid']['execution'].update(plan_bytes=10*8192,max_witness_comparisons=12)
    groups=[SimpleNamespace(identifier=str(i),samples=[np.ones(4)]*2) for i in range(10000)]
    pairs,coverage=diagnostic_pairs(groups,config)
    result=list(pairs)
    assert len(result)==3 and coverage['examined_pairs']==10
    assert coverage['total_pairs']==49995000 and coverage['witness_comparisons']==12
    assert not coverage['complete']
    assert len({g.identifier for pair in result for g in pair})==4


def test_shared_face_statistics_skip_repeat_numeric_measurement(tmp_path,monkeypatch):
    from stash_ai.recognition.solver_shared import Arrays,scope
    from stash_ai.recognition.merge_solver_measurements import measured_pairs
    data=np.array([1.,0.],dtype='<f4').tobytes()
    a=SimpleNamespace(samples=[np.frombuffer(data,dtype='<f4')])
    Arrays.prepare(tmp_path)
    with scope(tmp_path,10000): expected=list(measured_pairs([(a,a)]))[0][2]
    def unexpected(*a,**kw): raise AssertionError('recomputed immutable pair')
    monkeypatch.setattr('stash_ai.recognition.merge_solver_measurements.reference_stats',unexpected)
    with scope(tmp_path,10000): assert list(measured_pairs([(a,a)]))[0][2]==expected


def test_trial_task_does_not_serialize_fixed_corpus():
    from stash_ai.recognition.solver_trial import Request,split,restore
    request=Request('a',{},'seed','scope','ns','install',[],{str(i):'1' for i in range(10000)},
                    {'groups':[]},None,(),None,False,{}, {},None)
    context,task=split(request)
    assert len(pickle.dumps(task))*100<len(pickle.dumps(request))
    assert restore(pickle.loads(pickle.dumps(task)),context)==request


def test_body_report_cache_is_generation_scoped_and_returns_private_proofs(tmp_path,monkeypatch):
    from tests.test_recognition_body_scaling import fixture as body_fixture
    import stash_ai.recognition.body_global as body
    from stash_ai.recognition.run_cache import scope
    config,state=body_fixture(tmp_path,[[1,0],[.999,.044710]])
    original=body._report;calls=[]
    def measured(*a,**kw): calls.append(1);return original(*a,**kw)
    monkeypatch.setattr(body,'_report',measured)
    try:
        with scope(config):
            a=body.report(state.connection,config,'scope','ns'); expected=len(a['pairs']);a['pairs'].clear()
            assert len(body.report(state.connection,config,'scope','ns')['pairs'])==expected and len(calls)==1
            state.connection.execute("UPDATE body_observations SET quality=json_set(quality,'$.quality.sharpness',0)")
            body.report(state.connection,config,'scope','ns')
            assert len(calls)==2
            state.connection.rollback()
            body.report(state.connection,config,'scope','ns')
            assert len(calls)==2  # The original committed generation remains valid.
    finally:state.__exit__(None,None,None)


def test_diagnostic_plan_too_small_for_a_proof_reports_no_measurements(tmp_path):
    from stash_ai.recognition.merge_solver_measurements import diagnostic_pairs
    config,_=fixture(tmp_path,'face')
    config.data['sapiensid']['execution']['plan_bytes']=1
    pairs,coverage=diagnostic_pairs([SimpleNamespace(samples=[np.ones(4)])]*2,config)
    assert list(pairs)==[] and coverage['pair_limit']==0 and not coverage['complete']
