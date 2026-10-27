import builtins
import pytest
from benchmarks.benchmark_merge_solver import fixture
from stash_ai.recognition.merge_solver_sampler import create
from stash_ai.recognition.merge_solver import solve
from stash_ai.recognition.fragmentation_report import _open_readonly


def test_optional_import_and_clear_dependency_error(monkeypatch):
    original=builtins.__import__
    def missing(name,*args,**kwargs):
        if name=='optuna':raise ImportError('not installed')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',missing)
    assert create('heuristic',0) is None
    with pytest.raises(ValueError,match='requirements-solver'):create('tpe',0)
    with pytest.raises(ValueError,match='seed'):create('heuristic',-1)


def test_tpe_proposals_reproduce_with_seed_and_completed_history(tmp_path):
    pytest.importorskip('optuna')
    config,_=fixture(tmp_path,'query');key='sapiensid.matching.face_floor'
    options={key:[.2,.5,.8],'sapiensid.matching.min_support':[1,2,4]}
    history=[({key:i/20},(int(i>5),i)) for i in range(12)]
    first=create('tpe',42).suggest(config,history,options,{key:2},15)
    second=create('tpe',42).suggest(config,history,options,{key:2},15)
    assert first==second and 0<=first[key]<=1
    assert type(first['sapiensid.matching.min_support']) is int


def test_tpe_uses_full_replay_and_checkpoint_reconstructs_proposals(tmp_path,monkeypatch):
    pytest.importorskip('optuna')
    config,goals=fixture(tmp_path/'input','disjoint');path=tmp_path/'solver.json'
    def stop(name,stage,item):
        if name=='trial-16' and stage=='complete':raise KeyboardInterrupt
    with _open_readonly(config.database) as db:
        first=solve(config,goals,db,maximum=25,sampler='tpe',seed=7,checkpoint=path,progress=stop)
    assert first['interrupted']
    assert any(t.get('search_strategy')=='tpe' for t in first['simulation']['trials'].values())
    import stash_ai.recognition.replay_simulation as replay
    original=replay.rebuild_selected;calls=[]
    def counting(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(replay,'rebuild_selected',counting)
    with _open_readonly(config.database) as db:
        result=solve(config,goals,db,maximum=25,sampler='tpe',seed=7,checkpoint=path,resume=True)
    assert result['search']['restored_trials']==17
    assert len(calls)==len(result['simulation']['trials'])-17
    assert result['best_result']['admissible']
    for name,item in first['simulation']['trials'].items():assert item==result['simulation']['trials'][name]
    with _open_readonly(config.database) as db:
        with pytest.raises(ValueError,match='incompatible'):
            solve(config,goals,db,maximum=25,sampler='tpe',seed=8,checkpoint=path,resume=True)


def test_checkpoint_resumes_across_process_hash_seeds(tmp_path):
    pytest.importorskip('optuna')
    import json,os,subprocess,sys,yaml
    from stash_ai.recognition.config import plain
    config,goals=fixture(tmp_path/'input','disjoint')
    config.source_path.write_text(yaml.safe_dump(plain(config.data)))
    goal_path=tmp_path/'goals.json';goal_path.write_text(json.dumps(goals))
    script='''
import sys,json
from pathlib import Path
from stash_ai.recognition.config import load
from stash_ai.recognition.fragmentation_report import _open_readonly
from stash_ai.recognition.merge_solver import solve
config=load(Path(sys.argv[1]),database=Path(sys.argv[2]))
goals=json.loads(Path(sys.argv[3]).read_text())
resume=sys.argv[5]=='resume'
def progress(name,stage,item):
    if not resume and name=='trial-30' and stage=='complete':raise KeyboardInterrupt
with _open_readonly(config.database) as db:
    result=solve(config,goals,db,maximum=45,sampler='tpe',seed=42,checkpoint=Path(sys.argv[4]),resume=resume,progress=progress)
print(json.dumps(dict(restored=result['search']['restored_trials'],interrupted=result['interrupted'],executed=result['search']['executed'])))
'''
    command=[sys.executable,'-c',script,str(config.source_path),str(config.database),str(goal_path),str(tmp_path/'checkpoint.json')]
    first=subprocess.run(command+['start'],env=dict(os.environ,PYTHONHASHSEED='7'),capture_output=True,text=True,check=True)
    assert json.loads(first.stdout)['interrupted']
    second=subprocess.run(command+['resume'],env=dict(os.environ,PYTHONHASHSEED='999'),capture_output=True,text=True,check=True)
    result=json.loads(second.stdout)
    assert result['restored']==31 and not result['interrupted'] and result['executed']<=45
