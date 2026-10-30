import json
import pytest
from benchmarks.benchmark_merge_solver import fixture
from stash_ai.recognition.fragmentation_report import _open_readonly
from stash_ai.recognition.merge_solver import solve
from stash_ai.recognition.merge_solver_checkpoint import winner_file,AtomicFile
from stash_ai.recognition.merge_report_policy import patched


def test_resume_restores_completed_trials_and_live_yaml(tmp_path,monkeypatch):
    config,goals=fixture(tmp_path/'input','query')
    checkpoint=tmp_path/'solver.json';yaml=tmp_path/'winner.yml'
    def stop(name,stage,item):
        if stage=='complete' and name=='trial-2':
            from stash_ai.recognition.solver_journal_events import load_document
            assert yaml.exists() and load_document(checkpoint)['records'][-1]['file'].startswith('trial-2-')
            raise KeyboardInterrupt
    with _open_readonly(config.database) as db:
        first=solve(config,goals,db,maximum=7,checkpoint=checkpoint,winner_writer=winner_file(yaml),progress=stop)
    assert first['interrupted']
    import stash_ai.recognition.replay_simulation as replay
    original=replay.rebuild_selected;calls=[]
    def counting(*args,**kwargs):
        calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(replay,'rebuild_selected',counting)
    with _open_readonly(config.database) as db:
        resumed=solve(config,goals,db,maximum=7,checkpoint=checkpoint,resume=True,winner_writer=winner_file(yaml,resume=True))
    assert resumed['search']['restored_trials']==3 and len(calls)==4
    assert resumed['selected_trial'] in yaml.read_text()
    # Replay the entire journal without running production again.
    calls.clear()
    with _open_readonly(config.database) as db:
        repeated=solve(config,goals,db,maximum=7,checkpoint=checkpoint,resume=True,winner_writer=winner_file(yaml,resume=True))
    assert not calls and repeated['selected_trial']==resumed['selected_trial']
    assert repeated['simulation']['trials']==resumed['simulation']['trials']
    with _open_readonly(config.database) as db:
        with pytest.raises(ValueError,match='incompatible'):
            solve(patched(config,{'sapiensid.matching.face_floor':.79}),goals,db,maximum=7,
                  checkpoint=checkpoint,resume=True,winner_writer=winner_file(yaml,resume=True))


def test_atomic_output_preserves_previous_winner_on_failure(tmp_path,monkeypatch):
    target=tmp_path/'winner.yml';owner=AtomicFile(target);owner.write('old')
    def fail(*args):raise OSError('disk failure')
    monkeypatch.setattr('os.replace',fail)
    with pytest.raises(OSError):owner.write('new')
    assert target.read_text()=='old'
    target.write_text('user edit')
    with pytest.raises(FileExistsError):owner.write('new')


def test_resume_rejects_changed_retained_geometry(tmp_path):
    import sqlite3
    config,goals=fixture(tmp_path/'input','query');path=tmp_path/'solver.json'
    with _open_readonly(config.database) as db:solve(config,goals,db,maximum=1,checkpoint=path)
    with sqlite3.connect(config.database) as db:
        db.execute("UPDATE observations SET geometry='[1,2,3,4]'")
    with _open_readonly(config.database) as db:
        with pytest.raises(ValueError,match='incompatible'):solve(config,goals,db,maximum=1,checkpoint=path,resume=True)


def test_robustness_requires_multiple_full_confirmations():
    from stash_ai.recognition.merge_solver_robustness import confidence
    assert confidence([True,True])==1
    assert not confidence([]) and not confidence([True]) and not confidence([True,False])


def test_cli_resume_uses_owned_yaml_and_rejects_changed_yaml(tmp_path,capsys):
    import yaml
    from stash_ai.recognition.cli import main
    from stash_ai.recognition.config import plain
    config,goals=fixture(tmp_path/'input','query')
    config.source_path.write_text(yaml.safe_dump(plain(config.data)))
    checkpoint=tmp_path/'solver.json'
    argv=['report-merge','--solve','--max-trials','3','--checkpoint',str(checkpoint),
          '--face-config',str(config.source_path),'--faces-db',str(config.database),'--progress-style','off']
    for g in goals['groups']:argv+=['--performer-set',','.join(m['performer_id'] for m in g['members'])]
    assert main(argv)==0
    first=json.loads(capsys.readouterr().out)
    assert main(argv+['--resume'])==0
    repeated=json.loads(capsys.readouterr().out)
    assert first['selected_trial']==repeated['selected_trial']
    assert repeated['search']['restored_trials']==3
    destination=checkpoint.with_name(checkpoint.name+'.best.yml');destination.write_text('user edit')
    with pytest.raises(FileExistsError,match='changed outside'):main(argv+['--resume'])


def test_robustness_is_replayed_within_budget(tmp_path):
    config,goals=fixture(tmp_path/'input','query')
    with _open_readonly(config.database) as db:result=solve(config,goals,db,maximum=30)
    checks=[t for t in result['simulation']['trials'].values() if t.get('search_phase')=='robustness']
    assert checks and len(result['simulation']['trials'])<=30
    assert sum(map(len,result['robustness']['candidates'].values()))==len(checks)


def test_resume_before_baseline_cannot_adopt_unowned_yaml(tmp_path):
    from stash_ai.recognition.merge_solver_checkpoint import Journal
    path=tmp_path/'checkpoint.json';destination=tmp_path/'winner.yml'
    Journal(path,{'format':1},winner=winner_file(destination))
    destination.write_text('user document')
    with pytest.raises(FileExistsError,match='not owned'):
        Journal(path,{'format':1},resume=True,winner=winner_file(destination,resume=True))
    assert destination.read_text()=='user document'
