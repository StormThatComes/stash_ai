from benchmarks.benchmark_merge_solver import fixture
from stash_ai.recognition.fragmentation_report import _open_readonly
from stash_ai.recognition.merge_goals import assess
from stash_ai.recognition.replay_simulation import replay
from stash_ai.recognition.merge_report_policy import patched
from stash_ai.recognition.merge_solver_preflight import cells, ordered


def test_prepared_history_matches_independent_copies(tmp_path,monkeypatch):
    config,goals=fixture(tmp_path,'disjoint')
    import stash_ai.recognition.replay_simulation as module
    original=module.copy_history
    calls=[]
    def copying(*args,**kwargs):
        calls.append(1)
        return original(*args,**kwargs)
    monkeypatch.setattr(module,'copy_history',copying)
    variants={'relaxed':patched(config,{'sapiensid.matching.face_floor':.4,'sapiensid.matching.min_support':3})}
    with _open_readonly(config.database) as db:
        before=list(db.execute('SELECT * FROM members'))
        consume=lambda n,o,r,p:dict(goals=assess(goals,o,r,p))
        shared=replay(config,variants,source=db,scope='scope',consume=consume)
        assert len(calls)==1
        separate=replay(config,variants,source=db,scope='scope',consume=consume,reuse_history=False)
        assert len(calls)==3 and before==list(db.execute('SELECT * FROM members'))
    for name in shared['trials']:
        assert shared['trials'][name]['goals']==separate['trials'][name]['goals']
        assert shared['trials'][name]['versus_baseline']==separate['trials'][name]['versus_baseline']


def test_preflight_orders_without_dropping_candidates(tmp_path):
    config,_=fixture(tmp_path,'query')
    key='sapiensid.matching.face_floor'
    feedback={'query':{'positive_witnesses':[{key:.7}], 'negative_witnesses':[{key:.3}], 'truncated':True}}
    proposals=[{key:.2},{key:.8},{key:.5}]
    result=list(ordered(config,proposals,feedback))
    assert result[0]=={key:.5} and len(result)==len(proposals)
    assert list(cells([.2,.4,.8]))==[.3,.6]
    assert list(cells([1,2,3,8],integer=True))==[5]
