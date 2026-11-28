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


def test_local_screen_uses_real_replays_and_never_removes_a_proposal(tmp_path):
    from stash_ai.recognition.solver_local_screen import LocalScreen
    from stash_ai.recognition.offline_evaluation import _selection
    config,goals = fixture(tmp_path,'disjoint')
    first,other = goals['groups']
    subset = dict(goals,groups=[first])
    candidates = [{'recognition.match_threshold':v} for v in (.86,.87,.89,.91)]
    with _open_readonly(config.database) as db:
        _,scans = _selection(db,config,None,goals['scope'])
        available = scans+[dict(id=f'other{i}',scene=str(i+10)) for i in range(32)]
        screen = LocalScreen()
        ordered = list(screen.order(candidates,config,db,subset,available,
                                    [{'rival':other['members'][0]['identity_id']}]))
        assert len(screen.info['scenes'])==3 and screen.info['replays']==4
        assert sorted(p['recognition.match_threshold'] for p in ordered)==[.86,.87,.89,.91]


def test_local_order_rewards_independent_bridges_and_deduplicates_projections():
    from stash_ai.recognition.solver_local_screen import score
    labels = {str(i): (0, 'front', 'scene', i) for i in range(10)}
    labels.update(rear=(0,'rear','scene',11), body=(0,'front','scene',0))
    assert score(labels, {str(i):'owner' for i in range(10)})[1] == 0
    bridge = {'0':'owner','body':'owner','rear':'owner'}
    assert score(labels, bridge) == (0,1,2)
    assert score(labels, dict(bridge,body='other')) < score(labels,bridge)


def test_eighth_scene_can_supply_a_body_only_rival():
    from contextlib import closing
    import sqlite3
    from stash_ai.recognition.solver_local_screen import sample
    scans = [dict(id=str(i),scene=str(i)) for i in range(8)]
    members = [dict(identity_id='p'+s['id'],observations=[dict(id='f'+s['id'],scan=s['id'],cue=0)]) for s in scans]
    with closing(sqlite3.connect(':memory:')) as db:
        db.executescript('CREATE TABLE members(person,observation); CREATE TABLE observations(id,scan,cue);'
                         'CREATE TABLE body_members(person,observation,status); CREATE TABLE body_observations(id,scan,cue);'
                         "INSERT INTO body_members VALUES ('outside','b','accepted'); INSERT INTO body_observations VALUES ('b','7',1);")
        selected, labels = sample(db,scans,dict(groups=[dict(index=0,members=members)]),[])
        assert selected == scans
        assert labels['b'] == -1
