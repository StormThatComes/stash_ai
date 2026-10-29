"""Set-local diagnosis must produce globally validated, replayable changes."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.fragmentation_report import _open_readonly
from stash_ai.recognition.merge_goals import resolve,evaluate
from stash_ai.recognition.merge_report_policy import patched
from stash_ai.recognition.merge_solver import solve
from stash_ai.recognition.merge_solver_targeted import TargetedSearch,plan
from tests.test_recognition_merge_report import seeded


def check_winner(c,goals,db,result):
    changes={r['parameter']:r['proposed'] for r in result['best_result']['changes']}
    assert evaluate(patched(c,changes),goals,db)['trials']['baseline']['goal_result']['fully_solved']
    return changes


def test_measured_face_bridge_precedes_liberal_floor_and_explains_members(tmp_path):
    c,_=seeded(tmp_path)
    before=c.database.read_bytes()
    with _open_readonly(c.database) as db:
        goals=resolve(db,c,[['900','901']])
        result=solve(c,goals,db,maximum=4)
        patch=check_winner(c,goals,db,result)
    trial=result['simulation']['trials']['trial-1']
    assert trial['search_phase']=='targeted' and trial['search_strategy']=='face'
    assert .69<patch['recognition.match_threshold']<=.7
    assert result['feasibility']['probes_completed']==0
    assert set(trial['search_target']['performers'])=={'900','901'}
    diagnosis=result['simulation']['trials']['baseline']['solver_feedback']['targeted']['sets'][0]
    assert len(diagnosis['fragments'])==2 and len(diagnosis['missing_target_scenes'])==1
    assert not diagnosis['satisfied'] and len(diagnosis['performers'])==2
    assert c.database.read_bytes()==before
    json.dumps(result,allow_nan=False)


def test_query_clears_joint_face_body_and_support_gates_in_one_replay(tmp_path):
    from tests.test_recognition_merge_solver_hybrid import prepared
    from tests.test_recognition_merge_solver_guided import make_auxiliary
    face,body=.431237,.967123
    c,people=prepared(tmp_path,[[1.,0,0,0],[face,np.sqrt(1-face**2),0,0]],
        [[1.,0],[body,np.sqrt(1-body**2)]],matching=dict(face_floor=.8,person_threshold=.99,min_support=4))
    goals=make_auxiliary(c,people)
    with _open_readonly(c.database) as db:
        result=solve(c,goals,db,maximum=2)
        changes=check_winner(c,goals,db,result)
    trial=result['simulation']['trials']['trial-1']
    assert trial['search_phase']=='targeted' and trial['search_strategy']=='query'
    assert set(changes)=={'sapiensid.matching.'+k for k in ('face_floor','person_threshold','min_support')}
    assert changes['sapiensid.matching.min_support']==3
    assert {'face_floor','person_threshold','query_support'}<=set(trial['search_target']['blockers'])


def test_replan_after_admission_builds_missing_references_then_merges(tmp_path):
    c,_=seeded(tmp_path,clustering=dict(min_anchor_face_pixels=100,min_auxiliary_face_pixels=100))
    with _open_readonly(c.database) as db:
        goals=resolve(db,c,[['900','901']])
        result=solve(c,goals,db,maximum=9)
        check_winner(c,goals,db,result)
    trials=result['simulation']['trials']
    assert trials['trial-1']['search_strategy']=='admission'
    assert trials['trial-2']['search_phase']=='targeted'
    assert trials['trial-2']['search_parent']=='trial-1'
    assert trials['trial-2']['goal_result']['fully_solved']
    assert result['feasibility']['probes_completed']==0


def test_body_admission_can_recover_withheld_members_from_frozen_addresses(tmp_path):
    from tests.test_recognition_body_integration import seeded as bodies
    from tests.test_recognition_person_anchor import measured,PIXEL_PARAMETER
    c,_,ids,_=bodies(tmp_path,vectors=[[1,0]])
    with FaceDatabase(c.database) as state:measured(state.connection,ids[0],[95]*3)
    c.data['sapiensid']['identities']['anchor_min_person_pixels']=96
    with _open_readonly(c.database) as db:
        goals=resolve(db,c,[['900']],policy='strict')
        result=solve(c,goals,db,maximum=2)
        changes=check_winner(c,goals,db,result)
    assert changes=={PIXEL_PARAMETER:95}
    assert result['simulation']['trials']['trial-1']['search_strategy']=='admission'


def test_large_set_solves_without_initial_overmerge(tmp_path,monkeypatch):
    from tests.test_recognition_solver_bidirectional import plateau
    c,goals=plateau(tmp_path,monkeypatch,size=20)
    with _open_readonly(c.database) as db:
        result=solve(c,goals,db,maximum=5)
        check_winner(c,goals,db,result)
    first=result['simulation']['trials']['trial-1']
    assert first['search_phase']=='targeted' and first['goal_result']['fully_solved']
    assert first['goal_result']['disjoint'] and result['feasibility']['probes_completed']==0


def test_global_negative_scene_constraint_rejects_locally_good_merge(tmp_path):
    gram=np.asarray([[1.,.7,.8],[.7,1.,.8],[.8,.8,1.]])
    c,_=seeded(tmp_path,np.linalg.cholesky(gram).tolist())
    with _open_readonly(c.database) as db:
        goals=resolve(db,c,[['900','901']],scene_exclusions=[('900','3')])
        result=solve(c,goals,db,maximum=2)
    trials=result['simulation']['trials']
    assert trials['baseline']['goal_result']['admissible']
    assert trials['trial-1']['search_phase']=='targeted'
    assert not trials['trial-1']['goal_result']['scene_exclusions_satisfied']
    assert result['selected_trial']=='baseline'


def test_connected_pair_graph_is_not_a_production_merge_certificate(tmp_path):
    gram=np.asarray([[1.,.8,.3],[.8,1.,.8],[.3,.8,1.]])
    c,_=seeded(tmp_path,np.linalg.cholesky(gram).tolist())
    with _open_readonly(c.database) as db:
        goals=resolve(db,c,[['900','901','902']])
        result=solve(c,goals,db,maximum=2)
    first=result['simulation']['trials']['trial-1']
    assert first['search_phase']=='targeted'
    assert first['search_strategy']=='face_set_bridges'
    assert not first['goal_result']['fully_solved']
    assert not result['search']['proves_infeasibility']


def test_unrepresented_weak_performer_is_not_an_extra_merge_obligation():
    from tests.test_recognition_merge_identity_policy import goals,member,assignments
    from stash_ai.recognition.config import defaults
    goal=goals([member('a',scenes=['s1']*3),member('b',scenes=['s2']*3),
                member('weak',core=0,soft=4,scenes=['s1','s1','s2','s2'])])
    membership={oid:None if owner=='weak' else 'one' for oid,owner in assignments(goal).items()}
    frozen=SimpleNamespace(groups=[dict(evidence=goal['groups'][0]['identity_evidence'],addresses=set(membership))])
    result=plan(None,SimpleNamespace(data=defaults()),goal,frozen,membership,
        dict(quality=[dict(observation='weak-0',patch={'clustering.max_anchor_pose_deviation':.9})]),{})
    assert result['sets'][0]['satisfied'] and not result['candidates']
    assert next(m for m in result['sets'][0]['performers'] if m['performer']=='weak')['assigned']==0


def test_missing_body_can_use_measured_face_queries_without_fabricating_features(tmp_path):
    from benchmarks.benchmark_merge_solver import fixture
    c,goals=fixture(tmp_path,'missing_body')
    with _open_readonly(c.database) as db:
        result=solve(c,goals,db,maximum=6)
        changes=check_winner(c,goals,db,result)
    assert not any(k.startswith('sapiensid.') for k in changes)
    assert result['search']['targeted']['executed']>0
    assert all(q.get('person') is None for q in
        result['simulation']['trials']['baseline']['solver_feedback']['query']['queries'])


def test_absent_targeted_route_falls_through_to_feasibility(tmp_path,monkeypatch):
    import stash_ai.recognition.merge_solver_targeted as targeted
    c,_=seeded(tmp_path)
    original=targeted.plan
    def no_routes(*a,**kw):
        return dict(original(*a,**kw),candidates=[])
    monkeypatch.setattr(targeted,'plan',no_routes)
    with _open_readonly(c.database) as db:
        goals=resolve(db,c,[['900','901']])
        result=solve(c,goals,db,maximum=2)
        check_winner(c,goals,db,result)
    assert result['search']['targeted']['executed']==0
    assert result['simulation']['trials']['trial-1']['search_phase']=='feasibility'


def test_set_rotation_survives_replanning_from_an_improved_parent():
    lane=TargetedSearch(8)
    feedback=dict(targeted=dict(candidates=[dict(set=i,route='face',patch={'recognition.match_threshold':.7-i/10})
                                           for i in (0,0,1,2)]))
    assert lane.next('baseline',feedback,{})['set']==0
    assert lane.next('child',feedback,{})['set']==1
    assert lane.next('child',feedback,{})['set']==2
    lane.executed=8
    assert lane.next('child',feedback,{}) is None


def test_checkpoint_restores_targeted_order_and_rejects_budget_changes(tmp_path,monkeypatch):
    import stash_ai.recognition.replay_simulation as replay
    c,_=seeded(tmp_path,clustering=dict(min_anchor_face_pixels=100,min_auxiliary_face_pixels=100))
    checkpoint=tmp_path/'solver.json'
    def interrupt(name,stage,item):
        if stage=='complete' and name=='trial-1':raise KeyboardInterrupt
    with _open_readonly(c.database) as db:
        goals=resolve(db,c,[['900','901']])
        first=solve(c,goals,db,maximum=9,checkpoint=checkpoint,progress=interrupt)
        assert first['interrupted'] and first['search']['targeted']['executed']==1
        resumed=solve(c,goals,db,maximum=9,checkpoint=checkpoint,resume=True)
        assert resumed['search']['restored_trials']==2 and resumed['best_result']['fully_solved']
        monkeypatch.setattr(replay,'rebuild_selected',lambda *a,**k:pytest.fail('Completed replay must be reused'))
        repeated=solve(c,goals,db,maximum=9,checkpoint=checkpoint,resume=True)
        assert repeated['simulation']['trials']==resumed['simulation']['trials']
        with pytest.raises(ValueError,match='incompatible'):
            solve(c,goals,db,maximum=9,checkpoint=checkpoint,resume=True,targeted_trials=0)


@pytest.mark.parametrize('maximum',[1,2,4,12])
def test_targeted_budget_respects_total_and_opt_out(tmp_path,maximum):
    c,_=seeded(tmp_path)
    with _open_readonly(c.database) as db:
        goals=resolve(db,c,[['900','901']])
        result=solve(c,goals,db,maximum=maximum)
        disabled=solve(c,goals,db,maximum=maximum,targeted_trials=0)
    assert result['search']['attempted']<=maximum
    assert result['search']['targeted']['executed']<=result['search']['targeted']['budget']
    assert disabled['search']['targeted']['executed']==0
    if maximum>1:assert disabled['simulation']['trials']['trial-1']['search_phase']=='feasibility'


@pytest.mark.parametrize('style',['auto','off'])
def test_cli_logs_target_and_preserves_json(tmp_path,capsys,style):
    from stash_ai.recognition.cli import main,parser
    c,_=seeded(tmp_path)
    args=['report-merge','--face-config',str(c.source_path),'--faces-db',str(c.database),
          '--performer-set','900,901','--solve','--max-trials','2','--progress-style',style]
    assert main(args)==0
    output=capsys.readouterr();result=json.loads(output.out)
    assert result['status']=='fully_solved' and result['search']['targeted']['executed']==1
    if style=='off':assert output.err==''
    else:assert 'Search: phase=targeted set=1 route=face performers=' in output.err
    with pytest.raises(SystemExit):parser().parse_args(args+['--targeted-trials','-1'])
