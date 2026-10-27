from dataclasses import replace
from tests.test_recognition_merge_solver_exploration import auxiliary_goals
from stash_ai.recognition.config import plain, digest
from stash_ai.recognition.fragmentation_report import _open_readonly
from stash_ai.recognition.merge_solver import solve
from stash_ai.recognition.merge_solver_feedback import QueryFeedback
from stash_ai.recognition.hybrid_query_observer import observe_queries
from stash_ai.recognition.merge_goals import evaluate


def test_fair_budget_measures_both_sets_before_exhaustion(tmp_path):
    config,goals = auxiliary_goals(tmp_path,sets=2)
    observer = QueryFeedback(goals)
    original = observer.begin
    def limited(db,banks,settings,births):
        data = plain(settings.data);data['sapiensid']['execution']['max_candidate_pairs'] = 4
        return original(db,banks,replace(settings,data=data),births)
    observer.begin = limited
    with _open_readonly(config.database) as db, observe_queries(observer):
        evaluate(config,goals,db)
    hints = observer.finish(config)
    assert {q['set'] for q in hints['queries']} == {0,1}
    assert hints['comparisons'] <= 4 and hints['truncated']


def test_completed_feedback_has_configuration_provenance_and_active_gates(tmp_path):
    config,goals = auxiliary_goals(tmp_path)
    with _open_readonly(config.database) as db:
        report = solve(config,goals,db,maximum=3)
    baseline = report['simulation']['trials']['baseline']['solver_feedback']
    provenance = baseline['provenance']
    assert provenance['configuration_sha256'] == digest(plain(config.data))
    assert provenance['trial'] == 'baseline'
    assert 'sapiensid.matching.face_floor' in provenance['active_parameters']
    assert any(r['stage']=='hybrid_query.choice' for r in provenance['blocker_trace'])
    assert report['simulation']['trials']['trial-1']['solver_feedback']['provenance']['trial'] == 'trial-1'


def test_benchmark_uses_real_replay_and_deterministic_evidence(tmp_path):
    from benchmarks.benchmark_merge_solver import fixture,measure
    config,goals = fixture(tmp_path/'a','query')
    other,other_goals = fixture(tmp_path/'b','query')
    assert goals == other_goals
    result = measure(config,goals,3)
    assert result['first_solution'] and result['history'][0]['trial']=='baseline'
    assert result['milestones']['10']['completed']==3
