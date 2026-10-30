from contextlib import closing
from dataclasses import replace
import pytest

from stash_ai.recognition.config import plain
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.fragmentation_report import _open_readonly
from stash_ai.recognition.merge_goals import resolve, evaluate
from stash_ai.recognition.merge_report import build
from stash_ai.recognition.resource_errors import ResourceLimit


def test_body_handoff_remains_a_diagnostic_destination(tmp_path):
    from tests.test_recognition_solver_feasibility import body_handoff
    from stash_ai.recognition.merge_solver import rebuilt_goals
    from stash_ai.recognition.merge_solver_feedback import QueryFeedback
    config,faces,bodies=body_handoff(tmp_path)
    with FaceDatabase(config.database) as state:
        goals=resolve(state.connection,config,[['900']],policy='identity-physical')
        owner=state.connection.execute('SELECT person FROM members LIMIT 1').fetchone()[0]
        state.connection.execute('DELETE FROM members')
        assert rebuilt_goals(state.connection,goals)['groups'][0]['members'] == [dict(identity_id=owner)]
        assert all(QueryFeedback(goals).sets[oid] == 0 for oid in bodies)


def test_optional_body_failure_preserves_face_report_and_marks_coverage(tmp_path,monkeypatch):
    from tests.test_recognition_body_integration import seeded
    config,_,_,people=seeded(tmp_path)
    def limited(*args,**kwargs):raise ResourceLimit('test diagnostic plan limit')
    monkeypatch.setattr('stash_ai.recognition.body_global._report',limited)
    report=build(config,people[0],people[1])
    assert report['status']=='diagnostics_incomplete'
    assert not report['measurement_coverage']['independent_person']['complete']
    assert {p['route'] for p in report['paths']} >= {'strict_core','confirmed_views'}
    assert not report['recommendations']


def test_pair_does_not_hide_data_errors(tmp_path,monkeypatch):
    from tests.test_recognition_merge_report import seeded
    config,people=seeded(tmp_path)
    def corrupt(*args,**kwargs):raise ValueError('invalid embedding')
    monkeypatch.setattr('stash_ai.recognition.body_global.report',corrupt)
    with pytest.raises(ValueError,match='invalid embedding'):build(config,*people[:2])


def test_simulation_of_same_identity_keeps_success_status(tmp_path):
    from tests.test_recognition_merge_report import seeded
    config,people=seeded(tmp_path)
    report=build(config,people[0],people[0],simulate=True)
    assert report['status']=='already_merged'
    assert report['simulation']['trials']['baseline']['requested_pair']['fully_merged']


def test_query_witness_budget_truncates_only_diagnostics(tmp_path):
    from tests.test_recognition_merge_solver_exploration import auxiliary_goals
    from stash_ai.recognition.merge_solver_feedback import QueryFeedback
    from stash_ai.recognition.hybrid_query_observer import observe_queries
    config,goals=auxiliary_goals(tmp_path)
    observer=QueryFeedback(goals);original=observer.begin
    def limited(db,banks,settings,births):
        data=plain(settings.data);data['sapiensid']['execution']['max_witness_comparisons']=1
        return original(db,banks,replace(settings,data=data),births)
    observer.begin=limited
    with closing(_open_readonly(config.database)) as db:
        expected=evaluate(config,goals,db)['trials']['baseline']['goal_result']
        with observe_queries(observer):actual=evaluate(config,goals,db)['trials']['baseline']['goal_result']
    report=observer.finish(config)
    assert expected==actual
    assert report['witness_comparisons'] <= 1
    assert 'max_witness_comparisons' in report['truncation_reasons']
    assert not report['proposals']


def test_local_hint_budget_reserves_late_scenes_and_rotates_overflow(monkeypatch):
    from stash_ai.recognition import local_body_hints as hints
    goals=dict(groups=[dict(index=0,members=[dict(observations=[dict(id=str(i),scene=str(i)) for i in range(4)])])])
    monkeypatch.setattr(hints,'HINT_LIMIT',2)
    first=hints.Budget(goals);first.start({})
    early=first.allocation(dict(id='0'))[1];late=first.allocation(dict(id='1'))[1]
    early['hints']=early['hint_limit'];early['comparisons']=early['comparison_limit']
    assert late['hints']==0 and late['comparisons']==0
    assert sum(a['hint_limit'] for a in first.allocations.values())==2
    second=hints.Budget(goals);second.start({'0':1,'1':1})
    assert {key[1] for key in second.allocations}=={'2','3'}
