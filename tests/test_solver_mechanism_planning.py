"""Construction hints share production evidence rules and never choose winners."""
import json
from dataclasses import dataclass

import numpy as np
import pytest

from stash_ai.recognition.config import defaults
from stash_ai.recognition.mechanism_views import diagnose, for_goals
from stash_ai.recognition.mechanism_scenarios import plans, local_global_plans
from stash_ai.recognition.merge_solver_search import BeamSearch
from stash_ai.recognition.merge_report_policy import patched, parameter
from stash_ai.recognition.merge_solver_domains import normalize
from stash_ai.recognition.fragmentation_report import _open_readonly
from stash_ai.recognition.merge_goals import resolve, evaluate
from stash_ai.recognition.merge_solver import solve
from tests.test_recognition_merge_report import seeded


def row(key, seconds, *, owner='one', strict=True, fingerprint=None):
    quality = dict(pose_deviation=.1, pixels=80, sharpness=30)
    if fingerprint:
        quality['evidence_pixel_sha256'] = fingerprint
    return dict(id=key, scan='scan', scene='1', cue=int(seconds), seconds=seconds,
                geometry='[]', dimensions=2, embedding=np.array([1.,0.],'<f4').tobytes(),
                quality=json.dumps(quality), person=owner, strict_ok=strict, source='original')


def config():
    @dataclass
    class Config:
        data: dict
    data = defaults()
    data['clustering'].update(max_anchor_pose_deviation=.2, anchor_view_matching=True,
                              view_min_support=4, same_scene_min_support=4)
    return Config(data)


def test_view_plan_measures_support_and_composes_downstream_requirements():
    c = config()
    cohort, = diagnose(c,[row('a',0),row('b',1)])
    assert cohort['independent_support']==2 and cohort['required_support']==4
    assert cohort['rooted'] and not cohort['supported']
    assert cohort['blockers']==['independent_support']
    candidates = list(plans(c,[cohort],followups={'face':[{
        'clustering.view_cohesion_threshold':.6, 'recognition.match_threshold':.7,
        'clustering.cohesion_threshold':.6}]}))
    joint = next(p for p in candidates if p['strategy']=='view_then_face')
    assert joint['patch']['clustering.view_min_support']==2
    assert joint['patch']['clustering.same_scene_min_support']==2
    assert joint['patch']['clustering.view_cohesion_threshold']==.6
    assert joint['stages']==['admission','strict_core','view_support','global_face']
    assert joint['witnesses']==['a','b']


def test_repeated_pixels_and_timestamps_cannot_build_a_view():
    c = config()
    assert not diagnose(c,[row('a',0),row('b',0)])
    assert not diagnose(c,[row('a',0,fingerprint='a'*64),row('b',1,fingerprint='a'*64)])


@pytest.mark.parametrize('owner,strict', [(None,True),('one',False)])
def test_membership_and_unassigned_samples_do_not_certify_reference_authority(owner,strict):
    c = config()
    c.data['clustering'].update(view_min_support=2,same_scene_min_support=2)
    cohort, = diagnose(c,[row('a',0,owner=owner,strict=strict),row('b',1,owner=owner,strict=strict)])
    assert not cohort['rooted'] and not cohort['supported']
    assert 'strict_reference_authority' in cohort['blockers']


def test_missing_quality_remains_explicit_and_cannot_be_fabricated():
    c = config()
    rows = [row('a',0),row('b',1)]
    rows[0]['quality']='{}'
    cohort, = diagnose(c,rows,quality=[dict(observation='a',missing_measurements=['pose_deviation'])])
    assert 'pose_deviation' in cohort['blockers']
    assert not cohort['patch'] and not list(plans(c,[cohort]))


def test_joint_view_candidate_passes_real_replay_and_preserves_source(tmp_path):
    c,_ = seeded(tmp_path,count=2,clustering=dict(view_min_support=4,same_scene_min_support=4))
    before=c.database.read_bytes()
    with _open_readonly(c.database) as db:
        goals=resolve(db,c,[['900','901']])
        diagnosis=for_goals(db,c,goals,[])
        assert diagnosis['sampled']==4 and len(diagnosis['cohorts'])==2
        followups={'face':[{'recognition.match_threshold':.69,'clustering.cohesion_threshold':.69,
                            'clustering.view_match_threshold':.69,'clustering.view_cohesion_threshold':.69}]}
        joint=next(p for p in plans(c,diagnosis['cohorts'],followups=followups) if p['strategy']=='view_then_face')
        variant=patched(c,normalize(c,joint['patch']))
        result=evaluate(variant,goals,db)['trials']['baseline']['goal_result']
        assert result['fully_solved']
    assert c.database.read_bytes()==before


@pytest.mark.parametrize('unavailable', ['superseded', 'foreign_model', 'rejected', 'missing_embedding'])
def test_view_measurements_ignore_unavailable_retained_evidence(tmp_path, unavailable):
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.mechanism_views import load_rows
    from stash_ai.recognition.models import namespace
    c,_ = seeded(tmp_path,count=2)
    with FaceDatabase(c.database) as state:
        db = state.connection
        scan = dict(db.execute("SELECT * FROM scans WHERE scene='1'").fetchone())
        addresses = [r[0] for r in db.execute('SELECT id FROM observations WHERE scan=?',(scan['id'],))]
        assert len(load_rows(db,c,'scope',namespace(c),addresses))==2
        if unavailable == 'superseded':
            # Even an incomplete newer scan must prevent fallback to old evidence.
            scan.update(id='new-incomplete-scan',input_sha='new-input',status='pending')
            db.execute('INSERT INTO scans VALUES (?,?,?,?,?,?,?,?,?,?,?)',tuple(scan.values()))
        elif unavailable == 'foreign_model':
            db.execute("UPDATE scans SET namespace='foreign' WHERE id=?",(scan['id'],))
        elif unavailable == 'rejected':
            db.execute("UPDATE observations SET rejected='quality' WHERE scan=?",(scan['id'],))
        else:
            db.execute('UPDATE observations SET embedding=NULL WHERE scan=?',(scan['id'],))
        assert not load_rows(db,c,'scope',namespace(c),addresses)


def test_view_sampling_is_bounded_and_marks_unexamined_evidence(tmp_path):
    c,_ = seeded(tmp_path,count=12)
    with _open_readonly(c.database) as db:
        goals=resolve(db,c,[['900','901']])
        diagnosis=for_goals(db,c,goals,[])
    assert diagnosis['sampled']==16 and diagnosis['truncated']
    assert all(2 <= r['independent_support'] <= 8 for r in diagnosis['cohorts'])


def milestone(stage):
    return dict(cohorts=[dict(key='needed-profile',admitted=stage>=1,rooted=stage>=2,supported=stage>=3)])


def test_useful_intermediate_gets_descendants_without_becoming_the_winner():
    def factory(parent,changes,family,refine):
        yield 'mechanism_plan',dict(changes,next_step=True)
    beam=BeamSearch(['face'],factory,100)
    beam.observe('baseline',{},(1,0,0),'unchanged',mechanisms=milestone(1))
    beam.observe('constructed',{'view':True},(1,-1,-1),'unchanged',mechanisms=milestone(3))
    for i in range(10):
        beam.observe(f'unrelated-{i}',{'x':i},(1,0,-i),'unrelated-'+str(i))
    assert beam.incumbent=='baseline' and 'constructed' in beam.beam
    family,phase,parent,_,changes=beam.next()
    assert parent=='constructed' and changes['next_step']
    beam.completed(phase,parent=parent)
    assert beam.mechanism_parents=={'constructed'}
    assert beam.best_quality==(1,) and len(beam.beam)<=beam.width


def test_unsafe_or_unchanged_milestone_does_not_get_a_reserved_slot():
    beam=BeamSearch(['face'],lambda *a:iter(()),100)
    beam.observe('baseline',{},(1,0,0),'same',mechanisms=milestone(2))
    beam.observe('unsafe',{'x':1},(0,0,0),'same',mechanisms=milestone(3),admissible=False)
    beam.observe('same',{'x':2},(1,-1,-1),'same',mechanisms=milestone(2))
    unseen = milestone(3)
    unseen['cohorts'][0]['key'] = 'new-sampling-window'
    beam.observe('unseen',{'x':3},(1,-1,-1),'same',mechanisms=unseen)
    assert not beam.mechanism_due
    assert not beam.beam.get('unsafe',{}).get('mechanisms')


def test_local_face_bonus_respects_shared_cohesion_and_global_floor():
    from stash_ai.recognition.mechanism_scenarios import local_face_routes
    c=config()
    c.data['recognition']['match_threshold']=.9
    c.data['clustering']['cohesion_threshold']=.8
    candidates=list(local_face_routes(c,[{'recognition.match_threshold':.87},
                                         {'recognition.match_threshold':.7}]))
    candidate, = candidates
    bonus = candidate['clustering.same_scene_match_bonus']
    assert .03 <= bonus < .030002 and .9-bonus <= .87


def test_inspector_and_solver_share_local_global_recipe_and_fixed_global_floors(tmp_path):
    from tests.test_recognition_body_identity import configuration
    from stash_ai.recognition.inspection_merge_strategies import local_global
    from stash_ai.recognition.inspection_pair_variants import patch_values
    c=configuration(tmp_path)
    inputs={'local_face':[{}],'local_body':[{}]}
    common=list(local_global_plans(c,inputs))
    report=dict(identities=[dict(reference_samples=['a'],person_members=1)]*2)
    inspected=local_global(c,report,[])['patches']
    assert len(inspected)==len(common)==3
    for expected,actual in zip(common,inspected):
        values={k:v for k,v in expected['patch'].items() if v!=parameter(c.data,k)}
        assert patch_values(actual)==values
        assert not any('threshold' in k for k in values)
        patched(c,normalize(c,values))


def test_scene_strategy_cli_no_longer_requires_a_holdout():
    from stash_ai.recognition.cli import parser
    args=['report-merge','--performer-set','1,2','--compare-scene-strategies']
    assert parser().parse_args(args+['--solve']).solver_validation is None
    with pytest.raises(SystemExit):parser().parse_args(args)


@pytest.mark.parametrize('enabled', [True, False])
def test_yaml_strategy_screening_runs_without_holdout_and_respects_opt_out(tmp_path, monkeypatch, enabled):
    from tests.test_recognition_solver_feasibility import body_handoff
    from stash_ai.recognition import merge_solver
    from stash_ai.recognition.merge_goals import execute
    from stash_ai.recognition.cli import parser
    c,_,_ = body_handoff(tmp_path)
    c.data['execution']['solver_compare_scene_strategies'] = enabled
    observed = []
    def recorded(*args, **kwargs):
        observed.append(kwargs['compare_scene_strategies'])
        return solve(*args, **kwargs)
    monkeypatch.setattr(merge_solver, 'solve', recorded)
    output = tmp_path/'report.json'
    args = parser().parse_args(['report-merge','--performer-set','900','--solve',
                                '--max-trials','1','--progress-style','off','--output',str(output)])
    assert execute(args,c)==0
    assert observed==[enabled]
    assert 'holdout_validation' not in json.loads(output.read_text())


def test_tightening_preserves_categorical_mechanism_choices(tmp_path):
    from stash_ai.recognition.merge_solver_tighten import Tightener
    c,_=seeded(tmp_path)
    choices={'clustering.context_view_enrichment':True,'clustering.auxiliary_matching':True,
             'clustering.view_prototypes_enabled':True,'recognition.match_threshold':.7,
             'clustering.cohesion_threshold':.7}
    tightening=Tightener(c)
    tightening.begin('constructed',choices,{'matching_parameters':{'recognition.match_threshold':1}})
    proposal=tightening.candidate(choices,('recognition.match_threshold',),{})
    assert proposal and proposal['clustering.context_view_enrichment']
    assert proposal['recognition.match_threshold']>.7


def test_strict_allows_formation_order_comparison_without_disabling_tracks(tmp_path):
    from tests.test_recognition_body_identity import configuration
    from stash_ai.recognition.solver_config_policy import StrictPolicy
    c=configuration(tmp_path)
    c.data['sapiensid']['identities'].update(body_first_matching=True,tracklet_matching=True,
                                          match_threshold=.9,track_similarity_threshold=.8)
    policy=StrictPolicy(c)
    variant=patched(c,normalize(c,{'sapiensid.identities.body_first_matching':False}))
    assert variant.data['sapiensid']['identities']['turn_recovery'] is True
    assert not policy.violations(variant)
    assert 'sapiensid.identities.tracklet_matching' in policy.violations(
        patched(c,{'sapiensid.identities.tracklet_matching':False}))


def test_mechanism_trials_use_existing_budget_and_emit_explanations(tmp_path):
    c,_=seeded(tmp_path,count=2,clustering=dict(view_min_support=4,same_scene_min_support=4))
    with _open_readonly(c.database) as db:
        goals=resolve(db,c,[['900','901']])
        result=solve(c,goals,db,maximum=6,targeted_trials=0,feasibility_trials=0)
    assert result['search']['attempted']<=6
    initial=result['simulation']['trials']['baseline']['solver_feedback']['mechanisms']
    assert initial['cohorts'] and initial['plans']
    json.dumps(result,allow_nan=False)


def test_mechanism_operator_executes_and_exports_a_replayable_solution(tmp_path, monkeypatch):
    from stash_ai.recognition.merge_solver_bidirectional import BidirectionalSearch
    # Isolate this proposal lane; feedback, scheduling, constraints, replay and
    # final verification are real. Easier fixtures otherwise solve via old lanes.
    monkeypatch.setattr(BeamSearch, 'operations', lambda self,streams: iter(streams['mechanism_plan']))
    monkeypatch.setattr(BidirectionalSearch, 'next', lambda *a,**k: None)
    c,_=seeded(tmp_path,count=2,clustering=dict(view_min_support=4,same_scene_min_support=4,
                                              min_anchor_face_pixels=100))
    with _open_readonly(c.database) as db:
        goals=resolve(db,c,[['900','901']])
        result=solve(c,goals,db,maximum=12,targeted_trials=0,feasibility_trials=0)
        trials=result['simulation']['trials']
        executed=[t for t in trials.values() if t.get('search_strategy')=='mechanism_plan']
        assert executed and all(t.get('search_mechanism',{}).get('stages') for t in executed)
        assert result['best_result']['fully_solved']
        patch={r['parameter']:r['proposed'] for r in result['recommendation']['changes']}
        assert evaluate(patched(c,patch),goals,db)['trials']['baseline']['goal_result']['fully_solved']
    assert result['search']['attempted']<=12
