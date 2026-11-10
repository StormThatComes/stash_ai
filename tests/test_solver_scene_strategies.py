import json
import sqlite3
from contextlib import closing
import numpy as np
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.scene_strategy_validation import omission_damage
from stash_ai.recognition.solver_validation import validate_winner
from stash_ai.recognition.models import namespace
from stash_ai.recognition.scanning import scan_identity
from stash_ai.recognition.physical_goal import explicit_pairs,freeze
from tests.test_recognition_body_identity import configuration,scene


def test_damage_counts_each_physical_person_once():
    labels={'face':'A','body':'A','turn':'A','partner':'B'}
    coords={'face':('s',0),'body':('s',0),'turn':('s',1),'partner':('s',1)}
    def result(values):return {k:dict(accepted_identity=v) for k,v in zip(labels,values)}
    assert omission_damage(labels,coords,labels,result('AAAB'),result('AABB'))==dict(lost_correct_pairs=1,new_false_joins=1, new_projection_conflicts=0, new_partner_claims=1)


def test_all_strategy_families_share_retained_observations(tmp_path):
    config = configuration(tmp_path)
    ns = namespace(config)
    with FaceDatabase(config.database) as state:
        db=state.connection
        item,ids=scene(db,config)
        db.execute('UPDATE scans SET namespace=?,policy_sha=?',(ns,scan_identity(config,ns)))
        db.commit()
    manifest = tmp_path/'holdout.json'
    manifest.write_text(json.dumps(dict(scenes=['1'],observations={i:dict(truth='A',slices=[]) for i in ids},
                                        series=[['1',0,0]])))
    with closing(sqlite3.connect(':memory:')) as training:
        training.execute('CREATE TABLE scans(scope,namespace,scene)')
        result=validate_winner(config,training,dict(scope='scope',namespace=ns,best_result=dict(changes=[])),
                               manifest,compare_strategies=True)
    strategies = result['scene_strategies']
    assert len(strategies)==16
    assert strategies['body-first']['body_first_matching']
    assert not strategies['face-first']['body_first_matching']
    from stash_ai.recognition.scene_strategy_validation import training_profiles
    profiles = dict(training_profiles(config))
    differing = {k for k,v in profiles['body-first'].items() if v!=profiles['face-first'][k]}
    assert differing=={'sapiensid.identities.body_first_matching'}
    assert result["verdict"]["status"]=="insufficient_coverage"
    assert result["verdict"]["slice_coverage"]["entrant"]==dict(observations=0,performers=0)
    assert all(r['fragments_per_performer']=={'A':1} for r in result['scene_strategies'].values())
    assert all(set(w['damage'])=={'baseline','winner',*result['stress_strategies']} for w in result['omissions'])


def test_explicit_physical_vote_can_repair_unlinked_baseline(tmp_path):
    config = configuration(tmp_path)
    with FaceDatabase(config.database) as state:
        db=state.connection
        item,ids=scene(db,config)
        db.execute('UPDATE scans SET namespace=?',(namespace(config),))
        db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                   ('face',item['scan'],0,0.,'','[]','[]','{}',None,2,np.array([1.,0.],dtype='<f4').tobytes(),None))
        pairs=explicit_pairs(db,config,'scope',['face:'+ids[0]])
        unit=dict(id='unit',observations=[dict(id='face',scene='1')])
        evidence=dict(core_units=[unit],soft_units=[],scenes={})
        member=dict(observations=[dict(id='face',scene='1')],identity_evidence=evidence)
        goals=dict(groups=[dict(members=[member],identity_evidence=evidence)])
        freeze(db,config,goals,pairs)
        assert len(evidence['core_units'])==1
        assert [r['id'] for r in unit['observations']]==['face',ids[0]]


def test_unlabelled_face_preserves_a_turn_and_ordinary_false_joins_regress():
    from stash_ai.recognition.solver_validation import _metrics, holdout_verdict
    from stash_ai.recognition.reference_roles import STRICT_REASONS
    labels = {'a0':'A','a1':'A','a2':'A','face1':'A','b0':'B','b1':'B','b2':'B'}
    locations = {k:('1',int(k[-1])) for k in labels}
    samples = {k:dict(slices=['back'] if k in ('a1','face1') else []) for k in labels}
    def measure(join):
        decisions = {k:dict(accepted_identity=None if k=='face1' else 'A' if join else owner,
                            evidence_kind='person' if k=='a1' else 'face',
                            evidence=dict(reason=sorted(STRICT_REASONS)[0])) for k,owner in labels.items()}
        return dict(_metrics(labels,samples,locations,decisions),visible={})
    good,bad = measure(False),measure(True)
    assert good['sequences']['own_id_turns']==1
    verdict = holdout_verdict({'baseline/joint':good,'winner/joint':bad})
    assert verdict['status']=='regressed'
    assert verdict['measures']['physical_false_merge_pairs']['winner']==9


def test_partner_target_recovers_from_body_when_labelled_face_is_withheld():
    from stash_ai.recognition.solver_validation import partner_metrics
    labels = dict(front='A',face='A',body='A',partner='B')
    coords = {k:('1',0 if k=='front' else 1) for k in labels}
    decisions = {k:dict(accepted_identity=None if k=='face' else truth) for k,truth in labels.items()}
    metrics = partner_metrics([('front','face','partner')],labels,coords,decisions,())
    assert metrics['recovered']==1 and metrics['false_joins']==0


def test_training_screens_and_tunes_two_families_within_solver_budget(tmp_path):
    from tests.test_recognition_solver_feasibility import body_handoff
    from stash_ai.recognition.fragmentation_report import _open_readonly
    from stash_ai.recognition.merge_goals import resolve
    from stash_ai.recognition.merge_solver import solve
    config,_,_ = body_handoff(tmp_path)
    with closing(_open_readonly(config.database)) as db:
        goals = resolve(db,config,[['900']],policy='identity-physical')
        result = solve(config,goals,db,maximum=20,compare_scene_strategies=True)
    # The face-first control reuses the existing tracks-with-evidence configuration.
    assert result['search']['scene_families_tested']==15
    assert len(result['search']['scene_finalists'])==2
    assert result['search']['attempted']<=20
    assert any(t.get('search_phase')=='scene_refinement' for t in result['simulation']['trials'].values())


def test_omission_detects_shared_partner_with_conflicting_projections():
    labels = dict(face='A', body='A', partner='B')
    locations = dict.fromkeys(labels, ('s', 7))
    before = {k: dict(accepted_identity=v) for k, v in labels.items()}
    after = dict(before, body=dict(accepted_identity='B'))
    damage = omission_damage(labels, locations, labels, before, after)
    assert damage['new_projection_conflicts'] == damage['new_partner_claims'] == 1

def test_context_controls_and_labelled_threshold_boundaries(tmp_path):
    from stash_ai.recognition.scene_strategy_validation import variants, measured_context
    cfg = configuration(tmp_path)
    rows = list(variants(cfg).values())[:4]
    assert {(v.data['clustering']['same_scene_match_bonus']>0,
             v.data['sapiensid']['identities']['scene_match_bonus']>0) for v in rows} == {
                 (a,b) for a in (False,True) for b in (False,True)}
    probes = [dict(observation=k,scene_probe=dict(owner='A',score=v,runner=r,weight=1.))
              for k,v,r in [('return',.82,.65),('entrant',.83,.75)]]
    result, = measured_context({'match_threshold':.85},probes,{'return':'A'},{'entrant':{'A'}})
    assert abs(result['sapiensid.identities.scene_match_bonus']-.03)<.000002
    assert result['sapiensid.identities.ambiguity_margin']>.16

def test_all_four_control_settings_survive_normalization_and_yaml_diff(tmp_path):
    from stash_ai.recognition.scene_strategy_validation import training_profiles
    from stash_ai.recognition.merge_report_policy import patched, proposal, parameter
    from stash_ai.recognition.merge_solver_domains import normalize
    cfg = configuration(tmp_path)
    cfg.data['clustering']['same_scene_match_bonus'] = .03
    cfg.data['sapiensid']['identities']['scene_match_bonus'] = None
    controls = set()
    for name, settings in list(training_profiles(cfg))[:4]:
        patch = proposal(cfg,normalize(cfg,settings))
        replay = patched(cfg,{row['parameter']:row['proposed'] for row in patch['changes']})
        controls.add((replay.data['clustering']['same_scene_match_bonus'],
                      parameter(replay.data,'sapiensid.identities.scene_match_bonus')))
    assert controls == {(a,b) for a in (0.,.03) for b in (0.,.03)}
