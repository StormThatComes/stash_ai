"""Solver hints and objective use the same physical evidence as recognition."""
import json
from stash_ai.recognition.database import FaceDatabase
from tests.test_recognition_body_identity import configuration, scene


def query_votes(p):
 from tests.test_recognition_merge_solver_exploration import auxiliary_goals
 from stash_ai.recognition.merge_solver_feedback import QueryFeedback
 from stash_ai.recognition.hybrid_query_observer import observe_queries
 from stash_ai.recognition.merge_goals import evaluate
 from stash_ai.recognition.fragmentation_report import _open_readonly
 c,goals=auxiliary_goals(p)
 with FaceDatabase(c.database) as state:
  db=state.connection
  db.execute("UPDATE person_evidence SET metadata=json_set(metadata,'$.quality.pixel_digest','same-body') WHERE observation IN (SELECT id FROM observations WHERE json_extract(quality,'$.pose_deviation')>.2)")
  db.commit()
 observer=QueryFeedback(goals)
 with _open_readonly(c.database) as db,observe_queries(observer):r=evaluate(c,goals,db)
 q=observer.finish(c)
 return dict(query_support=[(e.get('query_support'),e.get('face_query_support')) for e in q['queries']],actual_independent_body_support=1,withheld=r['trials']['baseline']['goal_result']['groups'][0]['withheld'])


def quality_hints(p):
 from stash_ai.recognition.body_solver import options_and_quality
 from stash_ai.recognition.person_anchor import anchor_gate
 from stash_ai.recognition.database import encoded
 c=configuration(p)
 with FaceDatabase(c.database) as state:
  db=state.connection;item,ids=scene(db,c)
  m=dict(person=dict(bounds=[0,0,512,512],keypoints=[[10,10]]*5+[[-1,-1]]*12),quality=dict(keypoints=17,sharpness=100))
  db.execute('UPDATE body_observations SET quality=?',(encoded(m),))
  goals=dict(scope='scope',namespace='ns',groups=[dict(members=[dict(identity_id='p',observations=[dict(id=i) for i in ids])])])
  opts,patches,measurements=options_and_quality(db,c,goals,frozen_goals=goals)
  return dict(gate=anchor_gate(m,c.data['sapiensid']['identities']),patches=patches,measurements=measurements,keypoints_values=opts['sapiensid.identities.anchor_min_keypoints'])


def test_query_feedback_counts_body_pixels_independently_from_faces(tmp_path):
    result=query_votes(tmp_path)
    assert result['query_support'] and all(body==1 and face==3 for body,face in result['query_support'])


def test_retained_pose_hints_use_canonical_encoder_points(tmp_path):
    result=quality_hints(tmp_path)
    assert result['gate']['keypoints']==5 and not result['gate']['eligible']
    assert result['patches']==[{'sapiensid.identities.anchor_min_keypoints':5}]
    assert len(result['measurements'])==3


def test_hybrid_only_exposes_quality_parameters_and_measurements(tmp_path):
    from tests.test_recognition_merge_solver_exploration import auxiliary_goals
    from stash_ai.recognition.body_parameters import active,QUALITY
    from stash_ai.recognition.body_solver import options_and_quality
    from stash_ai.recognition.person_anchor import anchor_gate
    c,goals=auxiliary_goals(tmp_path)
    assert not c.data['sapiensid']['identities']['enabled']
    assert set(QUALITY)<=set(active(c))
    with FaceDatabase(c.database) as state:
        db=state.connection
        db.execute("UPDATE person_evidence SET metadata=json_set(metadata,'$.quality.sharpness',20)")
        options,patches,measurements=options_and_quality(db,c,goals,frozen_goals=goals)
        assert patches==[{'sapiensid.identities.anchor_min_sharpness':20.}]
        assert measurements and 20. in options['sapiensid.identities.anchor_min_sharpness']


def test_outsider_replacement_cannot_pass_zero_new_observations():
    from stash_ai.recognition.merge_goals import assess
    from stash_ai.recognition.merge_solver_outside import outsiders,apply
    goals={'groups':[{'index':0,'members':[dict(identity_id='g',performer_id=None,observations=[{'id':'goal'}])]}]}
    before={'goal':'a','old':'a','new':'b'};after={'goal':'b','old':'a','new':'b'}
    observations=dict.fromkeys(before,'1')
    baseline=assess(goals,observations,before,before);trial=assess(goals,observations,before,after)
    added=outsiders(trial,after,{'goal'})-outsiders(baseline,before,{'goal'})
    assert added=={(0,'new')}
    apply(trial,dict(new_outsider_observations=len(added)),baseline,dict(new_outsider_observations=0))
    assert not trial['outside_preserved'] and not trial['fully_solved']


def test_outsider_limits_ignore_destination_uuid_renaming():
    from stash_ai.recognition.merge_solver_outside import outsiders
    baseline=dict(groups=[dict(index=0,resulting_identities={'a':3})])
    trial=dict(groups=[dict(index=0,resulting_identities={'renamed':3})])
    assert outsiders(baseline,{'g':'a','x':'a'},{'g'})==outsiders(trial,{'g':'renamed','x':'renamed'},{'g'})


def test_body_mode_benchmark_contains_supported_person_appearances(tmp_path):
    from benchmarks.benchmark_merge_solver import fixture
    from stash_ai.recognition.hybrid_evidence import catalogue
    c,goals=fixture(tmp_path,'body_modes')
    with FaceDatabase(c.database) as state:
        bank=catalogue(state.connection,c,goals['scope'],goals['namespace'])
        assert len(bank)==2
        assert all(len(i.body_modes)==2 and all(len(m)==4 for m in i.body_modes) for i in bank)
