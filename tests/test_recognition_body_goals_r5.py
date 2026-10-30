import math
from contextlib import closing

from stash_ai.recognition.database import FaceDatabase, encoded
from stash_ai.recognition.body_identity import scene_plan
from tests.test_recognition_body_identity import configuration, scene


def test_weak_body_uses_second_candidate_after_copresence_veto(tmp_path):
    config = configuration(tmp_path)
    vectors = [[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in [0,0,0,40,40,40,10]]
    with FaceDatabase(config.database) as state:
        item, ids = scene(state.connection, config, vectors=vectors, times=[0,1,2,3,4,5,0])
        state.connection.execute('UPDATE body_observations SET quality=? WHERE id=?',
            (encoded({'quality': {'keypoints':4, 'sharpness':100, 'pixel_digest':'weak'}}), ids[-1]))
        _, _, decisions = scene_plan(state.connection, item['scan'], config)
    assert decisions[ids[-1]]['status'] == 'accepted'
    assert decisions[ids[-1]]['person'] == decisions[ids[3]]['person']
    assert decisions[ids[-1]]['person'] != decisions[ids[0]]['person']


def test_physical_policy_preserves_body_handoff_without_duplicate_votes(tmp_path):
    from tests.test_recognition_solver_feasibility import body_handoff
    from stash_ai.recognition.fragmentation_report import _open_readonly
    from stash_ai.recognition.merge_goals import resolve, assess
    from stash_ai.recognition.merge_solver import solve
    config, faces, bodies = body_handoff(tmp_path)
    with closing(_open_readonly(config.database)) as db:
        ordinary = resolve(db, config, [['900']])
        goals = resolve(db, config, [['900']], policy='identity-physical')
        observations = dict.fromkeys(faces+bodies, '1')
        body_only = dict.fromkeys(bodies, 'person')
        assert not assess(ordinary, observations, {}, body_only)['fully_solved']
        status = assess(goals, observations, {}, body_only)
        assert status['fully_solved'] and not status['strict_fully_solved']
        assert status['groups'][0]['set_core_units'] == 3
        both = dict.fromkeys(faces+bodies, 'person')
        assert assess(goals, observations, {}, both)['groups'][0]['set_core_destinations'] == {'person':3}
        conflict = dict.fromkeys(faces, 'face') | dict.fromkeys(bodies, 'body')
        assert not assess(goals, observations, {}, conflict)['fully_solved']
        result = solve(config, goals, db, maximum=1)
        assert result['simulation']['trials']['baseline']['goal_result']['fully_solved']


def test_rival_policy_preserves_unknown_but_excludes_certified_conflicts():
    from stash_ai.recognition.competitor_eligibility import body, hybrid
    assert body({'blockers':['insufficient_support']})
    assert not body({'blockers':['person_copresence']})
    assert not body({'blockers':['face_contradiction']})
    assert hybrid(None, None, .5)
    assert not hybrid(None, .2, .5)


def test_local_body_hints_measure_threshold_and_support_without_changing_result(tmp_path):
    from stash_ai.recognition.merge_solver_feedback import QueryFeedback
    from stash_ai.recognition.hybrid_query_observer import observe_queries
    config = configuration(tmp_path)
    angle = math.radians(35)
    with FaceDatabase(config.database) as state:
        item, ids = scene(state.connection, config, vectors=[[1,0]]*3+[[math.cos(angle),math.sin(angle)]])
        state.connection.execute('UPDATE body_observations SET quality=? WHERE id=?',
            (encoded({'quality': {'keypoints':4, 'sharpness':100}}), ids[-1]))
        expected = scene_plan(state.connection, item['scan'], config)[2]
        goals = dict(groups=[dict(index=0, members=[dict(observations=[dict(id=oid) for oid in ids])])])
        feedback = QueryFeedback(goals)
        with observe_queries(feedback):
            actual = scene_plan(state.connection, item['scan'], config)[2]
        hints = feedback.finish(config)['local_body']
    assert actual == expected
    query = next(q for q in hints if q['observation'] == ids[-1])
    assert abs(query['mean']-math.cos(angle)) < 1e-6
    assert query['support'] == 3 and query['support_gate'] == 'min_support'
