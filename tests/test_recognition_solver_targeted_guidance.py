import uuid
from types import SimpleNamespace

from stash_ai.recognition.config import defaults
from stash_ai.recognition.merge_solver_constraints import alternative_separation, joint_separation
from stash_ai.recognition.merge_solver_targets import deficits, bridge_gain
from stash_ai.recognition.merge_solver_diagnostics import latest_decisions, fair_queries
from stash_ai.recognition.merge_solver_feedback import PREFIX
from tests.test_recognition_merge_identity_policy import member, goals, assignments


def test_alternative_reference_does_not_force_weak_route_or_foreign_merge():
    config = SimpleNamespace(data=defaults())
    face, body = PREFIX+'face_floor', PREFIX+'person_threshold'
    good, weak, rival = {face:.8, body:.8}, {face:.3, body:.3}, {face:.5, body:.5}
    assert joint_separation(config, [good, weak], [rival])['unresolved']
    proposals = alternative_separation(config, [[good, weak]], [rival])
    best = proposals[0]
    assert all(best[k] <= v for k, v in good.items())
    assert any(best[k] > v for k, v in rival.items())


def test_deficits_prioritize_missing_scene_and_complementary_bridge():
    goal = goals([member('a', scenes=['s1']*3), member('b', scenes=['s2']*3), member('c', scenes=['s1']*3)])
    predicted = assignments(goal)
    predicted.update({f'c-{i}': 'a' for i in range(3)})
    focus = deficits(goal, predicted, predicted)
    assert focus['sets'][0]['missing_target_scenes'] == ['s2']
    assert focus['priorities']['b-0'] > focus['priorities'].get('c-0', 0)
    assert bridge_gain('a', 'b', focus) > bridge_gain('a', 'c', focus)
    merged = dict.fromkeys(predicted, 'merged')
    assert not deficits(goal, merged, merged)['priorities']


def test_latest_decisions_ignore_unrelated_json_and_order_across_policies(tmp_path):
    from benchmarks.benchmark_merge_solver import fixture
    from stash_ai.recognition.database import FaceDatabase, encoded
    config, goal = fixture(tmp_path, 'query')
    oid = next(iter(assignments(goal)))
    with FaceDatabase(config.database) as state:
        db = state.connection
        scan = db.execute('SELECT id FROM scans WHERE id<>(SELECT scan FROM observations WHERE id=?)', (oid,)).fetchone()[0]
        db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?)',
            (str(uuid.uuid4()), scan, 'later-policy', None, 'withheld', encoded(dict(observation=oid, reason='latest'))))
        db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?)',
            (str(uuid.uuid4()), scan, 'later-policy', None, 'withheld', 'malformed'))
        result = latest_decisions(db, {oid})
        assert len(result) == 1 and result[0]['reason'] == 'latest'
        assert result[0]['status'] == 'withheld'


def test_sampling_rotates_even_when_quota_is_smaller_than_number_of_sets(tmp_path):
    from tests.test_recognition_merge_solver_exploration import auxiliary_goals
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.clustering import matching_policy
    config, goal = auxiliary_goals(tmp_path, sets=2)
    visits, seen = {}, set()
    with FaceDatabase(config.database) as state:
        db = state.connection
        db.execute('UPDATE candidates SET policy=?', (matching_policy(config),))
        scans = [dict(scan=r[0]) for r in db.execute('SELECT id FROM scans')]
        births = [m['identity_id'] for g in goal['groups'] for m in g['members']]
        for _ in range(20):
            selected, eligible = fair_queries(db, config, goal, scans, births, 1, visits=visits)
            assert len(selected) == 1
            oid = selected.pop()
            if len(seen) < eligible:
                assert oid not in seen
            visits[oid] = visits.get(oid, 0)+1
            seen.add(oid)
        assert len(seen) == eligible
    assert {g['index'] for g in goal['groups'] for m in g['members']
            for r in m['observations'] if r['id'] in seen} == {0, 1}


def test_missing_body_generates_face_route_from_real_replay(tmp_path):
    from benchmarks.benchmark_merge_solver import fixture
    from stash_ai.recognition.fragmentation_report import _open_readonly
    from stash_ai.recognition.merge_solver import solve
    config, goal = fixture(tmp_path, 'missing_body')
    with _open_readonly(config.database) as db:
        result = solve(config, goal, db, maximum=12)
    feedback = result['simulation']['trials']['baseline']['solver_feedback']
    assert feedback['query']['face_proposals']
    assert result['best_result']['fully_solved']
    assert all(not k.startswith('sapiensid.') for p in feedback['query']['face_proposals'] for k in p)
