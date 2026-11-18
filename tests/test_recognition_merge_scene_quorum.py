"""A large scene cannot hide a smaller scene's unmerged reference evidence."""
from contextlib import nullcontext

import pytest

from stash_ai.recognition.merge_goal_index import prepared
from stash_ai.recognition.merge_goals import unmet_conditions
from tests.test_recognition_merge_identity_policy import member, goals, assignments, status


@pytest.mark.parametrize('indexed', [False, True])
@pytest.mark.parametrize('remaining', ['small', None])
@pytest.mark.parametrize('small_count', [2, 3, 5])
def test_one_link_cannot_hide_separate_or_withheld_scene_core(indexed, remaining, small_count):
    goal = goals([member('large', core=100), member('small', core=small_count)])
    predicted = assignments(goal)
    predicted.update({f'small-{i}': remaining for i in range(small_count)})
    predicted['small-0'] = 'large'
    with prepared(goal) if indexed else nullcontext():
        result = status(goal, predicted)
    group = result['groups'][0]
    assert result['admissible'] and not result['fully_solved']
    assert group['covered_scenes'] == group['required_scenes'] == 2
    assert group['set_core_destinations']['large'] >= group['set_core_required']
    need = goal['groups'][0]['identity_evidence']['protected_scenes']['small']
    assert group['scene_core_deficits'] == {'small': dict(required=need, assigned=1)}
    assert unmet_conditions(result)[0]['scene_core_deficits'] == group['scene_core_deficits']


@pytest.mark.parametrize('linked, solved', [(1, False), (3, False), (4, True), (5, True)])
def test_scene_quorum_uses_frozen_core_and_keeps_outlier_tolerance(linked, solved):
    goal = goals([member('large', core=100), member('small', core=5, soft=4)])
    predicted = assignments(goal)
    predicted.update({f'small-{i}': 'large' for i in range(linked)})
    # Auxiliary links cover the scene but cannot replace its frozen references.
    predicted.update({f'small-{i}': 'large' for i in range(5, 9)})
    result = status(goal, predicted)
    assert result['fully_solved'] is solved
    assert result['groups'][0]['scene_core_required'] == {'large': 80, 'small': 4}
    assert result['groups'][0]['scene_core_support']['small']['large'] == linked
    assert bool(result['groups'][0]['scene_core_deficits']) is not solved


def test_search_prioritizes_unmerged_scene_core_after_pooled_quorum_is_met():
    from stash_ai.recognition.merge_solver_targets import deficits
    goal = goals([member('large', core=100, soft=1), member('small', core=5)])
    predicted = assignments(goal)
    predicted['large-100'] = 'fragment'
    predicted['small-0'] = 'large'
    focus = deficits(goal, predicted, predicted)
    group = focus['sets'][0]
    assert group['core_deficit'] == 0 and not group['missing_target_scenes']
    assert group['scene_core_deficits'] == {'small': dict(required=4, assigned=1)}
    assert focus['priorities']['small-1'] > focus['priorities'].get('large-100', 0)
    assert not deficits(goal, dict.fromkeys(predicted, 'merged'), dict.fromkeys(predicted, 'merged'))['priorities']


def uneven_scene_fixture(tmp_path):
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.merge_goals import resolve
    from tests.test_recognition_merge_report import seeded
    config, people = seeded(tmp_path, count=15)
    with FaceDatabase(config.database) as state:
        db = state.connection
        ids = [r[0] for r in db.execute("SELECT o.id FROM observations o JOIN scans s ON s.id=o.scan "
                                       "WHERE s.scene='2' AND o.cue>=5")]
        db.executemany('DELETE FROM members WHERE observation=?', ((oid,) for oid in ids))
        db.executemany("DELETE FROM candidates WHERE json_extract(payload,'$.observation')=?", ((oid,) for oid in ids))
        db.executemany('DELETE FROM observations WHERE id=?', ((oid,) for oid in ids))
        db.execute("UPDATE scans SET total=5,cursor=5 WHERE scene='2'")
        # One of the second performer's five references joins the first; four
        # remain separate. The old pooled quorum incorrectly certified 16/20.
        db.execute("UPDATE observations SET embedding=(SELECT embedding FROM observations o "
                   "JOIN scans s ON s.id=o.scan WHERE s.scene='1' LIMIT 1) "
                   "WHERE scan IN (SELECT id FROM scans WHERE scene='2') AND cue=0")
        db.commit()
        goal = resolve(db, config, [people], by_performer=False)
    return config, goal


def exported_production_result(config, goal, result, tmp_path):
    from stash_ai.recognition.config import load
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.merge_goal_output import configuration_document
    from stash_ai.recognition.merge_goals import assess
    from stash_ai.recognition.rebuild import rebuild_selected
    winner = tmp_path/'winner.yml'
    winner.write_text(configuration_document(config, result['best_result'], {}))
    assert result['best_result']['quality'] in winner.read_text()
    config = load(winner)
    with FaceDatabase(config.database) as state:
        db = state.connection
        selection = [dict(scan=r['id'], scene=r['scene']) for r in db.execute('SELECT id,scene FROM scans')]
        rebuild_selected(state, selection, config)
        predicted = dict(db.execute('SELECT observation,person FROM members'))
        observations = dict(db.execute('SELECT o.id,s.scene FROM observations o JOIN scans s ON s.id=o.scan'))
        return assess(goal, observations, {}, predicted)


def test_solver_export_and_production_rebuild_do_not_certify_partial_scene_merge(tmp_path):
    from stash_ai.recognition.fragmentation_report import _open_readonly
    from stash_ai.recognition.merge_goal_output import explain
    from stash_ai.recognition.merge_solver import solve
    from stash_ai.recognition.merge_solver_report import compact_document
    config, goal = uneven_scene_fixture(tmp_path)
    with _open_readonly(config.database) as db:
        result = solve(config, goal, db, maximum=1)
    assert result['best_result']['quality'] == 'partial_solution'
    assert not result['best_result']['fully_solved']
    document = dict(goal, **result)
    explain(document)
    compact = compact_document(document, config)
    example = compact['simulation']['trials']['baseline']['goal_result']['unmet_examples'][0]
    assert example['scene_core_deficit_count'] == 1
    assert example['scene_core_deficits'] == {'2': dict(required=4, assigned=1)}
    actual = exported_production_result(config, goal, result, tmp_path)
    assert sorted(actual['groups'][0]['resulting_identities'].values()) == [4, 16]
    assert not actual['fully_solved']
    assert actual['groups'][0]['scene_core_deficits'] == {'2': dict(required=4, assigned=1)}


def test_solver_can_find_a_solution_meeting_each_scenes_core_quorum(tmp_path):
    from stash_ai.recognition.fragmentation_report import _open_readonly
    from stash_ai.recognition.merge_solver import solve
    config, goal = uneven_scene_fixture(tmp_path)
    with _open_readonly(config.database) as db:
        result = solve(config, goal, db, maximum=8)
    assert not result['simulation']['trials']['baseline']['goal_result']['fully_solved']
    assert result['best_result']['fully_solved']
    group = result['simulation']['trials'][result['selected_trial']]['goal_result']['groups'][0]
    assert not group['scene_core_deficits']
    assert group['scene_core_support']['2'][group['set_target']] >= 4
    actual = exported_production_result(config, goal, result, tmp_path)
    assert actual['fully_solved']
    assert len(actual['groups'][0]['resulting_identities']) == 1
