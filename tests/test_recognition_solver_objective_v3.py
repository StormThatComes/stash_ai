import copy

import pytest

from tests.test_recognition_merge_identity_policy import member, goals, assignments, status
from stash_ai.recognition.merge_goal_identity import freeze, destination_owners
from stash_ai.recognition.merge_goals import objective, unmet_conditions
from stash_ai.recognition.merge_solver_outside import apply, validate


@pytest.mark.parametrize('crossed', [1, 2, 3, 5])
def test_cross_set_ownership_is_invariant_to_historical_repartition(crossed):
    original = member('a', core=10, scenes=['scene-a']*10)
    rival = member('b')
    whole = goals([original], [rival])
    fragments = []
    for i, rows in enumerate((original['observations'][:2], original['observations'][2:])):
        fragments.append(dict(original, identity_id=f'part-{i}', performer_id=f'part-{i}',
                              observations=rows, identity_evidence=freeze(rows)))
    split = goals(fragments, [rival])
    predicted = assignments(whole)
    predicted.update({f'a-{i}': 'b' for i in range(crossed)})
    before, after = status(whole, predicted), status(split, predicted)
    assert objective(before) == objective(after)
    assert before['conflicts'] == after['conflicts']
    assert destination_owners(whole, predicted) == destination_owners(split, predicted)
    assert before['disjoint'] == (crossed <= 2)


def test_small_scene_core_conflict_cannot_be_diluted_and_duplicates_do_not_add_votes():
    a = member('a', core=103, scenes=['large']*100+['small']*3)
    b = member('b')
    goal = goals([a], [b])
    predicted = assignments(goal)
    predicted.update({f'a-{i}': 'b' for i in range(100, 103)})
    result = status(goal, predicted)
    assert not result['disjoint'] and result['cross_set_evidence_pairs'] == 9
    duplicate = dict(a, identity_id='duplicate', performer_id='duplicate')
    repeated = status(goals([a, duplicate], [b]), predicted)
    assert repeated['cross_set_evidence_pairs'] == 9


def test_pair_score_does_not_weight_dense_scenes_more_heavily():
    def evaluate(density):
        goal = goals([member('a', core=density+2, scenes=['dense']*density+['sparse']*2)])
        predicted = {f'a-{i}': 'x' if i < density or i == density else 'y' for i in range(density+2)}
        return status(goal, predicted)['groups'][0]
    for count in (2, 10, 100):
        assert evaluate(count)['set_pair_fraction'] == .5
        assert evaluate(count)['raw_set_pair_fraction'] == .5


@pytest.mark.parametrize('field', ['coverage_lost', 'split_pairs', 'newly_joined_pairs', 'new_outsider_observations'])
def test_outside_limits_precede_goal_improvement(field):
    goal = goals([member('a'), member('b')])
    baseline = status(goal, assignments(goal))
    solved = status(goal, dict.fromkeys(assignments(goal), 'merged'))
    outside = dict(coverage_lost=0, split_pairs=0, newly_joined_pairs=0)
    limits = validate({field: 0})
    apply(baseline, outside, baseline, limits)
    if field == 'new_outsider_observations':
        solved['groups'][0]['outsider_observations'] += 1
    else:
        outside[field] = 1
    apply(solved, outside, baseline, limits)
    assert objective(baseline) > objective(solved)
    assert not solved['admissible'] and not solved['fully_solved']
    assert any(v['reason'] == 'outside_impact_limit' for v in unmet_conditions(solved))


def test_outside_limits_are_optional_and_validate_values():
    goal = goals([member('a')])
    result = status(goal, assignments(goal))
    original = copy.deepcopy(result)
    apply(result, {}, result, validate(None))
    assert result == original
    for limits in ({'coverage_lost': -1}, {'split_pairs': True}, {'unknown': 1}):
        with pytest.raises(ValueError): validate(limits)


def test_solver_applies_outside_limits_in_full_replay(tmp_path):
    from benchmarks.benchmark_merge_solver import fixture
    from stash_ai.recognition.fragmentation_report import _open_readonly
    from stash_ai.recognition.merge_solver import solve
    config, goal = fixture(tmp_path, 'query')
    with _open_readonly(config.database) as db:
        result = solve(config, goal, db, maximum=4, outside_limits={'coverage_lost': 0})
    assert result['best_result']['fully_solved']
    assert all(t['goal_result']['outside_preserved'] for t in result['simulation']['trials'].values())
