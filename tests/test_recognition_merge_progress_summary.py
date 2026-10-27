import pytest

from stash_ai.recognition.merge_goal_output import group_summary
from stash_ai.recognition.merge_goals import assess
from tests.test_recognition_merge_best_effort import goal_sets


@pytest.mark.parametrize('destinations,complete,components,fragments', [
    ('abcd', 1, '[1,1,1,1]', 4),
    ('aabb', 2, '[2,2]', 2),
    ('aaab', 3, '[3,1]', 2),
    ('dddd', 4, '[4]', 1),  # The final identity need not be the first performer.
])
def test_complete_components_distinguish_independent_merges(destinations, complete, components, fragments):
    goals = goal_sets([['a'], ['b'], ['c'], ['d']])
    observations = dict.fromkeys('abcd', 'scene')
    group = assess(goals, observations, {}, dict(zip('abcd', destinations)))['groups'][0]
    line = group_summary(group)
    assert f'Set 1: largest_complete={complete}/4 complete_components={components}' in line
    assert 'assigned=4/4 (100.0%)' in line
    assert f'fragments={fragments}' in line
    assert 'missing_obs=0 withheld_obs=0 missing_performers=0 withheld_performers=0 split_performers=0' in line


@pytest.mark.parametrize('blocker', ['missing', 'withheld', 'split'])
def test_partial_merges_remain_visible_when_no_performer_is_complete(blocker):
    goals = goal_sets([[f'{p}-{n}' for n in range(100)] for p in range(4)])
    observations = {o['id']: 'scene' for m in goals['groups'][0]['members'] for o in m['observations']}
    predictions = dict.fromkeys(observations, 'shared')
    for performer in range(4):
        observation = f'{performer}-99'
        if blocker == 'missing':
            del observations[observation]
            del predictions[observation]
        else:
            predictions[observation] = None if blocker == 'withheld' else f'split-{performer}'
    line = group_summary(assess(goals, observations, {}, predictions)['groups'][0])
    assert 'largest_complete=0/4 complete_components=[]' in line
    assert 'shared_coverage=99.0%' in line
    if blocker == 'split':
        assert 'assigned=400/400 (100.0%)' in line and 'split_performers=4' in line and 'fragments=5' in line
    else:
        assert 'assigned=396/400 (99.0%)' in line and 'fragments=1' in line
        assert f'{blocker}_obs=4' in line and f'{blocker}_performers=4' in line
        assert 'split_performers=0' in line


def test_empty_performer_has_explicit_reason_without_dividing_by_zero():
    group = assess(goal_sets([[]]), {}, {}, {})['groups'][0]
    line = group_summary(group)
    assert 'largest_complete=0/1 complete_components=[] assigned=0/0 (0.0%)' in line
    assert 'missing_obs=0 withheld_obs=0' in line and 'empty_performers=1' in line


def test_assigned_percentage_uses_observations_and_shared_coverage_weights_performers():
    many = [str(i) for i in range(100)]
    goals = goal_sets([many, ['small']])
    observations = dict.fromkeys(many + ['small'], 'scene')
    predictions = dict.fromkeys(many, 'shared')
    line = group_summary(assess(goals, observations, {}, predictions)['groups'][0])
    assert 'assigned=100/101 (99.0%) shared_coverage=50.0%' in line
    assert 'withheld_obs=1' in line and 'withheld_performers=1' in line
