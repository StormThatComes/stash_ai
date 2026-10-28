"""A set declares one person; historical fragments are not preservation goals."""
import copy

import pytest

from stash_ai.recognition.merge_goal_identity import diagnostic_goals, destination_owners
from stash_ai.recognition.merge_goal_output import group_summary
from stash_ai.recognition.merge_goals import objective, unmet_conditions
from tests.test_recognition_merge_identity_policy import member, goals, assignments, status


def overlapping_fragments(strong_count=3):
    names = ['a', 'b', 'c'] + [f'fragment-{i}' for i in range(strong_count-3)]
    return goals([member(n, core=4, scenes=['s1']*2+['s2']*2) for n in names] +
                 [member('weak', core=0, soft=20, scenes=['s1']*10+['s2']*10)])


@pytest.mark.parametrize('weak_result', ['split', 'withheld', 'missing'])
@pytest.mark.parametrize('strong_count', [3, 19])
def test_partial_merge_beats_baseline_despite_unrepresented_weak_fragment(weak_result, strong_count):
    goal = overlapping_fragments(strong_count)
    snapshot = copy.deepcopy(goal)
    before = assignments(goal)
    after = {oid: ('merged' if d in ('a', 'b') else d) for oid, d in before.items()}
    weak = [f'weak-{i}' for i in range(20)]
    after.update({oid: ('merged' if i % 2 else 'c') if weak_result == 'split' else None
                  for i, oid in enumerate(weak)})
    result = status(goal, after, missing=weak if weak_result == 'missing' else ())
    group = result['groups'][0]
    assert result['admissible'] and not result['fully_solved']
    assert group['represented_performers'] == strong_count and group['total_performers'] == strong_count+1
    assert group['covered_scenes'] == group['required_scenes'] == 2
    assert group['soft_lost_units'] == (0 if weak_result == 'split' else 20)
    assert group['soft_budget_excess'] == (0 if weak_result == 'split' else 16)
    assert objective(result) > objective(status(goal, before))
    assert [v['reason'] for v in unmet_conditions(result)] == ['incomplete_identity_set']
    assert 'scenes_preserved=2/2' in group_summary(group)
    assert goal == snapshot


def test_full_set_merge_can_drop_weak_fragment_when_other_members_cover_its_scenes():
    goal = overlapping_fragments()
    after = {oid: None if d == 'weak' else 'merged' for oid, d in assignments(goal).items()}
    result = status(goal, after)
    assert result['fully_solved'] and not result['strict_fully_solved']
    assert result['groups'][0]['represented_performers'] == 3
    assert result['groups'][0]['set_target'] == 'merged'
    assert not unmet_conditions(result)
    assert not diagnostic_goals(goal)['groups'][0]['members'][-1]['observations']


def test_split_core_fragment_is_preserved_across_other_destinations_of_same_set():
    goal = goals([member(n, core=4, scenes=['same']*4) for n in ('a', 'b', 'c', 'split')])
    before = assignments(goal)
    after = {oid: ('merged' if d in ('a', 'b') else d) for oid, d in before.items()}
    after.update({f'split-{i}': 'merged' if i % 2 else 'c' for i in range(4)})
    result = status(goal, after)
    group = result['groups'][0]
    assert group['represented_performers'] == 3
    assert group['set_core_retention'] == 1
    assert result['admissible'] and not result['fully_solved']
    assert objective(result) > objective(status(goal, before))


@pytest.mark.parametrize('destination', [None, 'other-person'])
def test_other_person_in_same_scene_cannot_replace_lost_set_scene(destination):
    goal = goals([member('a', scenes=['s1']*3), member('weak', core=0, soft=4, scenes=['s2']*4)],
                 [member('other-person', scenes=['s2']*3)])
    after = assignments(goal)
    after.update({f'weak-{i}': destination for i in range(4)})
    result = status(goal, after)
    assert result['disjoint'] and not result['admissible']
    assert result['groups'][0]['missing_scenes'] == ['s2']
    assert result['groups'][0]['soft_lost_units'] == 4
    assert dict(set=0, reason='set_scene_not_preserved', scenes=['s2']) in unmet_conditions(result)
    assert destination_owners(goal, after)['other-person'] == {1}


def test_weak_only_scene_is_preserved_but_must_join_common_destination_for_full_solution():
    goal = goals([member('a', scenes=['s1']*3), member('weak', core=0, soft=4, scenes=['s2']*4)])
    before = assignments(goal)
    partial = status(goal, before)
    assert partial['admissible'] and not partial['fully_solved']
    after = dict(before, **{'weak-0': 'a'})
    merged = status(goal, after)
    assert merged['fully_solved']  # One independent link preserves the weak-only scene.
    assert objective(merged) > objective(partial)
    assert merged['groups'][0]['soft_lost_units'] == 0


def test_scene_presence_alone_does_not_certify_two_substantial_core_fragments_as_merged():
    goal = goals([member('a', scenes=['same']*3), member('b', scenes=['same']*3)])
    partial = status(goal, assignments(goal))
    assert partial['admissible'] and not partial['fully_solved']
    after = dict.fromkeys(assignments(goal), 'merged')
    assert status(goal, after)['fully_solved']
    after.update({f'b-{i}': None for i in range(3)})
    discarded = status(goal, after)
    assert discarded['admissible'] and not discarded['fully_solved']
    assert objective(partial) > objective(discarded)


def test_substantial_cross_set_core_collision_is_not_diluted_by_larger_set():
    goal = goals([member('large', core=100), member('small')], [member('other')])
    predicted = assignments(goal)
    predicted.update({f'small-{i}': 'other' for i in range(3)})
    result = status(goal, predicted)
    assert not result['disjoint'] and not result['admissible']
    assert result['cross_set_evidence_pairs'] == 9
    assert destination_owners(goal, predicted)['other'] == {0, 1}


def test_ambiguous_soft_destination_cannot_cover_two_sets_without_core_owner():
    goal = goals([member('a'), member('weak-a', core=0, soft=3)],
                 [member('b'), member('weak-b', core=0, soft=3)])
    predicted = assignments(goal)
    predicted.update({f'{m}-{i}': 'ambiguous' for m in ('weak-a', 'weak-b') for i in range(3)})
    result = status(goal, predicted)
    assert not result['admissible']
    assert [g['missing_scenes'] for g in result['groups']] == [['weak-a'], ['weak-b']]


def test_set_objective_is_invariant_to_repartitioning_members_and_uuid_order():
    original = member('a', core=8, scenes=['s1']*4+['s2']*4)
    whole = goals([original])
    fragments = [dict(original, identity_id=f'part-{i}', performer_id=f'part-{i}',
                      observations=original['observations'][i::2]) for i in range(2)]
    from stash_ai.recognition.merge_goal_identity import freeze
    for fragment in fragments:
        fragment['identity_evidence'] = freeze(fragment['observations'])
    split = goals(fragments)
    predicted = {r['id']: 'x' if i < 5 else 'y' for i, r in enumerate(original['observations'])}
    expected = objective(status(whole, predicted))
    assert objective(status(split, predicted)) == expected
    renamed = {oid: {'x': 'z', 'y': 'a'}[d] for oid, d in predicted.items()}
    assert objective(status(goals(list(reversed(fragments))), renamed)) == expected


def test_duplicate_units_across_members_do_not_manufacture_set_core_quorum():
    goal = goals([member('a', core=4, pixels=['a'*64]*4), member('b', core=1, pixels=['b'*64])])
    predicted = dict.fromkeys(assignments(goal), 'merged')
    assert goal['groups'][0]['identity_evidence']['core_required'] == 2
    predicted['b-0'] = None
    # The other scene must also survive, independently of the repeated crop.
    assert not status(goal, predicted)['fully_solved']


def test_empty_historical_fragment_does_not_invalidate_nonempty_set():
    goal = goals([member('a'), member('empty', core=0)])
    result = status(goal, assignments(goal))
    assert result['fully_solved'] and result['groups'][0]['represented_performers'] == 1
    assert not unmet_conditions(result)
