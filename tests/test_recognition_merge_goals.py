import pytest
from stash_ai.recognition.cli import parser
from stash_ai.recognition.merge_goals import resolve
from stash_ai.recognition.fragmentation_report import _open_readonly
from tests.test_recognition_merge_report import seeded


def test_sets_and_alias_overlap(tmp_path):
    config, people = seeded(tmp_path)
    with _open_readonly(config.database) as db:
        result = resolve(db, config, [['900', '900'], ['901']])
        assert len(result['groups'][0]['members']) == 1
        assert len(result['groups'][0]['members'][0]['observations']) == 3
        with pytest.raises(ValueError, match='Overlapping'):
            resolve(db, config, [['900'], ['900']])
    args = parser().parse_args(['report-merge', '--performer-set', '900,901', '--performer-set', '902'])
    assert args.performer_set == [['900', '901'], ['902']]
    assert len(parser().parse_args(['report-merge', '--identity-id', 'a', '--identity-id', 'b', '--identity-id', 'c']).identity_id) == 3


def test_final_assignments_reject_bridges_and_lost_members():
    from stash_ai.recognition.merge_goals import assess
    goals = {'groups': [dict(index=i, members=[dict(identity_id=str(i), performer_id=str(i),
        observations=[dict(id=k) for k in keys])]) for i, keys in enumerate([['a','b'], ['c']])]}
    obs = dict.fromkeys('abc', '1')
    assert assess(goals, obs, {}, dict(a='x', b='x', c='y'))['fully_solved']
    assert not assess(goals, obs, {}, dict(a='x', b='x', c='x'))['admissible']
    assert not assess(goals, obs, {}, dict(a='x', b=None, c='y'))['admissible']
    assert not assess(goals, {'a':'1','c':'1'}, {}, dict(a='x', c='y'))['admissible']


def test_objective_balances_sets_and_rejects_coverage_loss():
    from stash_ai.recognition.merge_goals import objective
    def status(a,b,admissible=True):
        return dict(admissible=admissible, fully_solved=False, groups=[dict(complete_fraction=x,
            largest_complete_performers=int(x*10), fragments=2) for x in (a,b)])
    assert objective(status(.6,.6)) > objective(status(1.,.2))
    assert objective(status(.6,.6)) > objective(status(1.,1.,False))
