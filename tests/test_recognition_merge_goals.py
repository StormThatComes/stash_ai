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
        return dict(admissible=admissible, fully_solved=False, disjoint=True, cross_set_observation_pairs=0,
            groups=[dict(complete_fraction=x, assigned_fraction=1., consolidation_fraction=x,
            largest_complete_performers=int(x*10), fragments=2) for x in (a,b)])
    assert objective(status(.6,.6)) > objective(status(1.,.2))
    assert objective(status(.6,.6)) > objective(status(1.,1.,False))


def test_real_solver_and_export_reproduce_disjoint_partition(tmp_path):
    import json
    from stash_ai.recognition.merge_solver import solve
    from stash_ai.recognition.merge_goal_output import export
    from stash_ai.recognition.config import load
    from stash_ai.recognition.merge_goals import evaluate
    config, people = seeded(tmp_path, vectors=[[1.,0,0,0], [.7,.71414284285,0,0], [0,0,1.,0]])
    before = config.database.read_bytes()
    with _open_readonly(config.database) as db:
        goals = resolve(db, config, [['900','901'], ['902']])
        result = dict(goals, **solve(config, goals, db, maximum=12))
        assert result['status'] == 'fully_solved'
        assert result['recommendation']
        assert result['simulation']['trials'][result['selected_trial']]['goal_result']['disjoint']
        assert any(not i['goal_result']['disjoint'] for i in result['simulation']['trials'].values())
    output, yaml = tmp_path/'result.json', tmp_path/'result.yml'
    export(config, result, output, yaml)
    new = load(yaml)
    assert new.database == config.database
    with _open_readonly(config.database) as db:
        replay = evaluate(new, goals, db)
    assert replay['trials']['baseline']['goal_result']['fully_solved']
    assert config.database.read_bytes() == before
    assert json.loads(output.read_text())['yaml_exported']
    with pytest.raises(FileExistsError):
        export(config, result, output, yaml)


def test_interrupted_search_preserves_only_completed_trials(tmp_path):
    from stash_ai.recognition.merge_solver import solve
    config, _ = seeded(tmp_path)
    def progress(name, stage, item):
        if name != 'baseline' and stage == 'start':
            raise KeyboardInterrupt
    with _open_readonly(config.database) as db:
        goals = resolve(db, config, [['900','901']])
        result = solve(config, goals, db, maximum=8, progress=progress)
    assert result['interrupted']
    assert list(result['simulation']['trials']) == ['baseline']
    assert result['recommendation'] is None


def test_budget_one_keeps_baseline_and_never_claims_impossibility(tmp_path):
    from stash_ai.recognition.merge_solver import solve
    config, _ = seeded(tmp_path)
    with _open_readonly(config.database) as db:
        goals = resolve(db, config, [['900','901']])
        result = solve(config, goals, db, maximum=1)
        assert result['selected_trial'] == 'baseline'
        assert result['search']['budget_exhausted']
        assert not result['search']['proves_infeasibility']
        assert result['recommendation'] is None
        with pytest.raises(ValueError):
            solve(config, goals, db, maximum=0)


def test_target_outsider_join_is_reported():
    from stash_ai.recognition.merge_goals import assess
    goals = dict(groups=[dict(index=0, members=[dict(identity_id='a', performer_id='1', observations=[dict(id='x')])])])
    result = assess(goals, dict(x='1',z='2'), dict(x='a',z='outsider'), dict(x='merged',z='merged'))
    assert result['groups'][0]['additional_retained_identities'] == ['outsider']
    assert result['groups'][0]['outsider_observations'] == 1
