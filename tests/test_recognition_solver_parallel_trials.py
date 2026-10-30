import json
from dataclasses import replace
import pytest

from benchmarks.benchmark_merge_solver import fixture
from stash_ai.recognition.fragmentation_report import _open_readonly
from stash_ai.recognition.merge_solver import solve
from stash_ai.recognition.merge_report_policy import patched
from stash_ai.recognition.merge_goals import evaluate


@pytest.mark.parametrize('scheduling', ['batch', 'async'])
def test_parallel_replay_resume_and_winner_verification(tmp_path, scheduling):
    config, goals = fixture(tmp_path/'input', 'disjoint')
    checkpoint = tmp_path/'checkpoint.json'
    execution = dict(cpus=4, trials=2, scheduling=scheduling, profile=True)
    original = config.database.read_bytes()
    def interrupt(name, stage, item):
        if name == 'trial-2' and stage == 'complete':
            raise KeyboardInterrupt
    with _open_readonly(config.database) as db:
        first = solve(config, goals, db, maximum=8, execution=execution,
            checkpoint=checkpoint, progress=interrupt)
        assert first['interrupted']
        resumed = solve(config, goals, db, maximum=8, execution=execution,
            checkpoint=checkpoint, resume=True)
        repeated = solve(config, goals, db, maximum=8, execution=execution,
            checkpoint=checkpoint, resume=True)
        assert repeated['simulation']['trials'] == resumed['simulation']['trials']
        assert repeated['search']['restored_trials'] == len(resumed['simulation']['trials'])
        patch = {r['parameter']:r['proposed'] for r in resumed['best_result']['changes']}
        verification = evaluate(patched(config, patch), goals, db)
        assert verification['trials']['baseline']['goal_result']['fully_solved']
    trials = resumed['simulation']['trials']
    assert len(trials) <= 8
    assert len({t["worker"]["worker_pid"] for n,t in trials.items() if n != "baseline"}) >= 2
    assert len({json.dumps(t.get('yaml_patch', {}), sort_keys=True) for t in trials.values()}) == len(trials)
    events = json.loads(checkpoint.read_text())['schedule']
    active = peak = 0
    for event in events:
        active += 1 if event['kind'] == 'dispatch' else -1
        peak = max(peak, active)
    assert peak == 2 and active == 0
    assert config.database.read_bytes() == original


def test_parallel_cli_and_joint_memory_limit(tmp_path):
    from stash_ai.recognition.cli import parser
    args = ['report-merge', '--performer-set', '900,901', '--solve',
            '--solver-cpus', '16', '--solver-trials', '4', '--solver-memory-gib', '96']
    parsed = parser().parse_args(args)
    assert parsed.solver_cpus == 16 and parsed.solver_trials == 4
    with pytest.raises(SystemExit):
        parser().parse_args(args + ['--solver-cpus', '17'])
