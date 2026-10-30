"""Trial mutations follow executed configurations, including branch reversions."""
import json

import pytest
import yaml

from tests.test_recognition_merge_solver_exploration import auxiliary_goals
from stash_ai.recognition.config import load, plain
from stash_ai.recognition.fragmentation_report import _open_readonly
from stash_ai.recognition.merge_goals import evaluate
from stash_ai.recognition.merge_solver_search import BeamSearch, PREFIX


def scripted_search(monkeypatch):
    # Both executed candidates come from baseline. Intermediate proposals must
    # not advance the comparison point, including a failed validation.
    changes = [{PREFIX+'face_floor':.785}, {PREFIX+'face_floor':.785},
               {PREFIX+'face_floor':float('nan')}, {}, {'clustering.max_anchor_pose_deviation':.4}]
    pending = iter(changes)
    def next_trial(self):
        patch = next(pending)
        family = 'hybrid' if any(k.startswith(PREFIX) for k in patch) else 'face'
        return family, 'baseline_exploration', 'baseline', 'coordinate_probe', patch
    monkeypatch.setattr(BeamSearch, 'next', next_trial)


def arguments(config, goals, tmp_path, style='auto'):
    config.source_path.write_text(yaml.safe_dump(plain(config.data)))
    argv = ['report-merge','--face-config',str(config.source_path),'--faces-db',str(config.database),
            '--solve','--targeted-trials','0','--feasibility-trials','0','--max-trials','3','--yaml-output',str(tmp_path/'winner.yml'),'--progress-style',style]
    for group in goals['groups']:
        argv.extend(['--performer-set',','.join(m['performer_id'] for m in group['members'])])
    return argv


@pytest.mark.parametrize('style', ['auto', 'off'])
def test_cli_logs_previous_executed_config_and_keeps_export_relative_to_input(tmp_path, monkeypatch, capsys, style):
    from stash_ai.recognition.cli import main
    config, goals = auxiliary_goals(tmp_path, sets=2)
    scripted_search(monkeypatch)
    assert main(arguments(config, goals, tmp_path, style)) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)  # Progress never contaminates JSON stdout.
    trials = result['simulation']['trials']
    assert list(trials) == ['baseline','trial-1','trial-2']
    assert trials['baseline']['previous_trial'] is None
    assert trials['baseline']['changes_from_previous'] == []
    assert trials['trial-1']['previous_trial'] == 'baseline'
    assert trials['trial-1']['changes_from_previous'] == [
        dict(parameter=PREFIX+'face_floor', previous=.8, current=.785)]
    assert trials['trial-2']['previous_trial'] == 'trial-1'
    assert trials['trial-2']['search_parent'] == 'baseline'
    assert trials['trial-2']['changes_from_previous'] == [
        dict(parameter='clustering.max_anchor_pose_deviation', previous=.2, current=.4),
        dict(parameter=PREFIX+'face_floor', previous=.785, current=.8)]
    # Export patches still describe the input YAML, not the last search trial.
    assert trials['trial-2']['yaml_patch'] == {'clustering': {'max_anchor_pose_deviation':.4}}
    assert result['selected_trial'] == 'trial-1'
    assert result['best_result']['yaml_patch'] == {'sapiensid': {'matching': {'face_floor':.785}}}
    if style == 'off':
        assert captured.err == ''
    else:
        assert 'baseline: start (input configuration)' in captured.err
        assert ('trial-1: start changes_vs=baseline parent=baseline\n'
                '  sapiensid.matching.face_floor: 0.8 -> 0.785\n') in captured.err
        assert ('trial-2: start changes_vs=trial-1 parent=baseline\n'
                '  clustering.max_anchor_pose_deviation: 0.2 -> 0.4\n'
                '  sapiensid.matching.face_floor: 0.785 -> 0.8\n') in captured.err
        assert 'nan' not in captured.err
        assert 'best=trial-1 best_number=1' in captured.err
        lines = captured.err.splitlines()
        for completed, (name, item) in enumerate(trials.items(), 1):
            heading = f"Merge solver {name}: complete disjoint={item['goal_result']['disjoint']} best={item['best_trial']} best_number={item['best_trial_number']}"
            heading += f" goal_policy=identity strict_disjoint={item['goal_result']['strict_disjoint']}"
            heading += f" elapsed={item['elapsed_seconds']:.1f}s completed={completed}/3"
            position = lines.index(heading)
            stop = next((i for i in range(position+1, len(lines)) if lines[i].startswith('Merge solver ')), len(lines))
            details = lines[position+1:stop]
            headers = [line for line in details if line.startswith('  Set ')]
            for line, group in zip(headers, item['goal_result']['groups'], strict=True):
                assert line.startswith(f"  Set {group['index']+1}: scenes_preserved={group['covered_scenes']}/{group['required_scenes']}")
                assert f"consolidated={group['consolidated_performers']}/{group['total_performers']}" in line
                assert f"largest_complete={group['largest_complete_performers']}/{group['total_performers']}" in line
                assert f"represented={group['represented_performers']}/{group['total_performers']}" in line
                assert f"complete_components={json.dumps(group['complete_components'], separators=(',', ':'))}" in line
                assert f"assigned={group['assigned']}/{group['observations']}" in line
                assert f"missing_obs={group['missing']} withheld_obs={group['withheld']}" in line
                assert 'split_performers=' in line
            assert sum('Face anchor:' in line for line in details) == 2
            assert sum('SapiensID auxiliary:' in line for line in details) == 2
            assert any('performer_ids=["900","901"]' in line for line in details)
            assert any('performer_ids=["902","903"]' in line for line in details)
        # The final summary uses the winner's set details, including after a losing last trial.
        summary_position = next(i for i, line in enumerate(lines) if line.startswith('Merge solver best completed trial:'))
        winner_position = next(i for i, line in enumerate(lines) if line.startswith('Merge solver trial-1: complete '))
        winner_stop = next(i for i in range(winner_position+1, len(lines)) if lines[i].startswith('Merge solver '))
        winner_details = [line for line in lines[winner_position+1:winner_stop] if not line.startswith('  Construction:')]
        assert lines[summary_position+2:summary_position+2+len(winner_details)] == winner_details
    with _open_readonly(config.database) as db:
        repeated = evaluate(load(tmp_path/'winner.yml'), goals, db)['trials']['baseline']['goal_result']
    assert repeated['fully_solved'] and repeated['disjoint']


def test_mutation_is_printed_before_rebuild_and_interrupt_keeps_prior_winner(tmp_path, monkeypatch, capsys):
    from stash_ai.recognition.cli import main
    from stash_ai.recognition import replay_simulation
    config, goals = auxiliary_goals(tmp_path)
    scripted_search(monkeypatch)
    original = replay_simulation.rebuild_selected
    calls = 0
    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            # Inspect the console at entry into the expensive replay, not after
            # the command finishes: even an interrupted trial explains its YAML.
            output = capsys.readouterr()
            assert 'trial-2: start changes_vs=trial-1 parent=baseline' in output.err
            assert 'sapiensid.matching.face_floor: 0.785 -> 0.8' in output.err
            assert output.out == ''
            raise KeyboardInterrupt
        return original(*args, **kwargs)
    monkeypatch.setattr(replay_simulation, 'rebuild_selected', interrupted)
    assert main(arguments(config, goals, tmp_path)) == 130
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result['interrupted'] and result['selected_trial'] == 'trial-1'
    assert set(result['simulation']['trials']) == {'baseline','trial-1'}
    assert 'trial-2: complete' not in captured.err
    assert load(tmp_path/'winner.yml').data['sapiensid']['matching']['face_floor'] == .785
