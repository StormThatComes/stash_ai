"""The shareable JSON keeps replay outcomes and blockers without observation histories."""
import json

import pytest
import yaml

from benchmarks.benchmark_merge_solver import fixture
from stash_ai.recognition.cli import main, parser
from stash_ai.recognition.config import plain
from stash_ai.recognition.merge_solver_report import compact_document


@pytest.mark.parametrize('first_mode', ['compact', 'full'])
def test_cli_report_modes_share_checkpoint_winner_and_blockers(tmp_path, capsys, first_mode):
    config, _ = fixture(tmp_path/'input', 'disjoint')
    config.source_path.write_text(yaml.safe_dump(plain(config.data)))
    checkpoint = tmp_path/'checkpoint.json'
    winner = tmp_path/'winner.yml'
    args = ['report-merge', '--solve', '--face-config', str(config.source_path), '--faces-db', str(config.database),
            '--performer-set', '900,901', '--performer-set', '902,903', '--max-trials', '8',
            '--progress-style', 'off', '--checkpoint', str(checkpoint), '--yaml-output', str(winner)]
    reports = {}
    for mode in (first_mode, 'full' if first_mode == 'compact' else 'compact'):
        path = tmp_path/'full.json'
        extra = ['--solver-report', 'full', '--output', str(path)] if mode == 'full' else []
        if reports:
            extra.append('--resume')
        assert main(args + extra) == 0
        stdout = capsys.readouterr().out
        reports[mode] = json.loads(path.read_text() if mode == 'full' else stdout)
        if len(reports) == 1:
            first_yaml = winner.read_bytes()
        else:
            assert winner.read_bytes() == first_yaml
    compact, full = reports['compact'], reports['full']
    assert compact['solver_report_format'] == 'compact-v1'
    assert 'solver_report_format' not in full
    assert compact['selected_trial'] == full['selected_trial']
    assert compact['best_result']['changes'] == full['best_result']['changes']
    assert compact['provenance'] == full['provenance']
    trials = compact['simulation']['trials']
    raw = full['simulation']['trials']
    assert list(trials) == list(raw)
    states = compact['merge_diagnostics']['set_states']
    for name, row in trials.items():
        assert row.get('yaml_patch') == raw[name].get('yaml_patch')
        assert row['goal_result']['fully_solved'] == raw[name]['goal_result']['fully_solved']
        assert 'decision_evidence' in raw[name] and 'decision_evidence' not in row
        assert 'hybrid_solver_evidence' not in row and 'independent_person' not in row
        for ref, construction in zip(row['goal_result']['set_states'], raw[name]['goal_result']['construction']['groups'], strict=True):
            assert states[ref]['index'] == construction['index']
            assert states[ref]['observed_blockers'] == construction['observed_blockers']
    diagnosis = compact['merge_diagnostics']['sets'][0]
    assert diagnosis['merged_trials'] + diagnosis['unmerged_trials'] == len(trials)
    assert any(b['reason'] == 'face_floor' for b in diagnosis['observed_blockers'])
    measured = next(e for e in trials['baseline']['solver_feedback']['examples'] if e['source'] == 'measured_query')
    assert measured['face']['minimum'] < measured['thresholds']['face_floor']
    assert len(json.dumps(compact)) < len(json.dumps(full)) / 4

    # Scale only the history. Extra repeated decision payloads must never enter compact output.
    full['simulation']['trials'] = {f'trial-{i}': raw['baseline'] for i in range(1001)}
    scaled = compact_document(full, config)
    assert len(scaled['simulation']['trials']) == 1001
    assert scaled['merge_diagnostics']['sets'][0]['unmerged_trials'] == 1001
    assert len(scaled['merge_diagnostics']['set_states']) == len(full['groups'])
    assert len(json.dumps(scaled, indent=2)) < 8 * 1024**2


def test_solver_report_option_is_scoped_to_solve():
    with pytest.raises(SystemExit):
        parser().parse_args(['report-merge', '--performer-set', '900,901', '--solver-report', 'compact'])
