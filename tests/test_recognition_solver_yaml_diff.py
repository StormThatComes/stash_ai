import json

import pytest
import yaml

from benchmarks.benchmark_merge_solver import fixture
from stash_ai.recognition.cli import main, parser
from stash_ai.recognition.config import plain, digest, _merge, validate
from stash_ai.recognition.fragmentation_report import _open_readonly
from stash_ai.recognition.merge_goal_output import configuration_document, export, summary
from stash_ai.recognition.merge_goals import evaluate
from stash_ai.recognition.merge_report_policy import patched, proposal
from stash_ai.recognition.merge_solver import solve
from stash_ai.recognition.merge_solver_checkpoint import winner_file


def arguments(config, goals, destination, maximum=3):
    config.source_path.write_text(yaml.safe_dump(plain(config.data)))
    argv = ['report-merge', '--face-config', str(config.source_path), '--faces-db', str(config.database),
            '--solve', '--max-trials', str(maximum), '--yaml-output', str(destination), '--yaml-mode', 'diff',
            '--progress-style', 'off']
    for group in goals['groups']:
        argv.extend(['--performer-set', ','.join(m['performer_id'] for m in group['members'])])
    return argv


def test_diff_preserves_nested_changed_values_and_excludes_unchanged_settings(tmp_path):
    config, _ = fixture(tmp_path, 'query')
    changes = {'sapiensid.matching.face_floor': .6, 'sapiensid.matching.min_support': 1,
               'recognition.ambiguity_margin': 0., 'clustering.anchor_view_matching': False}
    recommendation = proposal(config, changes)
    # Even a redundant entry in an API result must not leak into the diff.
    recommendation['changes'].append(dict(parameter='models.directory', proposed=config.data['models']['directory']))
    document = configuration_document(config, recommendation, {}, mode='diff')
    assert yaml.safe_load(document) == {
        'clustering': {'anchor_view_matching': False},
        'recognition': {'ambiguity_margin': 0.},
        'sapiensid': {'matching': {'face_floor': .6, 'min_support': 1}},
    }
    assert digest(plain(config.data)) in document
    merged = plain(config.data)
    _merge(merged, yaml.safe_load(document))
    validate(merged)
    assert merged == plain(patched(config, changes).data)
    with pytest.raises(ValueError):
        configuration_document(config, {'changes': [dict(parameter='sapiensid.matching.face_floor', proposed=2.)]}, {}, mode='diff')


@pytest.mark.parametrize('maximum', [1, 3])
def test_cli_diff_matches_winner_against_input_and_has_no_standalone_run_command(tmp_path, capsys, maximum):
    config, goals = fixture(tmp_path/'input', 'query')
    destination = tmp_path/'diff.yml'
    argv = arguments(config, goals, destination, maximum)
    original = config.source_path.read_bytes()
    assert main(argv) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert not captured.err
    assert yaml.safe_load(destination.read_text()) == result['best_result']['yaml_patch']
    assert result['yaml_export_mode'] == 'diff' and 'run_argv' not in result
    assert result['yaml_diff_baseline']['configuration_path'] == str(config.source_path)
    assert result['yaml_diff_baseline']['configuration_sha256'] == digest(plain(config.data))
    assert config.source_path.read_bytes() == original
    if maximum == 1:
        assert result['selected_trial'] == 'baseline' and yaml.safe_load(destination.read_text()) == {}
        assert 'Unmet conditions:' in destination.read_text()
    else:
        assert result['selected_trial'] != 'baseline'
        changes = {c['parameter']: c['proposed'] for c in result['best_result']['changes']}
        with _open_readonly(config.database) as db:
            repeated = evaluate(patched(config, changes), goals, db)['trials']['baseline']['goal_result']
        assert repeated['fully_solved']


def test_diff_is_live_before_interrupt_and_resumes_without_switching_format(tmp_path):
    config, goals = fixture(tmp_path/'input', 'query')
    path, destination = tmp_path/'checkpoint.json', tmp_path/'diff.yml'
    owner = winner_file(destination)
    snapshots = []
    def progress(name, stage, item):
        if stage == 'complete':
            snapshots.append(yaml.safe_load(destination.read_text()))
            if name == 'trial-1':
                raise KeyboardInterrupt
    with _open_readonly(config.database) as db:
        result = dict(goals, **solve(config, goals, db, maximum=4, checkpoint=path, winner_writer=owner,
                                    yaml_mode='diff', progress=progress))
    assert result['interrupted'] and snapshots[0] == {}
    assert snapshots[-1] == result['best_result']['yaml_patch'] and snapshots[-1]
    export(config, result, tmp_path/'result.json', destination, yaml_owner=owner, yaml_mode='diff')
    original = destination.read_bytes()
    with _open_readonly(config.database) as db:
        with pytest.raises(ValueError, match='incompatible'):
            solve(config, goals, db, maximum=4, checkpoint=path, resume=True,
                  winner_writer=winner_file(destination, resume=True), yaml_mode='full')
        assert destination.read_bytes() == original
        resumed = solve(config, goals, db, maximum=4, checkpoint=path, resume=True,
                        winner_writer=winner_file(destination, resume=True), yaml_mode='diff')
    assert resumed['search']['restored_trials'] == 2
    assert yaml.safe_load(destination.read_text()) == resumed['best_result']['yaml_patch']


def test_diff_works_with_automatic_checkpoint_yaml_path(tmp_path, capsys):
    config, goals = fixture(tmp_path/'input', 'query')
    path = tmp_path/'checkpoint.json'
    argv = arguments(config, goals, path)
    argv[argv.index('--yaml-output')] = '--checkpoint'
    assert main(argv) == 0
    result = json.loads(capsys.readouterr().out)
    assert yaml.safe_load(path.with_name(path.name+'.best.yml').read_text()) == result['best_result']['yaml_patch']
    assert main(argv+['--resume']) == 0
    assert json.loads(capsys.readouterr().out)['search']['restored_trials'] == 3


def test_partial_scope_diff_keeps_unmet_explanation_without_run_argv(tmp_path):
    from tests.test_recognition_merge_report import seeded
    from tests.test_recognition_merge_solver_inputs import supersede
    from stash_ai.recognition.merge_goals import resolve
    config, people = seeded(tmp_path)
    supersede(config, '2', reason='policy')
    with _open_readonly(config.database) as db:
        goals = resolve(db, config, [people], by_performer=False)
        result = dict(goals, **solve(config, goals, db, maximum=3))
    destination = tmp_path/'diff.yml'
    export(config, result, tmp_path/'report.json', destination, yaml_mode='diff')
    document = destination.read_text()
    assert 'best_effort' in document and 'Unmet conditions:' in document and 'Partial input scope:' in document
    assert yaml.safe_load(document) == result['best_result']['yaml_patch']
    assert 'run_argv' not in document
    text = summary(result)
    assert 'YAML diff saved' in text and 'simulation.selection' in text and 'run_argv' not in text


@pytest.mark.parametrize('reason', ['interrupted', 'no_compatible_scans'])
def test_diff_without_completed_baseline_creates_no_yaml(tmp_path, reason):
    from tests.test_recognition_merge_solver_inputs import supersede
    config, goals = fixture(tmp_path/'input', 'query')
    if reason == 'no_compatible_scans':
        for scene in ('1', '2'):
            supersede(config, scene, reason='policy')
    def progress(*args):
        raise KeyboardInterrupt
    destination = tmp_path/'diff.yml'
    owner = winner_file(destination)
    with _open_readonly(config.database) as db:
        result = dict(goals, **solve(config, goals, db, maximum=3, yaml_mode='diff', winner_writer=owner,
                                    progress=progress if reason == 'interrupted' else None))
    export(config, result, tmp_path/'report.json', destination, yaml_owner=owner, yaml_mode='diff')
    assert not destination.exists() and not result['yaml_exported']


@pytest.mark.parametrize('extra', [[], ['--solve'], ['--yaml-output', 'diff.yml']])
def test_diff_requires_solver_and_a_destination(extra):
    with pytest.raises(SystemExit):
        parser().parse_args(['report-merge', '--performer-set', '1,2', '--yaml-mode', 'diff', *extra])
