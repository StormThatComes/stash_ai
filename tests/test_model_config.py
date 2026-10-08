"""Model selection and prediction settings must be explicit in custom YAML."""
import pytest
import yaml

from stash_ai.config import load_model_settings
from stash_ai.models import SOURCES


@pytest.mark.parametrize('model', list(SOURCES))
@pytest.mark.parametrize('missing', ['scene_threshold', 'marker_threshold', 'batch_size'])
def test_selected_model_rejects_missing_prediction_settings(tmp_path, model, missing):
    values = dict(scene_threshold=.45, marker_threshold=.70, batch_size=4)
    del values[missing]
    path = tmp_path / 'models.yml'
    path.write_text(yaml.safe_dump(dict(version=1, enabled_models=[model], **{model: values})))
    with pytest.raises(ValueError, match=f'Missing {model} settings: {missing}'):
        load_model_settings(path, None, None)


@pytest.mark.parametrize('model', list(SOURCES))
def test_selected_model_requires_its_section(tmp_path, model):
    path = tmp_path / 'models.yml'
    path.write_text(yaml.safe_dump(dict(version=1, enabled_models=[model])))
    with pytest.raises(ValueError, match=f'Missing {model} settings'):
        load_model_settings(path, None, None)


def test_model_selection_has_no_implicit_joytag_fallback(tmp_path):
    path = tmp_path / 'models.yml'
    path.write_text('version: 1\n')
    with pytest.raises(ValueError, match='enabled_models is required'):
        load_model_settings(path, None, None)


def test_cli_selection_uses_explicit_values_without_requiring_other_models(tmp_path):
    path = tmp_path / 'models.yml'
    path.write_text(yaml.safe_dump(dict(version=1, wd=dict(
        scene_threshold=.52, marker_threshold=.73, batch_size=3))))
    settings = load_model_settings(path, ['wd'], 'cpu', offline=True)
    assert len(settings) == 1
    model = settings[0]
    assert (model.scene_threshold, model.marker_threshold, model.batch_size) == (.52, .73, 3)
    assert model.device == 'cpu' and not model.download_missing
    assert model.scene_cache_floor is model.marker_cache_floor is None


@pytest.mark.parametrize('yaml_value', [0, 4, 16, 'absent'])
@pytest.mark.parametrize('cli_value', [None, 0, 3, 16])
def test_prefetch_precedence_and_prediction_identity_are_independent(tmp_path, yaml_value, cli_value):
    from stash_ai.commands.arguments import parse_args
    from stash_ai.runtime.context import RuntimeContext
    from stash_ai.inference.identity import model_identity
    from stash_ai.models import Vocabulary
    app = RuntimeContext()
    path = tmp_path / 'models.yml'
    data = dict(version=1, enabled_models=['wd'], wd=dict(scene_threshold=.4, marker_threshold=.7, batch_size=4))
    path.write_text(yaml.safe_dump(data))
    identity = model_identity(app, load_model_settings(path, None, None)[0], Vocabulary(['one'], {}))
    if yaml_value != 'absent':
        data['prefetch_batches'] = yaml_value
    path.write_text(yaml.safe_dump(data))
    command = ['--model-config', str(path)]
    if cli_value is not None:
        command += ['--prefetch-batches', str(cli_value)]
    args = parse_args(app, command)
    assert args.prefetch_batches == (cli_value if cli_value is not None else 4 if yaml_value == 'absent' else yaml_value)
    assert model_identity(app, load_model_settings(path, None, None)[0], Vocabulary(['one'], {})) == identity


@pytest.mark.parametrize('value', [True, False, -1, 17, None, 1.2, '4', [], {}])
def test_yaml_rejects_invalid_prefetch_values(tmp_path, value):
    path = tmp_path / 'models.yml'
    path.write_text(yaml.safe_dump(dict(version=1, prefetch_batches=value, wd=dict(
        scene_threshold=.4, marker_threshold=.7, batch_size=4))))
    with pytest.raises(ValueError, match='prefetch_batches'):
        load_model_settings(path, ['wd'], None)


@pytest.mark.parametrize('name,mode', [
    ('prepare_workers', []), ('merge_workers', ['--merge-only']),
    ('model_report_workers', ['--model-report']),
])
@pytest.mark.parametrize('yaml_value', [1, 7, 16, 'absent'])
@pytest.mark.parametrize('cli_value', [None, 1, 3, 16])
def test_worker_configuration_precedence_and_inference_identity(tmp_path, name, mode, yaml_value, cli_value):
    import os
    from stash_ai.commands.arguments import parse_args
    from stash_ai.runtime.context import RuntimeContext
    from stash_ai.inference.identity import model_identity
    from stash_ai.models import Vocabulary
    app = RuntimeContext()
    path = tmp_path / 'models.yml'
    data = dict(version=1, enabled_models=['wd'], wd=dict(scene_threshold=.4, marker_threshold=.7, batch_size=4))
    path.write_text(yaml.safe_dump(data))
    vocabulary = Vocabulary(['one'], {})
    identity = model_identity(app, load_model_settings(path, None, None)[0], vocabulary)
    if yaml_value != 'absent':
        data[name] = yaml_value
    path.write_text(yaml.safe_dump(data))
    command = ['--model-config', str(path), *mode]
    if mode == ['--model-report']:
        command += [str(tmp_path/'report.json')]
    if cli_value is not None:
        command += ['--' + name.replace('_', '-'), str(cli_value)]
    args = parse_args(app, command)
    default = min(4, os.cpu_count() or 1) if name == 'model_report_workers' else 4
    assert getattr(args, name) == (cli_value if cli_value is not None else default if yaml_value == 'absent' else yaml_value)
    assert model_identity(app, load_model_settings(path, None, None)[0], vocabulary) == identity


@pytest.mark.parametrize('name', ['prepare_workers', 'merge_workers', 'model_report_workers'])
@pytest.mark.parametrize('value', [True, False, 0, -1, 17, None, 1.2, '4', [], {}])
def test_yaml_rejects_invalid_worker_counts(tmp_path, name, value):
    from stash_ai.config import load_model_configuration
    path = tmp_path/'models.yml'
    path.write_text(yaml.safe_dump(dict(version=1, **{name: value})))
    with pytest.raises(ValueError, match=name):
        load_model_configuration(path)


@pytest.mark.parametrize('mode', ['--merge-only', '--model-report'])
def test_worker_settings_preserve_saved_evidence_commands_without_model_yaml(tmp_path, mode):
    from stash_ai.commands.arguments import parse_args
    from stash_ai.runtime.context import RuntimeContext
    command = [mode, *([str(tmp_path/'report.json')] if mode == '--model-report' else []),
               '--model-config', str(tmp_path/'missing.yml')]
    args = parse_args(RuntimeContext(), command)
    assert args.merge_workers == 4 and args.model_report_workers >= 1


@pytest.mark.parametrize('mode', ['--db-check', '--run-report', '--reuse-report', '--prune-run-history'])
def test_unrelated_offline_commands_do_not_read_worker_configuration(tmp_path, monkeypatch, mode):
    import stash_ai.config as config
    from stash_ai.commands.arguments import parse_args
    from stash_ai.runtime.context import RuntimeContext
    path = tmp_path/'invalid.yml'
    path.write_text('not: model configuration\n')
    monkeypatch.setattr(config, 'load_model_configuration',
                        lambda _: pytest.fail('Unused worker configuration was read'))
    command = [mode, '--model-config', str(path)]
    if mode in ('--run-report', '--reuse-report'):
        command.insert(1, str(tmp_path/'report.json'))
    if mode == '--prune-run-history':
        command += ['--history-keep-last', '10']
    parse_args(RuntimeContext(), command)


def test_worker_configuration_is_read_once_for_inference(tmp_path, monkeypatch):
    import stash_ai.config as config
    from stash_ai.commands.arguments import parse_args
    from stash_ai.runtime.context import RuntimeContext
    calls = []
    monkeypatch.setattr(config, 'load_model_configuration', lambda path: (
        calls.append(path), dict(prefetch_batches=2, prepare_workers=3, merge_workers=5))[1])
    args = parse_args(RuntimeContext(), ['--model-config', str(tmp_path/'models.yml')])
    assert len(calls) == 1
    assert (args.prefetch_batches, args.prepare_workers, args.merge_workers) == (2, 3, 5)
