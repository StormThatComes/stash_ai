"""Offline health inspection isolates I/O failures and detects changing assets."""
import hashlib
import json
import os
from pathlib import Path

import pytest

from stash_ai.models import ModelSettings, SOURCES
from stash_ai.reports.model_assets import build_report, _file_status


def settings(root):
    joy, wd = root/'joy', root/'wd'
    joy.mkdir(); wd.mkdir()
    (joy/'top_tags.txt').write_text('one\n')
    (joy/'config.json').write_text('{}')
    (joy/'model.safetensors').write_bytes(b'weights')
    (wd/'selected_tags.csv').write_text('name,category\none,0\n')
    (wd/'model.onnx').write_bytes(b'weights')
    return [ModelSettings(SOURCES['joytag'], joy, .4, .4, 4), ModelSettings(SOURCES['wd'], wd, .4, .4, 4)]


def test_stable_assets_keep_hashes_and_sources_unchanged(tmp_path):
    values = settings(tmp_path)
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in tmp_path.glob('*/*')}
    report = build_report(values)
    assert all(model['healthy'] and model['labels'] == 1 for model in report['models'])
    assert report['models'][0]['files']['top_tags.txt']['sha256'] == hashlib.sha256(b'one\n').hexdigest()
    assert '_identity' not in json.dumps(report)
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before}


@pytest.mark.parametrize('phase', ['stat', 'open', 'read'])
def test_per_file_io_errors_do_not_hide_independent_models(tmp_path, monkeypatch, phase):
    values = settings(tmp_path)
    target = values[0].model_dir/'model.safetensors'
    original_stat, original_open = Path.stat, Path.open
    def denied_stat(self, *args, **kwargs):
        if self == target and phase == 'stat': raise PermissionError('PRIVATE_PATH_OR_TOKEN')
        return original_stat(self, *args, **kwargs)
    class Reader:
        def __init__(self, stream): self.stream = stream
        def __enter__(self): return self
        def __exit__(self, *args): self.stream.close()
        def fileno(self): return self.stream.fileno()
        def read(self, size): raise OSError('PRIVATE_PATH_OR_TOKEN')
    def denied_open(self, *args, **kwargs):
        if self == target:
            if phase == 'open': raise PermissionError('PRIVATE_PATH_OR_TOKEN')
            if phase == 'read': return Reader(original_open(self, *args, **kwargs))
        return original_open(self, *args, **kwargs)
    monkeypatch.setattr(Path, 'stat', denied_stat)
    monkeypatch.setattr(Path, 'open', denied_open)
    report = build_report(values)
    first, second = report['models']
    assert not first['healthy'] and second['healthy']
    assert first['files']['model.safetensors']['status'] == 'read_error'
    assert first['files']['model.safetensors']['error'] in ('OSError', 'PermissionError')
    assert 'PRIVATE_PATH_OR_TOKEN' not in json.dumps(report)


@pytest.mark.parametrize('replacement', [False, True])
def test_metadata_change_between_hash_and_parsing_is_never_healthy(tmp_path, monkeypatch, replacement):
    from stash_ai.backends import vocabulary
    values = settings(tmp_path)
    target = values[0].model_dir/'top_tags.txt'
    initial = target.stat()
    original = vocabulary.load_vocabulary
    def changed(setting):
        if setting.source.key == 'joytag':
            if replacement:
                alternate = target.with_suffix('.new')
                alternate.write_text('two\n')  # Same size and restored mtime.
                os.utime(alternate, ns=(initial.st_atime_ns, initial.st_mtime_ns))
                alternate.replace(target)
            else:
                target.write_text('one\ntwo\n')
        return original(setting)
    monkeypatch.setattr(vocabulary, 'load_vocabulary', changed)
    report = build_report(values)
    first, second = report['models']
    assert not first['healthy'] and second['healthy']
    assert first['metadata'] == 'changed_during_scan' and first['labels'] is None
    assert first['files']['top_tags.txt']['status'] == 'changed_during_scan'


def test_weight_changed_while_hashing_is_reported_unstable(tmp_path, monkeypatch):
    values = settings(tmp_path)
    target = values[0].model_dir/'model.safetensors'
    original = Path.open
    class Reader:
        def __init__(self, stream): self.stream, self.changed = stream, False
        def __enter__(self): return self
        def __exit__(self, *args): self.stream.close()
        def fileno(self): return self.stream.fileno()
        def read(self, size):
            data = self.stream.read(size)
            if not self.changed:
                with original(target, 'wb') as writer: writer.write(b'replaced weight bytes')
                self.changed = True
            return data
    def opened(self, *args, **kwargs):
        stream = original(self, *args, **kwargs)
        return Reader(stream) if self == target and args == ('rb',) else stream
    monkeypatch.setattr(Path, 'open', opened)
    report = build_report(values)
    assert report['models'][0]['files']['model.safetensors']['status'] == 'changed_during_scan'
    assert not report['models'][0]['healthy'] and report['models'][1]['healthy']


def test_missing_empty_invalid_metadata_and_cli_failure_status(tmp_path, monkeypatch, capsys):
    import stash_ai.reports.model_assets as assets
    values = settings(tmp_path)
    (values[0].model_dir/'model.safetensors').write_bytes(b'')
    (values[1].model_dir/'model.onnx').unlink()
    (values[1].model_dir/'selected_tags.csv').write_text('not,a,vocabulary\n')
    monkeypatch.setattr(assets, 'load_model_settings', lambda *args, **kwargs: values)
    from types import SimpleNamespace
    args = SimpleNamespace(model_config=None, models=['all'], model_assets_report=tmp_path/'report.json')
    assert assets.run(None, args) == 1
    report = json.loads(args.model_assets_report.read_text())
    assert report['models'][0]['files']['model.safetensors']['status'] == 'empty'
    assert report['models'][1]['files']['model.onnx']['status'] == 'missing'
    assert report['models'][1]['metadata'] == 'invalid_ValueError'
    assert 'No inference backends were loaded' in capsys.readouterr().out


def test_diagnostic_reports_assets_despite_invalid_or_missing_inference_tuning(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from stash_ai.config import load_model_settings
    from stash_ai.reports.model_assets import run
    values = settings(tmp_path)
    config = tmp_path/'models.yml'
    config.write_text('version: 1\nenabled_models: [joytag, wd]\ndevice: invalid\n'
                      'checkpoint_batches: false\nprefetch_batches: -1\nprepare_workers: []\n'
                      'joytag: {model_dir: joy, batch_size: 0, scene_threshold: wrong}\n'
                      'wd: {model_dir: wd, marker_cache_floor: false}\n')
    with pytest.raises(ValueError): load_model_settings(config, None, None, offline=True)
    # No backend, download or connection is involved in this command.
    destination = tmp_path/'assets.json'
    args = SimpleNamespace(model_config=config, models=None, model_assets_report=destination)
    assert run(None, args) == 0
    assert all(row['healthy'] for row in json.loads(destination.read_text())['models'])
    config.write_text('version: 1\nenabled_models: [joytag]\njoytag: {model_dir: []}\n')
    with pytest.raises(ValueError, match='path'): run(None, args)


@pytest.mark.parametrize('value', [[], {}, True, '0', .0, None])
def test_dtq_category_types_have_controlled_diagnostic_failure(tmp_path, value):
    from stash_ai.backends.vocabulary import load_vocabulary
    directory = tmp_path/'dtq'
    directory.mkdir()
    (directory/'tag_to_id.json').write_text('{"one":0}')
    (directory/'tag_category.json').write_text(json.dumps({'one':value}))
    (directory/'config.json').write_text('{"num_classes":1,"image_size":448}')
    (directory/'model.onnx').write_bytes(b'weights')
    setting = ModelSettings(SOURCES['dtq'], directory, .4, .4, 1, download_missing=False)
    with pytest.raises(ValueError, match='integer category'): load_vocabulary(setting)
    report = build_report([setting])
    assert report['models'][0]['metadata'] == 'invalid_ValueError'
    (directory/'tag_category.json').write_text('{"one":999}')
    assert load_vocabulary(setting).categories == {'one':'Uncategorized'}
    (directory/'tag_category.json').write_text('{}')
    assert load_vocabulary(setting).categories == {'one':'Uncategorized'}
