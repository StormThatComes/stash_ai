from pathlib import Path

import pytest

from stash_ai.recognition.config import load, parse_yaml


def test_default_config_and_immutable_snapshot(tmp_path):
    p = tmp_path / 'facial_recognition.yml'
    p.write_text('version: 1\nrecognition: {match_threshold: 0.75}\n')
    config = load(p)
    p.write_text('version: 1\nrecognition: {match_threshold: 0.95}\n')
    assert config.data['recognition']['match_threshold'] == .75
    assert config.snapshot()['policy_origin'] == 'custom'
    assert config.database == tmp_path / 'faces_state.sqlite3'
    assert '.75' in config.snapshot()['source'] or '0.75' in config.snapshot()['source']
    with pytest.raises(TypeError):
        config.data['recognition']['match_threshold'] = .2


@pytest.mark.parametrize('text', [
    'version: 1\nversion: 1', 'version: true', 'version: 2', 'version: 1\nwrong: 1',
    'version: 1\nrecognition: {match_threshold: .nan}',
    'version: 1\nrecognition: {min_support: true}',
    'version: 1\nrecognition: {match_threshold: 0.4}',
    'version: 1\nmodels: {recognizer: unknown}', 'version: 1\nnames: {file: false}',
    'version: 1\nportraits: {source: unknown}', 'version: 1\nportraits: {source: false}',
    'version: 1\nportraits: {aspect_ratio: 0}', 'version: 1\nportraits: {aspect_ratio: .nan}',
    'version: 1\nportraits: {jpeg_quality: 101}', 'version: 1\nportraits: {jpeg_quality: true}',
    'version: 1\nmodels: &a {device: cpu}\nnames: *a',
])
def test_reject_bad_configuration_before_work(tmp_path, text):
    path = tmp_path / 'bad.yml'
    path.write_text(text)
    with pytest.raises(ValueError):
        load(path)
    assert not (tmp_path / 'faces_state.sqlite3').exists()


def test_missing_explicit_file_and_limits(tmp_path):
    assert load(root=tmp_path).source == b''
    assert load(root=tmp_path).snapshot()['policy_origin'] == 'bundled-default'
    with pytest.raises(FileNotFoundError):
        load(tmp_path / 'missing.yml')
    with pytest.raises(ValueError):
        parse_yaml(b'x' * 131073)


def test_semantic_cache_identity_ignores_filename_and_names(tmp_path):
    one, two = tmp_path / 'a.yml', tmp_path / 'b.yml'
    one.write_text('version: 1\n')
    two.write_text('version: 1\nnames: {file: names.yml}\nportraits: {min_sharpness: 99}\n')
    assert load(one).identity('models', 'detection') == load(two).identity('models', 'detection')
    assert load(one).identity('portraits') != load(two).identity('portraits')


def test_shipped_template_matches_default_schema():
    config = load(Path(__file__).resolve().parents[1] / 'facial_recognition.yml')
    assert config.data['recognition']['min_support'] == 3
    assert config.data['portraits']['source'] == 'vtt'
    assert config.data['portraits']['margin'] == .35


def test_portrait_source_and_framing_do_not_invalidate_face_scan(tmp_path):
    from stash_ai.recognition.models import namespace
    from stash_ai.recognition.scanning import scan_identity
    initial = load(root=tmp_path)
    path = tmp_path / 'changed.yml'
    path.write_text('version: 1\nportraits: {source: video, margin: 0.5, aspect_ratio: 0.8, jpeg_quality: 85}\n')
    changed = load(path)
    assert initial.data['portraits']['source'] == 'vtt'
    assert namespace(initial) == namespace(changed)
    assert scan_identity(initial, namespace(initial)) == scan_identity(changed, namespace(changed))
    assert initial.identity('portraits') != changed.identity('portraits')
