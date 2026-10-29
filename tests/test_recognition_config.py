from pathlib import Path

import pytest

from stash_ai.recognition.config import defaults, load, parse_yaml, validate


def test_removed_body_time_gate_requires_explicit_config_cleanup(tmp_path):
    p=tmp_path/'old.yml'
    p.write_text('version: 1\nsapiensid:\n  identities:\n    min_seconds: 1\n')
    with pytest.raises(ValueError,match='min_seconds was removed; delete this key'):
        load(p)
    assert 'min_seconds' not in defaults()['sapiensid']['identities']


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


@pytest.mark.parametrize('limit', [1073741825, 8589934592, 68719476736])
def test_large_reference_budget_loads_and_remains_enforced(tmp_path, limit):
    from stash_ai.recognition.reference_usage import ReferenceUsage
    path = tmp_path / 'large-references.yml'
    path.write_text(f'version: 1\nclustering: {{max_reference_bytes: {limit}}}\n')
    config = load(path)
    usage = ReferenceUsage(config.data['clustering']['max_reference_bytes'])
    # Exercise accounting above the old ceiling without allocating large arrays.
    usage.add_counts(limit, 1, people=1, kind='Test')
    assert usage.bytes == limit
    with pytest.raises(ValueError, match='clustering.max_reference_bytes'):
        usage.add_counts(1, 1, kind='Test')


@pytest.mark.parametrize('limit', [1048575, 68719476737])
def test_reference_budget_outside_supported_range_is_rejected(tmp_path, limit):
    path = tmp_path / 'invalid-references.yml'
    path.write_text(f'version: 1\nclustering: {{max_reference_bytes: {limit}}}\n')
    with pytest.raises(ValueError, match='clustering.max_reference_bytes'):
        load(path)


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
    path = Path(__file__).resolve().parents[1] / 'facial_recognition.yml'
    # Check the file before load() fills missing fields with defaults.
    supplied = parse_yaml(path.read_bytes())
    def paths(mapping, prefix=''):
        found = set()
        for key, value in mapping.items():
            name = prefix + key
            found.add(name)
            if isinstance(value, dict):
                found.update(paths(value, name + '.'))
        return found
    expected = defaults()
    actual_paths, expected_paths = paths(supplied), paths(expected)
    assert actual_paths == expected_paths, dict(
        missing=sorted(expected_paths-actual_paths), unknown=sorted(actual_paths-expected_paths))
    validate(supplied)
    assert supplied['sapiensid'] == expected['sapiensid']
    assert supplied['execution']['portrait_ranking_cache_bytes'] == expected['execution']['portrait_ranking_cache_bytes']
    config = load(path)
    assert config.data['models']['detector'] == 'scrfd_10g'
    assert config.data['recognition']['min_identity_support'] == 2
    assert config.data['recognition']['cross_scene_provisional'] is True
    assert config.data['clustering']['context_view_enrichment'] is True
    assert config.data['recognition']['min_merge_support'] == 3
    assert config.data['clustering']['same_scene_match_bonus'] == .03
    assert config.data['clustering']['same_scene_min_support'] == 3
    assert config.data['portraits']['source'] == 'vtt'
    assert config.data['portraits']['margin'] == .35
    assert config.data['execution']['prefetch_frames'] == 4
    assert config.data['execution']['prefetch_scenes'] is True


@pytest.mark.parametrize('settings', [
    'prefetch_frames: -1', 'prefetch_frames: 129', 'prefetch_frames: true',
    'prefetch_frames: 1.5', 'prefetch_scenes: 1', 'prefetch_scenes: null',
])
def test_reject_invalid_prefetch_configuration(tmp_path, settings):
    path = tmp_path / 'invalid.yml'
    path.write_text(f'version: 1\nexecution: {{{settings}}}\n')
    with pytest.raises(ValueError, match='execution.prefetch'):
        load(path)


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


def test_split_support_defaults_singletons_legacy_alias_and_validation(tmp_path):
    config = load(root=tmp_path)
    assert config.data['recognition']['min_identity_support'] == 1
    assert config.data['recognition']['min_merge_support'] == 3
    for key in ('min_identity_support', 'min_merge_support'):
        valid = tmp_path / (key + '.yml')
        valid.write_text(f'version: 1\nrecognition: {{{key}: 1}}\n')
        assert load(valid).data['recognition'][key] == 1
        valid.write_text(f'version: 1\nrecognition: {{{key}: 0}}\n')
        with pytest.raises(ValueError, match=key):
            load(valid)
    old = tmp_path / 'old.yml'
    old.write_text('version: 1\nrecognition: {min_support: 2}\n')
    assert load(old).data['recognition']['min_identity_support'] == 2
    assert load(old).data['recognition']['min_merge_support'] == 2
    old.write_text('version: 1\nrecognition: {min_support: 2, min_merge_support: 3}\n')
    with pytest.raises(ValueError, match='cannot be combined'):
        load(old)


import json
import sqlite3
import uuid
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest


def test_recognition_numeric_validation_reports_oversized_integer_field(tmp_path):
    from stash_ai.recognition.config import load
    path = tmp_path / 'face.yml'
    path.write_text('version: 1\nexecution: {checkpoint_cues: ' + str(10**1000) + '}\n')
    with pytest.raises(ValueError, match='execution.checkpoint_cues'):
        load(path)

