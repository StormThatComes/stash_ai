"""Retained inference follows weights and preprocessing, not file timestamps."""
from dataclasses import replace
import hashlib
import os

import pytest

from tests.helpers.runtime import runtime as app
from stash_ai.inference.identity import model_identity
from stash_ai.models import WD, JOYTAG, ModelSettings, Vocabulary

pytestmark = pytest.mark.usefixtures('reset_temporal')


@pytest.mark.parametrize('field', ['JOYTAG_CLIP_MEAN', 'JOYTAG_CLIP_STD'])
def test_joytag_normalization_changes_inference_identity(tmp_path, field):
    config = ModelSettings(JOYTAG, tmp_path, .4, .7, 1)
    vocabulary = Vocabulary(['core'], {})
    before = model_identity(app, config, vocabulary)
    changed = replace(app, **{field: [.1, .2, .3]})
    assert model_identity(changed, config, vocabulary) != before
    # Runtime list mutations cannot silently alter an already captured identity.
    getattr(changed, field)[0] = .8
    assert before['preprocessing'][field.removeprefix('JOYTAG_CLIP_').lower()][0] != .8
    assert model_identity(app, replace(config, source=WD), vocabulary) == model_identity(changed, replace(config, source=WD), vocabulary)


def test_weight_change_invalidates_even_when_size_and_timestamp_match(tmp_path):
    weight = tmp_path/'model.onnx'
    weight.write_bytes(b'AAAA')
    original = weight.stat()
    config = ModelSettings(WD, tmp_path, .4, .7, 1)
    vocabulary = Vocabulary(['a'], {})
    before = model_identity(app, config, vocabulary)
    weight.write_bytes(b'BBBB')
    os.utime(weight, ns=(original.st_atime_ns, original.st_mtime_ns))
    assert model_identity(app, config, vocabulary) != before
    after = model_identity(app, config, vocabulary)
    os.utime(weight, ns=(original.st_atime_ns, original.st_mtime_ns + 1000))
    assert model_identity(app, config, vocabulary) == after


def test_download_refresh_hashes_only_previously_missing_files(tmp_path, monkeypatch):
    config = ModelSettings(WD, tmp_path, .4, .7, 1)
    vocabulary = Vocabulary(['a'], {})
    (tmp_path/'selected_tags.csv').write_bytes(b'metadata')
    calls = []
    original = hashlib.file_digest
    def counted(stream, algorithm):
        calls.append(stream.name)
        return original(stream, algorithm)
    monkeypatch.setattr(hashlib, 'file_digest', counted)
    before = model_identity(app, config, vocabulary)
    (tmp_path/'model.onnx').write_bytes(b'weights')
    after = model_identity(app, config, vocabulary, known_files=before['files'])
    assert after != before
    assert calls == [str(tmp_path/'selected_tags.csv'), str(tmp_path/'model.onnx')]
    assert model_identity(app, config, vocabulary, known_files=after['files']) == after
    assert len(calls) == 2
