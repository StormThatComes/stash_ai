"""Retained inference follows weights and preprocessing, not file timestamps."""
from dataclasses import replace
import hashlib
import os

import pytest

from tests.helpers.runtime import runtime as app
from stash_ai.inference.identity import model_identity, asset_stamps
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
    stamps = asset_stamps(config)
    (tmp_path/'model.onnx').write_bytes(b'weights')
    after = model_identity(app, config, vocabulary, known_files=before['files'], known_stamps=stamps)
    assert after != before
    assert calls == [str(tmp_path/'selected_tags.csv'), str(tmp_path/'model.onnx')]
    assert model_identity(app, config, vocabulary, known_files=after['files'],
                          known_stamps=asset_stamps(config)) == after
    assert len(calls) == 2


def test_known_hash_without_change_guards_cannot_hide_replacement(tmp_path):
    weight = tmp_path/'model.onnx'
    weight.write_bytes(b'AAAA')
    config, vocabulary = ModelSettings(WD, tmp_path, .4, .7, 1), Vocabulary(['a'], {})
    before = model_identity(app, config, vocabulary)
    stamps, original = asset_stamps(config), weight.stat()
    weight.write_bytes(b'BBBB')
    os.utime(weight, ns=(original.st_atime_ns, original.st_mtime_ns))
    after = model_identity(app, config, vocabulary)
    assert after != before
    assert model_identity(app, config, vocabulary, known_files=before['files']) == after
    assert model_identity(app, config, vocabulary, known_files=before['files'], known_stamps=stamps) == after
    with pytest.raises(RuntimeError, match='asset changed'):
        model_identity(app, config, vocabulary, known_files=before['files'],
                       known_stamps=stamps, require_unchanged=True)


def test_change_during_hash_is_rejected(tmp_path, monkeypatch):
    weight = tmp_path/'model.onnx'
    weight.write_bytes(b'AAAA')
    config, vocabulary = ModelSettings(WD, tmp_path, .4, .7, 1), Vocabulary(['a'], {})
    original = hashlib.file_digest
    def digest(stream, algorithm):
        checksum = original(stream, algorithm)
        weight.write_bytes(b'BBBB')
        return checksum
    monkeypatch.setattr(hashlib, 'file_digest', digest)
    with pytest.raises(RuntimeError, match='changed while hashing'):
        model_identity(app, config, vocabulary)


@pytest.mark.parametrize('when', ['before', 'during'])
def test_asset_replacement_prevents_model_inference(model_run, monkeypatch, when):
    from stash_ai.runs.merged_model import MergedModelRun
    # Use the same lifecycle fixture as the existing merged-model tests.
    context = model_run
    context.config.model_dir.mkdir()
    weight = context.config.model_dir/'model.onnx'
    weight.write_bytes(b'AAAA')
    run = MergedModelRun(app, context.args, context.state, context.config,
                         context.pipeline.vocab, stats=context.stats, report=context.args.report)
    loader = app.services.load_tagger
    def load(*args):
        model = loader(*args)
        if when == 'during': weight.write_bytes(b'BBBB')
        return model
    monkeypatch.setattr(app.services, 'load_tagger', load)
    if when == 'before': weight.write_bytes(b'BBBB')
    run.run(context.scenes, ['scene', 'markers'], context.inputs)
    assert context.stats['processed'] == 0
    assert context.stats['errors'] == 1
    assert all(model.closed for model in context.pipeline.loaded)


from tests.test_merged_model import model_run  # Reuse the owned SQLite/backend fixture.
