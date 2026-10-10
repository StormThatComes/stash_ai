"""Optimizations preserve checkpoints, native validation and legacy input hashes."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from stash_ai.backends.onnx import OnnxTagger
from stash_ai.domain import Cue
from stash_ai.evidence.temporal.statistics import SceneAccumulator
from stash_ai.inference import batches, checkpoints, prediction, preparation, results
from stash_ai.inference.input_identity import cue_manifest
from stash_ai.inference.inputs import VerifiedInput
from stash_ai.models import ModelSettings, WD
from stash_ai.runtime.profile import ProfileMetrics
from stash_ai.runtime.report import RunReport
from stash_ai.storage.store import StateStore
from tests.helpers.inference import IndexedTagger
from tests.helpers.runtime import runtime as app

pytestmark = pytest.mark.usefixtures('reset_progress')


def test_accumulator_snapshot_is_taken_only_at_checkpoint_flush(pipeline, tmp_path, monkeypatch):
    calls, original = [], SceneAccumulator.state
    def state(accumulator):
        calls.append(accumulator.frames)
        return original(accumulator)
    monkeypatch.setattr(SceneAccumulator, 'state', state)
    model = IndexedTagger(ModelSettings(WD, Path('missing'), .4, .65, 4),
                          pipeline.scores['wd'], pipeline.vocab.labels)
    with StateStore(tmp_path / 'state.db', 's') as store:
        checkpoint = checkpoints.Checkpoint(store, '1', 'wd', 'sig', 10, 2, every=2)
        prediction.predict_scene(app, model, pipeline.stash, {}, pipeline.cues, checkpoint=checkpoint)
        assert calls == [8, 10]
        restored = checkpoints.Checkpoint(store, '1', 'wd', 'sig', 10, 2)
        assert restored.restore()[0] == restored.analysis['frames'] == 10


def test_native_pipeline_and_direct_adapter_each_validate_once(monkeypatch):
    from stash_ai import models
    native = OnnxTagger.__new__(OnnxTagger)
    native.labels = ['tag']
    native.settings = SimpleNamespace(source=WD)
    native.collate_batch = lambda images: images
    native._predict_collated_scores = lambda inputs, count: np.full((count, 1), .5, np.float32)
    calls, validate = [], models.validate_scores
    def checked(*args):
        calls.append(1)
        return validate(*args)
    monkeypatch.setattr(models, 'validate_scores', checked)
    monkeypatch.setattr(batches, 'validate_scores', checked)
    batches.prediction_scores(native, [object()])
    assert calls == [1]
    native.predict_batch([object()])
    assert calls == [1, 1]


@pytest.mark.parametrize('value', [float('nan'), -1., 1.1])
def test_external_adapter_output_is_still_validated(value):
    tagger = SimpleNamespace(labels=['tag'], settings=SimpleNamespace(source=WD),
                             predict_batch=lambda _: np.array([[value]], np.float32))
    with pytest.raises(RuntimeError, match='invalid probabilities'):
        batches.prediction_scores(tagger, [object()])


def test_verified_input_preserves_legacy_hash_and_reuses_workload_digest(monkeypatch):
    cues = [Cue(0., 1., 'żółć.jpg', (0, 0, 1, 1))]
    hashes = {'żółć.jpg': 'checksum'}
    snapshot = VerifiedInput.capture(cues, hashes)
    legacy = hashlib.sha256(json.dumps(cue_manifest(cues), ensure_ascii=False).encode()).hexdigest()
    assert snapshot.legacy_cue_sha256 == legacy and legacy != snapshot.cue_digest
    first, second = RunReport(), RunReport()
    first.profile = second.profile = ProfileMetrics()
    first.record_input_workload('1', 'wd', cues, hashes)
    monkeypatch.setattr('stash_ai.inference.input_identity.cue_manifest', lambda _: pytest.fail('manifest recreated'))
    second.record_input_workload('1', 'wd', cues, hashes, snapshot=snapshot)
    assert first.workload == second.workload


def test_output_capture_reuses_snapshot_without_changing_record(pipeline, monkeypatch):
    config = ModelSettings(WD, Path('missing'), .4, .65, 4)
    model = IndexedTagger(config, pipeline.scores['wd'], pipeline.vocab.labels)
    analysis = {}
    scores, frames = prediction.predict_scene(app, model, pipeline.stash, {}, pipeline.cues, analysis=analysis)
    snapshot = VerifiedInput.capture(pipeline.cues, {})
    monkeypatch.setattr(results, 'now', lambda: '2026-10-09T00:00:00+00:00')
    arguments = (app, config, pipeline.vocab, pipeline.cues, scores, frames, analysis)
    keywords = dict(inference_signature='sig', frame_input_signature=snapshot.frame_signature)
    expected = results.capture_outputs(*arguments, **keywords)
    monkeypatch.setattr(results, 'cue_manifest', lambda _: pytest.fail('manifest recreated'))
    actual = results.capture_outputs(*arguments, **keywords, verified_input=snapshot)
    assert actual == expected


@pytest.mark.parametrize('prefetch', [True, False])
def test_internal_prefetch_rejects_boolean_values(prefetch):
    with pytest.raises(ValueError, match='prefetch must be an integer'):
        next(preparation.prepared_batches(None, None, None, {}, [], prefetch))
