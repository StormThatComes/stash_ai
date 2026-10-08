"""Offline concurrency, bounded lookahead and durable-prefix pipeline contracts."""
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event, Lock, enumerate as threads, get_ident, current_thread
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from stash_ai.inference import batches, checkpoints, prediction, preparation, sprites
from stash_ai.models import ModelSettings, JOYTAG, WD
from stash_ai.runtime.profile import ProfileMetrics
from stash_ai.runtime.report import RunReport
from stash_ai.storage.store import StateStore
from tests.helpers.inference import IndexedTagger, sprite_frames
from tests.helpers.runtime import runtime as app

pytestmark = pytest.mark.usefixtures('reset_progress')


def assert_workers_closed():
    assert not any(thread.name.startswith(('prepare-frames', 'transform-frames', 'inference-batches'))
                   for thread in threads())


def test_small_batches_use_workers_across_batch_boundaries(pipeline):
    rendezvous = Barrier(8)
    active_threads = set()
    lock = Lock()
    config = ModelSettings(WD, Path('missing'), .4, .65, 4)
    model = IndexedTagger(config, pipeline.scores['wd'], pipeline.vocab.labels)
    original = model.prepare

    def prepare(image):
        index = image.getpixel((0, 0))
        if index < 8:
            with lock:
                active_threads.add(get_ident())
            rendezvous.wait(timeout=5)
        return original(image)

    model.prepare = prepare
    values = list(preparation.prepared_batches(app, model, pipeline.stash, {}, pipeline.cues,
                                              prefetch=1, workers=8))
    assert [value for prepared, _ in values for value in prepared] == list(range(10))
    assert len(active_threads) == 8
    assert_workers_closed()


@pytest.mark.parametrize('prefetch', [0, 1, 16, 128])
def test_collation_precedes_inference_and_preserves_partial_batch(pipeline, prefetch):
    model = IndexedTagger(ModelSettings(WD, Path('missing'), .4, .65, 4),
                          pipeline.scores['wd'], pipeline.vocab.labels)
    collated, predicted = [], []

    def collate(images):
        collated.append((tuple(images), current_thread().name))
        return tuple(images)

    def predict(inputs, count):
        predicted.append((inputs, count, current_thread().name))
        return model.scores[list(inputs)]

    model.collate_batch, model.predict_collated_batch = collate, predict
    model.predict_batch = lambda _: pytest.fail('Collated input was assembled again during inference')
    report = RunReport(write=lambda _: None)
    report.profile = ProfileMetrics()
    with report.model('wd'):
        report.work('fresh')
        output = list(batches.predicted_batches(app, model, pipeline.stash, {}, pipeline.cues,
                                               prefetch=prefetch, prepare_workers=8, report=report))
    assert [tuple(cue.start for cue in cues) for _, cues in output] == [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9)]
    np.testing.assert_array_equal(np.concatenate([scores for scores, _ in output]), model.scores)
    assert [row[:2] for row in predicted] == [(row[0], len(row[0])) for row in collated]
    if prefetch:
        assert all(name.startswith('prepare-frames') for _, name in collated)
        assert all(name == 'inference-batches' for _, _, name in predicted)
    else:
        assert all(name == current_thread().name for _, name in collated)
        assert all(name == current_thread().name for _, _, name in predicted)
    metrics = {row['name']: row for row in report.profile.export()}
    assert metrics['batch_collate']['units'] == 10
    assert metrics['inference_batch']['calls'] == 3
    assert metrics['inference_batch']['model'] == 'wd'
    assert metrics['inference_batch']['work'] == 'fresh'
    assert metrics['sprite_frame_read']['errors'] == 0
    assert_workers_closed()


def test_next_inference_overlaps_checkpoint_processing_on_owner_thread(pipeline, monkeypatch, tmp_path):
    config = ModelSettings(WD, Path('missing'), .4, .65, 4)
    model = IndexedTagger(config, pipeline.scores['wd'], pipeline.vocab.labels)
    second_started = Event()
    original_predict = model.predict_batch

    def predict(images):
        if images[0] == 4:
            second_started.set()
        return original_predict(images)

    model.predict_batch = predict
    owner = get_ident()
    record = checkpoints.Checkpoint.record
    flush = checkpoints.Checkpoint.flush

    def checked_record(checkpoint, processed, *args, **kwargs):
        assert get_ident() == owner
        if processed == 4:
            assert second_started.wait(5), 'Next inference waited for checkpoint processing'
        return record(checkpoint, processed, *args, **kwargs)

    def checked_flush(checkpoint):
        assert get_ident() == owner
        return flush(checkpoint)

    monkeypatch.setattr(checkpoints.Checkpoint, 'record', checked_record)
    monkeypatch.setattr(checkpoints.Checkpoint, 'flush', checked_flush)
    with StateStore(tmp_path / 'state.sqlite3', 'fixture') as store:
        checkpoint = checkpoints.Checkpoint(store, '1', 'wd', 'signature', 10, 2, every=1)
        report = RunReport(write=lambda _: None)
        report.profile = ProfileMetrics()
        result = prediction.predict_scene(app, model, pipeline.stash, {}, pipeline.cues,
                                          checkpoint=checkpoint, prefetch=2, report=report,
                                          prepare_workers=8)
        assert checkpoint.restore()[0] == 10
        assert checkpoint.saved_processed == 10
        metrics = {row['name']: row for row in report.profile.export()}
        assert metrics['inference_postprocess']['units'] == 10
    reference = IndexedTagger(config, pipeline.scores['wd'], pipeline.vocab.labels)
    assert result == prediction.predict_scene(app, reference, pipeline.stash, {}, pipeline.cues)
    assert_workers_closed()


def test_joytag_collates_on_cpu_and_enters_inference_mode_in_worker(pipeline, monkeypatch):
    torch = pytest.importorskip('torch')
    from stash_ai.backends.loader import JoyTagger
    stack = torch.stack
    stack_threads, model_threads = [], []

    class Network(torch.nn.Module):
        image_size = 2

        def forward(self, inputs):
            assert torch.is_inference_mode_enabled()
            model_threads.append(current_thread().name)
            return {'tags': inputs['image'].mean(dim=(2, 3))[:, :2]}

    def checked_stack(images):
        stack_threads.append(current_thread().name)
        return stack(images)

    monkeypatch.setattr(torch, 'stack', checked_stack)
    model = object.__new__(JoyTagger)
    model.settings = ModelSettings(JOYTAG, Path('missing'), .4, .65, 4)
    model.labels, model.batch_size = pipeline.vocab.labels, 4
    model.device, model.model = torch.device('cpu'), Network().eval()
    model.prepare_image = lambda image, size: torch.full((3, size, size), image.getpixel((0, 0)) / 10)
    try:
        output = list(batches.predicted_batches(app, model, pipeline.stash, {}, pipeline.cues,
                                               prefetch=2, prepare_workers=8))
        actual = np.concatenate([scores for scores, _ in output])
        expected = 1 / (1 + np.exp(-np.arange(10, dtype=np.float32) / 10))
        np.testing.assert_allclose(actual, np.repeat(expected[:, None], 2, axis=1), atol=1e-7)
        assert [len(cues) for _, cues in output] == [4, 4, 2]
        assert len(stack_threads) == 3 and all(name.startswith('prepare-frames') for name in stack_threads)
        assert model_threads == ['inference-batches'] * 3
        assert_workers_closed()
    finally:
        model.close()


@pytest.mark.parametrize('failure', ['read', 'transform', 'collate', 'inference'])
@pytest.mark.parametrize('error_type', [RuntimeError, KeyboardInterrupt])
def test_lookahead_failure_keeps_complete_prefix_and_resumes(pipeline, monkeypatch, tmp_path,
                                                            failure, error_type):
    config = ModelSettings(WD, Path('missing'), .4, .65, 4)
    model = IndexedTagger(config, pipeline.scores['wd'], pipeline.vocab.labels)
    original_prepare = model.prepare
    original_predict = model.predict_batch

    def fail():
        raise error_type('failed later batch')

    if failure == 'read':
        def frames(stash, paths, cues):
            for cue in cues:
                if cue.start == 4:
                    fail()
                yield from sprite_frames([cue])
        monkeypatch.setattr(sprites, 'iter_sprite_frames', frames)
    elif failure == 'transform':
        def prepare(image):
            if image.getpixel((0, 0)) == 4:
                fail()
            return original_prepare(image)
        model.prepare = prepare
    elif failure == 'collate':
        def collate(images):
            if images[0] == 4:
                fail()
            return images
        model.collate_batch = collate
        model.predict_collated_batch = lambda inputs, count: original_predict(inputs)
    else:
        def predict(images):
            if images[0] == 4:
                fail()
            return original_predict(images)
        model.predict_batch = predict

    with StateStore(tmp_path / 'state.sqlite3', 'fixture') as store:
        checkpoint = checkpoints.Checkpoint(store, '1', 'wd', 'signature', 10, 2)
        with pytest.raises(error_type, match='failed later batch'):
            prediction.predict_scene(app, model, pipeline.stash, {}, pipeline.cues,
                                     checkpoint=checkpoint, prefetch=2, prepare_workers=8)
        assert checkpoint.restore()[0] == 4
        assert model.seen == list(range(4))
        assert_workers_closed()
        monkeypatch.setattr(sprites, 'iter_sprite_frames', lambda stash, paths, cues: sprite_frames(cues))
        resumed = IndexedTagger(config, pipeline.scores['wd'], pipeline.vocab.labels)
        result = prediction.predict_scene(app, resumed, pipeline.stash, {}, pipeline.cues,
                                          checkpoint=checkpoint, prefetch=2, prepare_workers=8)
        assert resumed.seen == list(range(4, 10))
    reference = IndexedTagger(config, pipeline.scores['wd'], pipeline.vocab.labels)
    assert result == prediction.predict_scene(app, reference, pipeline.stash, {}, pipeline.cues)
    assert_workers_closed()


@pytest.mark.parametrize('result_prefetch', [1, 2, 16, 32, 128])
def test_full_result_queue_is_bounded_and_close_joins_worker(monkeypatch, result_prefetch):
    full = Event()
    calls, scores = [], []
    model = SimpleNamespace(settings=SimpleNamespace(source=WD), labels=['tag'])

    def inputs(*args, **kwargs):
        for index in range(result_prefetch + 4):
            yield [index], [index]

    def predict(images):
        calls.append(images[0])
        if len(calls) == result_prefetch + 2:
            full.set()
        result = np.array([[.5]], np.float32)
        scores.append(weakref.ref(result))
        return result

    model.predict_batch = predict
    monkeypatch.setattr(batches, 'prepared_batches', inputs)
    with closing(batches.predicted_batches(None, model, None, {}, [], prefetch=1,
                                           result_prefetch=result_prefetch)) as iterator:
        assert next(iterator)[1] == [0]
        assert full.wait(5)
        # One yielded result, N queued CPU results and one awaiting enqueue.
        assert calls == list(range(result_prefetch + 2))
    assert_workers_closed()
    assert all(reference() is None for reference in scores)


@pytest.mark.parametrize('value', [0, -1, 129, True, 1.5])
def test_result_queue_rejects_invalid_or_unbounded_capacity(value):
    with pytest.raises(ValueError, match='result_prefetch'):
        batches.BatchPredictions(None, None, None, {}, [], prefetch=1, result_prefetch=value)


def test_scene_close_cancels_read_but_keeps_borrowed_model_executor(monkeypatch):
    from stash_ai.runtime.cancellation import current
    read_started, cancelled = Event(), Event()
    model = SimpleNamespace(batch_size=1, settings=SimpleNamespace(source=WD), labels=['tag'],
                            prepare=lambda image: 0,
                            predict_batch=lambda images: np.array([[.5]], np.float32))

    def frames(*args):
        yield from sprite_frames([SimpleNamespace(start=0)])
        token = current()
        token.subscribe(cancelled.set)
        try:
            read_started.set()
            assert cancelled.wait(5), 'Closing scene did not cancel its preparation read'
            token.check()
        finally:
            token.unsubscribe(cancelled.set)

    monkeypatch.setattr(sprites, 'iter_sprite_frames', frames)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='inference-model') as executor:
        owner = executor.submit(current_thread).result()
        with closing(batches.predicted_batches(None, model, None, {}, [], prefetch=1,
                                               inference_executor=executor)) as iterator:
            next(iterator)
            assert read_started.wait(5)
        assert cancelled.is_set()
        # Closing a scene must finish its task without shutting down the
        # executor needed by the next scene and backend cleanup.
        assert executor.submit(current_thread).result(timeout=5) is owner
    assert not any(thread.name.startswith('inference-model') for thread in threads())
    assert_workers_closed()


@pytest.mark.parametrize('error_type', [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize('result_prefetch', [2, 32, 128])
def test_consumer_failure_checkpoints_only_processed_results(pipeline, monkeypatch, tmp_path, error_type, result_prefetch):
    from stash_ai.evidence.temporal.statistics import SceneAccumulator
    config = ModelSettings(WD, Path('missing'), .4, .65, 4)
    model = IndexedTagger(config, pipeline.scores['wd'], pipeline.vocab.labels)
    add = SceneAccumulator.add

    def fail_second(accumulator, scores, cues):
        if cues[0].start == 4:
            raise error_type('postprocessing failed')
        return add(accumulator, scores, cues)

    with StateStore(tmp_path / 'state.sqlite3', 'fixture') as store:
        checkpoint = checkpoints.Checkpoint(store, '1', 'wd', 'signature', 10, 2)
        with monkeypatch.context() as patch:
            patch.setattr(SceneAccumulator, 'add', fail_second)
            with pytest.raises(error_type, match='postprocessing failed'):
                prediction.predict_scene(app, model, pipeline.stash, {}, pipeline.cues,
                                         checkpoint=checkpoint, prefetch=2, prepare_workers=8,
                                         result_prefetch=result_prefetch)
        assert checkpoint.restore()[0] == 4
        assert_workers_closed()
        resumed = IndexedTagger(config, pipeline.scores['wd'], pipeline.vocab.labels)
        result = prediction.predict_scene(app, resumed, pipeline.stash, {}, pipeline.cues,
                                          checkpoint=checkpoint, prefetch=2, prepare_workers=8)
        assert resumed.seen == list(range(4, 10))
    reference = IndexedTagger(config, pipeline.scores['wd'], pipeline.vocab.labels)
    assert result == prediction.predict_scene(app, reference, pipeline.stash, {}, pipeline.cues)
    assert_workers_closed()


def test_close_cancels_preparation_read_in_progress(monkeypatch):
    from stash_ai.runtime.cancellation import current
    read_started, cancelled = Event(), Event()
    model = SimpleNamespace(batch_size=1, settings=SimpleNamespace(source=WD), labels=['tag'],
                            prepare=lambda image: 0,
                            predict_batch=lambda images: np.array([[.5]], np.float32))

    def frames(*args):
        yield from sprite_frames([SimpleNamespace(start=0)])
        token = current()
        token.subscribe(cancelled.set)
        try:
            read_started.set()
            assert cancelled.wait(5), 'Closing inference did not cancel its preparation read'
            token.check()
        finally:
            token.unsubscribe(cancelled.set)

    monkeypatch.setattr(sprites, 'iter_sprite_frames', frames)
    with closing(batches.predicted_batches(None, model, None, {}, [], prefetch=1)) as iterator:
        next(iterator)
        assert read_started.wait(5)
    assert cancelled.is_set()
    assert_workers_closed()
