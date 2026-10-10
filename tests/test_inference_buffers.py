"""Snapshots observe real ready payloads without consuming or changing work."""
from concurrent.futures import Future
from contextlib import closing
from queue import Queue
from threading import Event
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from stash_ai.inference import batches, preparation
from stash_ai.inference.batch_stream import PreparedBatchStream
from stash_ai.inference.buffer_usage import BufferUsage, PreparedBatchUsage, queued_batches
from stash_ai.models import WD
from stash_ai.runs.inference_progress import InferenceProgress, buffer_lines
from stash_ai.runtime.cache import RamCache
from tests.helpers.inference import IndexedTagger
from tests.helpers.runtime import runtime as app
from tests.test_inference_progress import bars
from tests.test_cross_scene_batches import instrument
from tests.test_merged_model import model_run, run_model, assert_inference_workers_closed
from stash_ai.runs.merged_model import MergedModelRun


pytestmark = pytest.mark.usefixtures('reset_progress')


def test_input_snapshot_tracks_in_flight_and_ignores_failed_cancelled_and_empty_futures():
    usage = PreparedBatchUsage(8)
    ready, running, failed, cancelled, empty = [Future() for _ in range(5)]
    ready.set_result(([1, 2], ['a', 'b']))
    running.set_running_or_notify_cancel()
    failed.set_exception(ValueError('preparation failed'))
    cancelled.cancel()
    empty.set_result(([], []))
    for future in (ready, running, failed, cancelled, empty):
        usage.submitted(future)
    assert usage.snapshot() == BufferUsage(1, 1, 9)  # Does not wait for the running task.
    assert ready.result() == ([1, 2], ['a', 'b'])
    usage.taken(ready)
    assert usage.snapshot() == BufferUsage(0, 2, 9)
    held = weakref.ref(ready)
    usage.submitted(ready)
    del ready
    usage.clear()
    assert held() is None


def test_input_snapshot_tracks_real_preparation_and_releases_futures(pipeline):
    from pathlib import Path
    from stash_ai.models import ModelSettings
    config = ModelSettings(WD, Path('missing'), .4, .65, 2)
    model = IndexedTagger(config, pipeline.scores['wd'], pipeline.vocab.labels)
    usage = PreparedBatchUsage(4)
    ready = Event()
    prepare = model.prepare

    def transform(image):
        if image.getpixel((0, 0)) == 9:
            ready.set()
        return prepare(image)

    model.prepare = transform
    with closing(preparation.prepared_batches(app, model, pipeline.stash, {}, pipeline.cues,
                                              prefetch=4, buffer_usage=usage)) as stream:
        assert next(stream)[1] == pipeline.cues[:2]
        assert ready.wait(5)
        # The final transform signalled before its future became ready.
        with usage._lock:
            pending = tuple(usage._pending)
        for future in pending:
            future.result(timeout=5)
        assert usage.snapshot() == BufferUsage(4, 1, 5)
    assert usage.snapshot() == BufferUsage(0, 0, 5)
    assert_inference_workers_closed()


def test_queue_snapshot_excludes_control_markers_and_preserves_payloads():
    queue = Queue(maxsize=4)
    payload = (np.array([[.5]], np.float32), ['cue'])
    queue.put(payload)
    queue.put(object())
    queue.put(batches._Failure(ValueError('failed')))
    assert queued_batches(queue) == BufferUsage(1, 0, 4)
    assert queue.qsize() == 3 and queue.get_nowait() is payload
    stream = PreparedBatchStream(2, None, 'wd')
    stream.queue.put(payload)
    stream.queue.put(stream.end)
    assert stream.buffer_usage() == BufferUsage(1, 0, 4)
    stream.close()
    assert stream.buffer_usage() == BufferUsage(0, 0, 4)


@pytest.mark.parametrize('limit', [1, 32, 128])
def test_result_snapshot_counts_full_queue_and_close_releases_scores(monkeypatch, limit):
    full = Event()
    calls, references = [], []
    model = SimpleNamespace(settings=SimpleNamespace(source=WD), labels=['tag'])

    def inputs(*args, **kwargs):
        for index in range(limit + 4):
            yield [index], [index]

    def predict(images):
        calls.append(images[0])
        if len(calls) == limit + 2:
            full.set()
        result = np.array([[.5]], np.float32)
        references.append(weakref.ref(result))
        return result

    model.predict_batch = predict
    monkeypatch.setattr(batches, 'prepared_batches', inputs)
    with closing(batches.BatchPredictions(None, model, None, {}, [], prefetch=4,
                                         result_prefetch=limit)) as stream:
        assert next(stream)[1] == [0]
        assert full.wait(5)
        assert stream.buffer_usage()['result'] == BufferUsage(limit, 2, limit + 2)
        assert stream.buffer_usage()['result'] == BufferUsage(limit, 2, limit + 2)
        assert calls == list(range(limit + 2))
    assert stream.buffer_usage()['result'] == BufferUsage(0, 0, limit + 2)
    assert all(reference() is None for reference in references)
    assert_inference_workers_closed()


def test_next_scene_snapshot_keeps_current_and_following_queues_separate(model_run, bars, monkeypatch):
    context = model_run
    context.args.prefetch_batches = 16
    next_finished = Event()

    def predict(scene, images):
        if scene == '2' and images[0][1] == 8:
            next_finished.set()

    instrument(context, monkeypatch, predict=predict)
    samples = []
    sample = MergedModelRun._buffer_snapshot
    from stash_ai.inference import checkpoints
    finish = checkpoints.finish

    def observe(run, job):
        snapshot = sample(run, job)
        samples.append((run, job, snapshot))
        return snapshot

    def commit(store, scene, *args):
        if scene == '1':
            assert next_finished.wait(5)
            run, job, _ = samples[-1]
            run._staged.preparation.job.batches.worker.result(timeout=5)
            snapshot = sample(run, job)
            assert snapshot['current']['result'] == BufferUsage(0, 0, 34)
            assert snapshot['next']['scene'] == '2'
            assert snapshot['next']['result'] == BufferUsage(5, 0, 34)
            assert 'next[2]' in buffer_lines(snapshot)[0]
        return finish(store, scene, *args)

    monkeypatch.setattr(MergedModelRun, '_buffer_snapshot', observe)
    monkeypatch.setattr(checkpoints, 'finish', commit)
    run_model(context)
    assert context.stats['processed'] == 2 and not context.stats['errors']
    assert all(bar.closed for bar in bars)
    assert_inference_workers_closed()


def test_ram_snapshot_does_not_touch_lru_statistics_or_pressure(monkeypatch):
    cache = RamCache(10000, rss_budget=40000)
    cache.put('first', b'data')
    state = list(cache.entries), dict(cache.stats), cache._next_pressure_check
    monkeypatch.setattr(cache, 'pressure', lambda *args, **kwargs: pytest.fail('UI evicted cache'))
    assert cache.usage() == (cache.size, 10000)
    assert (list(cache.entries), dict(cache.stats), cache._next_pressure_check) == state
    assert RamCache(0).usage() == (0, 0)


def test_buffer_format_distinguishes_disabled_queues_and_checkpoint_interval():
    rows = buffer_lines(dict(current={'input': None, 'result': None}, next=None,
                             cache=(2**30, 24 * 2**30), checkpoint=(3, 32)))
    assert rows == ('Buffers: in=off out=off | next=off',
                    'Cache: 1.0GiB/24.0GiB | checkpoint interval=3/32 batch')


def test_status_samples_only_when_the_frame_bar_redraws():
    calls = []
    frame = SimpleNamespace(update=lambda count: False)
    row = SimpleNamespace(set_description_str=lambda text: calls.append(text))
    display = InferenceProgress(None, total=10, initial=0, desc='', sample=lambda: dict(
        current={'input': BufferUsage(1, 2, 4), 'result': BufferUsage(2, 1, 33)}, next=None, cache=(0, 0), checkpoint=(1, 8)))
    display.frame, display.rows = frame, [row, row]
    display.update(4)
    assert calls == []
    frame.update = lambda count: True
    display.update(4)
    assert len(calls) == 2
