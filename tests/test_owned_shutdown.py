"""Blocked work keeps its payload/backend ownership until actual completion."""
from threading import Event
from time import monotonic
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from stash_ai.runtime import shutdown


def settled():
    deadline = monotonic() + 3
    while shutdown.busy() and monotonic() < deadline:
        Event().wait(.01)
    shutdown.require_idle()


def test_blocked_frame_source_keeps_sheet_and_model_until_worker_exits(tmp_path, monkeypatch):
    from stash_ai.recognition.backend import LazyBackend
    from stash_ai.recognition.media import SpriteInput
    from stash_ai.recognition.prefetch import FramePrefetch
    from tests.helpers.recognition import Backend, MediaClient
    from tests.test_recognition_clustering import setup
    monkeypatch.setattr(shutdown, 'TIMEOUT', .03)
    entered, finish = Event(), Event()
    media = MediaClient()
    source = SpriteInput(media, media.scene, setup(tmp_path))
    def frames(start):
        entered.set()
        finish.wait()
        assert source.cache
        yield from ()
    source.frames = frames
    model = Backend()
    backend = LazyBackend(None, {}, lambda *args: model)
    backend.borrow()
    worker = FramePrefetch(source, 0, 1).launch()
    try:
        assert entered.wait(2)
        with pytest.raises(shutdown.ShutdownIncomplete):
            worker.close()
        source.close()
        backend.close()
        assert source.cache and not model.closed and backend.model is model
        with pytest.raises(shutdown.ShutdownIncomplete, match='blocked'):
            backend.borrow()
    finally:
        finish.set()
        worker.channel.worker.join(2)
        settled()
    assert not source.cache and model.closed and backend.closed


def test_blocked_prediction_retains_borrowed_pixels_and_backend(monkeypatch):
    from stash_ai.inference.batches import BatchPredictions
    from stash_ai.runs.merged_model import MergedModelRun
    monkeypatch.setattr(shutdown, 'TIMEOUT', .03)
    entered, finish = Event(), Event()
    image = Image.new('RGB', (2, 2), 'red')
    class Inputs:
        closed = False
        count = 0
        def __iter__(self): return self
        def __next__(self):
            if self.count:
                raise StopIteration
            self.count += 1
            return [image], ['cue']
        def close(self):
            self.closed = True
            image.close()
    inputs = Inputs()
    def predict(batch):
        entered.set()
        finish.wait()
        assert batch[0].getpixel((0, 0)) == (255, 0, 0)
        return np.zeros((1, 1), dtype='float32')
    tagger = SimpleNamespace(settings=SimpleNamespace(source=SimpleNamespace(key='wd', title='WD')),
                             labels=['tag'], predict_batch=predict, closed=False)
    tagger.close = lambda: setattr(tagger, 'closed', True)
    owner = MergedModelRun.__new__(MergedModelRun)
    owner.tagger = tagger
    stream = BatchPredictions(None, tagger, None, {}, [], prefetch=1, prepared_inputs=inputs)
    try:
        assert entered.wait(2)
        with pytest.raises(shutdown.ShutdownIncomplete):
            stream.close()
        owner._close_tagger()
        assert not inputs.closed and not tagger.closed and owner.tagger is tagger
        assert image.getpixel((0, 0)) == (255, 0, 0)
        with pytest.raises(shutdown.ShutdownIncomplete):
            shutdown.require_idle()
    finally:
        finish.set()
        stream.worker.join(2)
        settled()
    assert inputs.closed and tagger.closed and owner.tagger is None


def test_blocked_transform_retains_iterator_and_backend(monkeypatch):
    from stash_ai.inference import preparation
    monkeypatch.setattr(shutdown, 'TIMEOUT', .03)
    entered, finish = Event(), Event()
    images = [Image.new('RGB', (2, 2), 'red') for _ in range(2)]
    for index, image in enumerate(images):
        image.info['index'] = index
    class Frames:
        index = 0
        closed = False
        def __iter__(self): return self
        def __next__(self):
            if self.index == len(images): raise StopIteration
            index = self.index
            self.index += 1
            return str(index), images[index]
        def close(self):
            self.closed = True
            for image in images: image.close()
    frames = Frames()
    pools = []
    factory = preparation.ThreadPoolExecutor
    def pool(*args, **kwargs):
        executor = factory(*args, **kwargs)
        pools.append(executor)
        return executor
    def prepare(image):
        if image.info['index'] == 1:
            entered.set()
            finish.wait()
            assert image.getpixel((0, 0)) == (255, 0, 0)
        return image.info['index']
    tagger = SimpleNamespace(batch_size=1, prepare=prepare,
                             settings=SimpleNamespace(source=SimpleNamespace(key='wd')), closed=False)
    monkeypatch.setattr(preparation, 'ThreadPoolExecutor', pool)
    monkeypatch.setattr(preparation.sprites, 'iter_sprite_frames', lambda *args: frames)
    batches = preparation.prepared_batches(None, tagger, None, {}, [], prefetch=1, workers=2)
    try:
        assert next(batches)[0] == [0]
        assert entered.wait(2)
        with pytest.raises(shutdown.ShutdownIncomplete): batches.close()
        shutdown.release(tagger, lambda: setattr(tagger, 'closed', True))
        assert not frames.closed and not tagger.closed
        with pytest.raises(shutdown.ShutdownIncomplete): shutdown.require_idle()
    finally:
        finish.set()
        for executor in pools:
            for thread in executor._threads: thread.join(2)
        settled()
        batches.close()
    assert frames.closed and tagger.closed


def test_repeated_shutdown_does_not_spawn_more_blocked_helpers(monkeypatch):
    from threading import enumerate as threads
    monkeypatch.setattr(shutdown, 'TIMEOUT', .03)
    finish = Event()
    owner = object()
    workers = None
    try:
        for _ in range(3):
            with pytest.raises(shutdown.ShutdownIncomplete):
                shutdown.call(finish.wait, owner)
            current = {thread for thread in threads() if thread.name == 'owned-shutdown'}
            if workers is None: workers = current
            assert current == workers and len(current) == 1
    finally:
        finish.set()
        settled()
