"""Faults at ownership transfer and blocked sinks remain bounded and recoverable."""
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from threading import Event, Thread
from types import SimpleNamespace
import time

import pytest
from PIL import Image
import numpy as np

from stash_ai.runtime.console import ConsolePresenter, ConsoleDeliveryError, present, _Action
from stash_ai.runtime.report import RunReport
from stash_ai.runtime.startup import submit_owned
from stash_ai.backends import onnx_preprocessing as prep
from stash_ai.inference.batches import BatchPredictions
from stash_ai.inference import preparation


def test_blocked_sink_releases_queue_and_never_replays_cancelled_messages():
    entered, release = Event(), Event()
    messages = []
    def write(message):
        entered.set()
        release.wait(5)
        messages.append(message)
    ui = ConsolePresenter(write, style='off', stream=StringIO(), coordination_timeout=.1).start()
    errors = []
    def first():
        try: ui.message('already started')
        except ConsoleDeliveryError as error: errors.append(error)
    caller = Thread(target=first)
    caller.start()
    try:
        assert entered.wait(2)
        for _ in range(ui.actions.maxsize):
            ui.actions.put_nowait(_Action(lambda: messages.append('cancelled')))
        started = time.monotonic()
        assert ui.call(lambda: messages.append('optional'), default='skipped') == 'skipped'
        ui.close()
        assert time.monotonic()-started < 1
        caller.join(1)
        assert not caller.is_alive() and len(errors) == 1
        assert ui.actions.empty() and ui.degraded
        with pytest.raises(ConsoleDeliveryError): ui.message('unavailable')
        assert ui.counts['console_critical_unconfirmed'] == 2
    finally:
        release.set()
        caller.join(2)
        ui.thread.join(2)
    assert messages == ['already started']


def test_dead_owner_reports_critical_delivery_failure_without_waiting():
    ui = ConsolePresenter(lambda _: None, style='off', coordination_timeout=.1).start()
    ui.stopped.set()
    ui.thread.join(2)
    with pytest.raises(ConsoleDeliveryError): ui.message('required')
    ui.close()


def test_console_launch_interrupted_after_start_is_rolled_back(monkeypatch):
    started = []
    real_start = Thread.start
    def launch(thread):
        real_start(thread)
        if thread.name == 'stash-console':
            started.append(thread)
            raise KeyboardInterrupt('after launch')
    monkeypatch.setattr(Thread, 'start', launch)
    report = RunReport(write=lambda _: None)
    original = report.write
    with pytest.raises(KeyboardInterrupt, match='after launch'):
        with present(report, 'off'): pytest.fail('entered after failed launch')
    assert report.write is original and not started[0].is_alive()


@pytest.mark.parametrize('stage', ['before_submit', 'after_submit', 'register'])
def test_borrowed_executor_never_opens_inputs_before_ownership(stage):
    calls = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        class Interrupting:
            def submit(self, operation):
                if stage == 'before_submit': raise KeyboardInterrupt('launch')
                future = executor.submit(operation)
                if stage == 'after_submit': raise KeyboardInterrupt('launch')
                return future
        def register(future):
            if stage == 'register': raise KeyboardInterrupt('launch')
        with pytest.raises(KeyboardInterrupt, match='launch'):
            submit_owned(Interrupting(), lambda: calls.append('opened'), register)
        assert executor.submit(lambda: 'still borrowed').result(timeout=2) == 'still borrowed'
    assert calls == []


def test_batch_thread_launch_failure_stops_the_already_started_owner(monkeypatch):
    entered, exited = Event(), Event()
    started = []
    def produce(owner):
        entered.set()
        owner.channel.stopped.wait(5)
        exited.set()
    monkeypatch.setattr(BatchPredictions, '_produce', produce)
    real_start = Thread.start
    def launch(thread):
        real_start(thread)
        if thread.name == 'inference-batches':
            started.append(thread)
            assert entered.wait(2)
            raise KeyboardInterrupt('producer launch')
    monkeypatch.setattr(Thread, 'start', launch)
    tagger = SimpleNamespace(settings=SimpleNamespace(source=SimpleNamespace(key='test')))
    with pytest.raises(KeyboardInterrupt, match='producer launch'):
        BatchPredictions(None, tagger, None, {}, [], prefetch=1)
    assert exited.is_set() and not started[0].is_alive()


def test_second_owned_executor_failure_closes_the_first(monkeypatch):
    from stash_ai.inference import sprites
    closed = []
    real_factory = preparation.ThreadPoolExecutor
    def factory(*args, **kwargs):
        if closed: raise KeyboardInterrupt('second executor')
        executor = real_factory(*args, **kwargs)
        shutdown = executor.shutdown
        def close(*args, **kwargs):
            closed.append('shutdown')
            return shutdown(*args, **kwargs)
        executor.shutdown = close
        closed.append('created')
        return executor
    monkeypatch.setattr(preparation, 'ThreadPoolExecutor', factory)
    # A generator supplies close(), as the real sprite reader does.
    monkeypatch.setattr(sprites, 'iter_sprite_frames', lambda *args: (item for item in ()))
    model = SimpleNamespace(batch_size=2)
    stream = preparation.prepared_batches(None, model, None, {}, [], prefetch=1, workers=2)
    with pytest.raises(KeyboardInterrupt, match='second executor'): next(stream)
    assert closed == ['created', 'shutdown']


@pytest.mark.parametrize('prepare', [prep.prepare_wd, prep.prepare_dtq, prep.prepare_oppai])
def test_transform_error_survives_every_owned_image_close_error(monkeypatch, prepare):
    borrowed = Image.new('RGB', (12, 8))
    primary = ValueError('transform failed')
    closed = []
    original = Image.Image.close
    def close(image):
        closed.append(image)
        original(image)
        raise OSError('image close failed')
    def array(*args, **kwargs): raise primary
    with monkeypatch.context() as patch:
        patch.setattr(Image.Image, 'close', close)
        patch.setattr(np, 'asarray', array)
        with pytest.raises(ValueError) as caught: prepare(borrowed, 16)
    assert caught.value is primary and len(closed) >= 2
    assert all(image is not borrowed for image in closed) and borrowed.getpixel((0, 0)) == (0, 0, 0)
    assert len(primary.__notes__) == len(closed)
    borrowed.close()
