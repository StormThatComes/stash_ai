"""Operational failures survive optional reporting and complete resource cleanup."""
from concurrent.futures import Future
from pathlib import Path
from threading import Thread
from types import SimpleNamespace

import pytest

from stash_ai.inference.batches import BatchPredictions
from stash_ai.inference.managed_stream import ManagedStream
from stash_ai.publication.dry_run import DryRunPlan
from stash_ai.runtime.report import RunReport


def fail(*args, **kwargs):
    raise OSError('secondary failure')


def test_terminated_thread_drains_payload_then_reports_missing_completion():
    stream = ManagedStream(2)
    stream.worker = Thread(target=lambda: None)
    stream.worker.start()
    stream.worker.join(1)
    stream.send('queued')
    assert stream.receive() == 'queued'
    with pytest.raises(RuntimeError, match='without completing'):
        stream.receive()
    stream.close()


@pytest.mark.parametrize('cancelled', [False, True])
def test_future_completion_without_sentinel_remains_supported(cancelled):
    stream = ManagedStream(1)
    stream.worker = Future()
    if cancelled:
        stream.worker.cancel()
    else:
        stream.worker.set_result(None)
    assert stream.receive() is stream.end
    stream.close()


def test_model_error_survives_input_close_failure():
    primary = ValueError('model failed')

    class Inputs:
        def __iter__(self): return self
        def __next__(self): return [object()], ['cue']
        def close(self): fail()

    def predict(_): raise primary
    tagger = SimpleNamespace(settings=SimpleNamespace(source=SimpleNamespace(key='wd', title='WD')),
                             labels=['tag'], predict_batch=predict)
    stream = BatchPredictions(None, tagger, None, {}, [], prefetch=1,
                              prepared_inputs=Inputs(), result_prefetch=2)
    try:
        with pytest.raises(ValueError) as caught:
            next(stream)
        assert caught.value is primary
        assert any('secondary failure' in note for note in primary.__notes__)
    finally:
        with pytest.raises(OSError):
            stream.close()
        assert not stream.worker.is_alive()


@pytest.mark.parametrize('prefetch,workers', [(0, 1), (2, 1), (2, 2)])
def test_cancellation_failure_still_closes_media_and_workers(monkeypatch, prefetch, workers):
    from PIL import Image
    from stash_ai.inference import preparation
    from stash_ai.runtime.cancellation import Cancellation

    image, closed, pools = Image.new('RGB', (1, 1)), [], []

    class Frames:
        remaining = 1
        def __iter__(self): return self
        def __next__(self):
            if not self.remaining: raise StopIteration
            self.remaining -= 1
            return 'cue', image
        def close(self): closed.append(True)

    factory = preparation.ThreadPoolExecutor
    def pool(*args, **kwargs):
        executor = factory(*args, **kwargs)
        pools.append(executor)
        return executor
    monkeypatch.setattr(preparation, 'ThreadPoolExecutor', pool)
    monkeypatch.setattr(preparation.sprites, 'iter_sprite_frames', lambda *args: Frames())
    token = Cancellation()
    token.subscribe(fail)
    tagger = SimpleNamespace(batch_size=1, prepare=lambda image: object(),
                             settings=SimpleNamespace(source=SimpleNamespace(key='wd')))
    stream = preparation.prepared_batches(None, tagger, None, {}, [], prefetch=prefetch,
                                           workers=workers, cancellation=token)
    try:
        assert len(next(stream)[0]) == 1
        with pytest.raises(OSError, match='secondary failure'):
            next(stream)
        assert closed == [True]
        assert all(not thread.is_alive() for executor in pools for thread in executor._threads)
    finally:
        stream.close()
        image.close()


@pytest.mark.parametrize('primary', [None, ValueError('operation failed')])
@pytest.mark.parametrize('kind', ['stage', 'request'])
def test_optional_timing_callbacks_do_not_change_operation_outcome(primary, kind):
    messages = []
    report = RunReport(write=messages.append)
    report.diagnostics = SimpleNamespace(record_stage=fail, record_api=fail)
    report.profile = SimpleNamespace(record=fail)
    report.history = SimpleNamespace(progress=fail)
    context = report.measure('test') if kind == 'stage' else report.request('read', True)
    if primary is None:
        with context: pass
    else:
        with pytest.raises(ValueError) as caught:
            with context: raise primary
        assert caught.value is primary
        assert any('Optional diagnostics_stage failed' in note for note in primary.__notes__)
    assert report.counts['diagnostics_stage_failures'] == 1
    assert report.counts['profile_stage_failures'] == 1
    assert report.counts['history_progress_failures'] == 1
    assert messages and not report.active_requests


def test_optional_warning_failure_does_not_recurse_or_mask_primary():
    report = RunReport(write=fail)
    report.diagnostics = SimpleNamespace(record_stage=fail)
    with pytest.raises(ValueError, match='primary'):
        with report.measure('test'): raise ValueError('primary')
    with report.measure('test'): pass
    assert report.counts['diagnostics_stage_failures'] == 2


class PlanFile:
    def __init__(self):
        self.closed, self.fail_write, self.fail_flush, self.fail_close = False, False, False, False
    def write(self, text):
        if self.fail_write: fail()
    def flush(self):
        if self.fail_flush: fail()
    def close(self):
        self.closed = True
        if self.fail_close: fail()


@pytest.mark.parametrize('boundary', ['write', 'flush'])
def test_plan_header_failure_closes_file(monkeypatch, tmp_path, boundary):
    file = PlanFile()
    setattr(file, 'fail_' + boundary, True)
    file.fail_close = True
    monkeypatch.setattr(Path, 'open', lambda *args, **kwargs: file)
    plan = DryRunPlan(tmp_path/'plan.jsonl')
    with pytest.raises(OSError) as caught:
        plan.__enter__()
    assert file.closed and plan.stream is None
    assert caught.value.__notes__


def test_plan_terminal_failure_closes_file_and_preserves_primary(monkeypatch, tmp_path):
    file = PlanFile()
    monkeypatch.setattr(Path, 'open', lambda *args, **kwargs: file)
    plan = DryRunPlan(tmp_path/'plan.jsonl')
    with pytest.raises(ValueError) as caught:
        with plan:
            file.fail_write = file.fail_close = True
            raise ValueError('primary')
    assert file.closed and plan.stream is None
    assert len(caught.value.__notes__) == 2


def test_plan_close_only_failure_is_visible(monkeypatch, tmp_path):
    file = PlanFile()
    monkeypatch.setattr(Path, 'open', lambda *args, **kwargs: file)
    with pytest.raises(OSError):
        with DryRunPlan(tmp_path/'plan.jsonl') as plan:
            file.fail_close = True
    assert file.closed and plan.stream is None
