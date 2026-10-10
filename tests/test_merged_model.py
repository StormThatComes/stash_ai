"""Merged model lifecycle against real input signatures and SQLite checkpoints."""
from collections import Counter
from concurrent.futures import Future
from threading import Event, current_thread, enumerate as threads
from types import SimpleNamespace

import pytest

from tests.helpers.runtime import runtime as app
import stash_ai.storage.store as merge_store
from stash_ai.inference import prediction
from stash_ai.inference.identity import model_identity
from stash_ai.inference.inputs import RunInputs, inspect_input
from stash_ai.models import ModelSettings, WD
from stash_ai.runs.merged_model import MergedModelRun
from tests.helpers.cli import runtime_args


pytestmark = pytest.mark.usefixtures('reset_progress')


@pytest.fixture
def model_run(pipeline, tmp_path):
    config = ModelSettings(WD, tmp_path / 'model', batch_size=2, scene_threshold=.4, marker_threshold=.65)
    args = runtime_args(resume=False, restart=False)
    scenes = {identifier: dict(paths=dict(pipeline.stash.scene['paths']),
                               publication_reason='missing')
              for identifier in ('1', '2')}
    with merge_store.StateStore(pipeline.path, 'fixture') as state:
        stats = Counter()
        inputs = RunInputs(pipeline.stash, state.cache, args.report)
        yield SimpleNamespace(pipeline=pipeline, config=config, args=args, scenes=scenes,
                              state=state, stats=stats, inputs=inputs)


def run_model(context, *, input_plan=None):
    MergedModelRun(
        app, context.args, context.state, context.config, context.pipeline.vocab,
        stats=context.stats, report=context.args.report,
    ).run(context.scenes, ['scene', 'markers'], context.inputs, input_plan=input_plan)


def assert_inference_workers_closed():
    assert not any(thread.name.startswith(('inference-model', 'inference-batches',
                                           'prepare-frames', 'transform-frames', 'scene-inputs', 'scene-batches'))
                   for thread in threads())


def test_next_input_read_overlaps_inference_and_keeps_sqlite_on_owner(model_run, monkeypatch):
    from stash_ai.inference.checkpoints import Checkpoint
    from stash_ai.runtime.profile import ProfileMetrics
    context = model_run
    context.args.prefetch_batches = 16
    context.args.report.profile = ProfileMetrics()
    for scene, current in context.scenes.items():
        current['paths']['vtt'] = '/vtt/' + scene
    started, release, finished = Event(), Event(), Event()
    owner = current_thread()
    inspect, load, record = context.inputs.inspect, app.services.load_tagger, Checkpoint.record

    def inspect_next(*args):
        if args[1]['vtt'] == '/vtt/2':
            assert current_thread() is not owner
            started.set()
            assert release.wait(5), 'Next scene was read synchronously before inference'
            result = inspect(*args)
            finished.set()
            return result
        return inspect(*args)

    def load_model(*args):
        model = load(*args)
        predict = model.predict_batch

        def predict_batch(images):
            assert started.wait(5), 'Next input read waited for the current scene to finish'
            release.set()
            assert finished.wait(5)
            return predict(images)

        model.predict_batch = predict_batch
        return model

    def record_checkpoint(*args, **kwargs):
        assert current_thread() is owner
        return record(*args, **kwargs)

    monkeypatch.setattr(context.inputs, 'inspect', inspect_next)
    monkeypatch.setattr(app.services, 'load_tagger', load_model)
    monkeypatch.setattr(Checkpoint, 'record', record_checkpoint)
    run_model(context)
    assert context.stats['processed'] == 2 and not context.stats['errors']
    metrics = {row['name']: row for row in context.args.report.profile.export()}
    assert metrics['scene_input_prefetch']['calls'] == metrics['scene_input_wait']['calls'] == 1
    assert metrics['inference_scene_finalize']['calls'] == metrics['inference_output_capture']['calls'] == 2
    assert_inference_workers_closed()


@pytest.mark.parametrize('error', [OSError, KeyboardInterrupt])
def test_next_input_failure_waits_for_its_scene_and_keeps_previous_result(model_run, monkeypatch, error):
    from stash_ai.inference import checkpoints
    from stash_ai.storage.input_checks import pending
    context = model_run
    context.args.prefetch_batches = 2
    context.scenes['2']['paths']['vtt'] = '/vtt/2'
    failed = Event()
    inspect, load, finish = context.inputs.inspect, app.services.load_tagger, checkpoints.finish
    owner = current_thread()

    def inspect_next(*args):
        if args[1]['vtt'] == '/vtt/2':
            failed.set()
            raise error('next input failed')
        return inspect(*args)

    def load_model(*args):
        model = load(*args)
        predict = model.predict_batch

        def predict_batch(images):
            assert failed.wait(5)
            return predict(images)

        model.predict_batch = predict_batch
        return model

    def finish_current(*args):
        assert current_thread() is owner
        assert failed.is_set() and not pending(context.state, '2')
        return finish(*args)

    monkeypatch.setattr(context.inputs, 'inspect', inspect_next)
    monkeypatch.setattr(app.services, 'load_tagger', load_model)
    monkeypatch.setattr(checkpoints, 'finish', finish_current)
    if error is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt, match='next input failed'):
            run_model(context)
    else:
        run_model(context)
        assert context.stats['errors'] == 1
    assert context.stats['processed'] == 1
    assert context.state.records('1', 'scene')['wd']['meta']['origin'] == 'inference'
    assert not context.state.records('2', 'scene')
    assert pending(context.state, '2') == {'wd': 'read_error'}
    assert_inference_workers_closed()


def test_interrupt_cancels_one_future_scene_before_model_close(model_run, monkeypatch):
    from stash_ai.runtime.cancellation import current
    context = model_run
    context.args.prefetch_batches = 16
    for index in range(1, 5):
        context.scenes[str(index)] = dict(paths=dict(vtt='/vtt/' + str(index), sprite='/sprite'))
    started, cancelled, stopped = Event(), Event(), Event()
    read_scenes = []
    inspect, load = context.inputs.inspect, app.services.load_tagger

    def inspect_next(*args):
        scene = args[1]['vtt']
        read_scenes.append(scene)
        if scene == '/vtt/2':
            token = current()
            token.subscribe(cancelled.set)
            try:
                started.set()
                assert cancelled.wait(5), 'Interrupted run did not cancel next input read'
                token.check()
            finally:
                token.unsubscribe(cancelled.set)
                stopped.set()
        return inspect(*args)

    def load_model(*args):
        model = load(*args)
        close = model.close

        def interrupt(images):
            assert started.wait(5)
            raise KeyboardInterrupt('current inference stopped')

        def close_model():
            assert stopped.is_set()
            close()

        model.predict_batch, model.close = interrupt, close_model
        return model

    monkeypatch.setattr(context.inputs, 'inspect', inspect_next)
    monkeypatch.setattr(app.services, 'load_tagger', load_model)
    with pytest.raises(KeyboardInterrupt, match='current inference stopped'):
        run_model(context)
    assert read_scenes == ['/vtt/1', '/vtt/2']
    assert cancelled.is_set() and context.pipeline.loaded[0].closed
    assert not context.state.records('2', 'scene')
    assert_inference_workers_closed()


def test_scene_lookahead_skips_completed_inputs(model_run, monkeypatch):
    from stash_ai.runs.cached_inputs import CachedInputPlan
    from stash_ai.storage.progress import request_refresh
    context = model_run
    context.args.prefetch_batches = 2
    for index in range(1, 4):
        context.scenes[str(index)] = dict(paths=dict(vtt='/vtt/' + str(index), sprite='/sprite'))
    run_model(context)
    context.args.resume = True
    context.stats.clear()
    request_refresh(context.state, ['1', '3'], ['wd'], ['scene', 'markers'])
    run = MergedModelRun(app, context.args, context.state, context.config, context.pipeline.vocab,
                         stats=context.stats, report=context.args.report)
    plan = CachedInputPlan(context.state, context.scenes, ['scene', 'markers'], [run], context.stats)
    inspected = []
    inspect = context.inputs.inspect

    def checked_inspect(*args):
        inspected.append(args[1]['vtt'])
        return inspect(*args)

    monkeypatch.setattr(context.inputs, 'inspect', checked_inspect)
    run.run(context.scenes, ['scene', 'markers'], context.inputs, input_plan=plan)
    assert inspected == ['/vtt/1', '/vtt/3']
    assert context.stats['processed'] == 2 and context.stats['reused'] == 1
    assert_inference_workers_closed()


def test_input_reader_close_failure_still_closes_backend(model_run, monkeypatch):
    context = model_run
    context.args.prefetch_batches = 2
    reader_threads = []

    def create_reader():
        owner = current_thread()
        reader_threads.append(owner)

        def close():
            assert current_thread() is owner
            raise RuntimeError('input reader close failed')

        return SimpleNamespace(get_media=context.pipeline.stash.get_media, close=close)

    monkeypatch.setattr(context.pipeline.stash, 'new_media_reader', create_reader, raising=False)
    with pytest.raises(RuntimeError, match='input reader close failed'):
        run_model(context)
    assert context.stats['processed'] == 2
    assert context.pipeline.loaded[0].closed
    assert_inference_workers_closed()


@pytest.mark.parametrize('prefetch', [0, 2])
def test_model_load_all_scenes_and_close_keep_one_backend_thread(model_run, monkeypatch, prefetch):
    from stash_ai.inference.checkpoints import Checkpoint
    context = model_run
    context.args.prefetch_batches = prefetch
    context.args.prepare_workers = 8
    context.scenes = {str(index): dict(paths=dict(context.pipeline.stash.scene['paths']))
                      for index in range(1, 21)}
    owner = current_thread()
    loaded, predicted, closed = [], [], []
    load = app.services.load_tagger

    def tracked_load(*args):
        loaded.append(current_thread())
        model = load(*args)
        predict, close = model.predict_batch, model.close

        def tracked_predict(images):
            predicted.append(current_thread())
            assert current_thread() is loaded[0]
            return predict(images)

        def tracked_close():
            closed.append(current_thread())
            assert current_thread() is loaded[0]
            close()

        model.predict_batch, model.close = tracked_predict, tracked_close
        return model

    record = Checkpoint.record

    def checked_record(*args, **kwargs):
        assert current_thread() is owner
        return record(*args, **kwargs)

    monkeypatch.setattr(app.services, 'load_tagger', tracked_load)
    monkeypatch.setattr(Checkpoint, 'record', checked_record)
    run_model(context)
    assert context.stats['processed'] == 20 and not context.stats['errors']
    assert len(loaded) == len(closed) == 1 and len(predicted) == 100
    assert all(thread is loaded[0] for thread in predicted + closed)
    assert (loaded[0] is owner) == (prefetch == 0)
    assert context.pipeline.loaded[0].closed
    assert_inference_workers_closed()


@pytest.mark.parametrize('error', [RuntimeError, KeyboardInterrupt])
def test_backend_failure_cleans_scene_and_preserves_model_worker(model_run, monkeypatch, error):
    context = model_run
    context.args.prefetch_batches = 2
    owner_threads, calls = [], []
    load = app.services.load_tagger

    def failing_load(*args):
        owner_threads.append(current_thread())
        model = load(*args)
        predict, close = model.predict_batch, model.close

        def failing_predict(images):
            assert current_thread() is owner_threads[0]
            calls.append(tuple(images))
            if len(calls) == 2:
                raise error('backend failed')
            return predict(images)

        def checked_close():
            assert current_thread() is owner_threads[0]
            close()

        model.predict_batch, model.close = failing_predict, checked_close
        return model

    monkeypatch.setattr(app.services, 'load_tagger', failing_load)
    if error is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt, match='backend failed'):
            run_model(context)
        assert calls == [(0, 1), (2, 3)]
        assert not context.state.records('2', 'scene')
    else:
        run_model(context)
        assert context.stats['processed'] == context.stats['errors'] == 1
        assert context.state.records('2', 'scene')['wd']['meta']['origin'] == 'inference'
    assert context.pipeline.loaded[0].closed
    assert_inference_workers_closed()


def test_interrupt_waiting_for_load_still_closes_initialized_model(model_run, monkeypatch):
    context = model_run
    context.args.prefetch_batches = 2
    started, released = Event(), Event()
    owner, interrupted = current_thread(), False
    load, result = app.services.load_tagger, Future.result
    load_threads, close_threads = [], []

    def slow_load(*args):
        load_threads.append(current_thread())
        started.set()
        assert released.wait(5)
        model = load(*args)
        close = model.close

        def tracked_close():
            close_threads.append(current_thread())
            close()

        model.close = tracked_close
        return model

    def interrupted_result(future, *args, **kwargs):
        nonlocal interrupted
        if current_thread() is owner and not interrupted:
            interrupted = True
            assert started.wait(5)
            released.set()
            raise KeyboardInterrupt('interrupted load wait')
        return result(future, *args, **kwargs)

    monkeypatch.setattr(app.services, 'load_tagger', slow_load)
    monkeypatch.setattr(Future, 'result', interrupted_result)
    with pytest.raises(KeyboardInterrupt, match='interrupted load wait'):
        run_model(context)
    assert context.pipeline.loaded[0].closed
    assert load_threads == close_threads and load_threads[0] is not owner
    assert not context.pipeline.loaded[0].seen
    assert_inference_workers_closed()


def test_close_failure_still_shuts_down_model_worker(model_run, monkeypatch):
    context = model_run
    context.args.prefetch_batches = 2
    load = app.services.load_tagger

    def failing_load(*args):
        model = load(*args)

        def close():
            raise RuntimeError('close failed')

        model.close = close
        return model

    monkeypatch.setattr(app.services, 'load_tagger', failing_load)
    with pytest.raises(RuntimeError, match='close failed'):
        run_model(context)
    assert context.stats['processed'] == 2
    assert context.state.connection.execute('SELECT 1').fetchone() == (1,)
    assert_inference_workers_closed()


@pytest.mark.parametrize('error', [RuntimeError, KeyboardInterrupt])
def test_scene_failure_continues_but_interruption_stops_and_closes_tagger(model_run, monkeypatch, error):
    context = model_run
    original = prediction.predict_scene
    calls = []

    def predict(*args, **kwargs):
        calls.append(kwargs['checkpoint'].scene)
        if len(calls) == 1:
            raise error('first scene failed')
        return original(*args, **kwargs)

    monkeypatch.setattr(prediction, 'predict_scene', predict)
    if error is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt, match='first scene failed'):
            run_model(context)
        assert calls == ['1']
        assert not context.stats['errors']
        assert context.state.records('2', 'scene') == {}
    else:
        run_model(context)
        assert calls == ['1', '2']
        assert context.stats['errors'] == context.stats['processed'] == 1
        assert context.state.records('2', 'scene')['wd']['meta']['origin'] == 'inference'
        assert ('inference', 'wd', 'RuntimeError') in context.args.report.groups
    assert context.state.records('1', 'scene') == {}
    from stash_ai.storage.progress import pending_refreshes
    assert pending_refreshes(context.state, ['scene', 'markers'])['1'] == [('wd', 'markers'), ('wd', 'scene')]
    assert len(context.pipeline.loaded) == 1
    assert context.pipeline.loaded[0].closed
    # The borrowed database remains usable after the model's finally block.
    assert context.state.connection.execute('SELECT 1').fetchone() == (1,)


@pytest.mark.parametrize('prefetch', [0, 2])
def test_loading_failure_stops_remaining_scenes_without_retrying(model_run, monkeypatch, prefetch):
    context = model_run
    context.args.prefetch_batches = prefetch
    inspected, loaded = [], []
    original = context.inputs.inspect

    def inspect(*args):
        inspected.append(args[1])
        return original(*args)

    def load(*args):
        loaded.append(args[0].source.key)
        raise OSError('missing weights')

    monkeypatch.setattr(context.inputs, 'inspect', inspect)
    monkeypatch.setattr(app.services, 'load_tagger', load)
    run_model(context)

    assert loaded == ['wd'] and len(inspected) == 1
    assert context.stats['errors'] == 1 and not context.stats['processed']
    assert set(context.args.report.groups) == {('load model', 'wd', 'OSError')}
    assert not context.state.records('1', 'scene') and not context.state.records('2', 'scene')
    assert_inference_workers_closed()


@pytest.mark.parametrize('reason', ['missing-input', 'completed'])
@pytest.mark.parametrize('prefetch', [0, 2])
def test_skipped_and_completed_scenes_do_not_load_a_model(model_run, monkeypatch, reason, prefetch):
    context = model_run
    context.args.prefetch_batches = prefetch
    if reason == 'missing-input':
        for current in context.scenes.values():
            current['paths'] = {}
    else:
        run_model(context)
        assert context.pipeline.loaded[0].closed
        context.args.resume = True
    context.stats.clear()
    monkeypatch.setattr(app.services, 'load_tagger', lambda *args: pytest.fail('unnecessary model load'))

    run_model(context)

    assert context.stats['reused' if reason == 'completed' else 'skipped'] == 2
    assert not context.stats['processed'] and not context.stats['errors']
    assert_inference_workers_closed()


def test_downloaded_weights_refresh_signatures_before_saving_and_enable_resume(model_run, monkeypatch):
    context = model_run
    before = model_identity(app, context.config, context.pipeline.vocab)
    original = app.services.load_tagger

    def load(*args):
        context.config.model_dir.mkdir()
        (context.config.model_dir / 'model.onnx').write_bytes(b'downloaded weights')
        return original(*args)

    monkeypatch.setattr(app.services, 'load_tagger', load)
    run_model(context)
    after = model_identity(app, context.config, context.pipeline.vocab)
    assert before != after
    expected = inspect_input(app, context.pipeline.stash, context.scenes['1']['paths'], after).signature
    for scene_id in context.scenes:
        for kind in ('scene', 'markers'):
            assert context.state.records(scene_id, kind)['wd']['meta']['inference_signature'] == expected

    context.args.resume = True
    context.stats.clear()
    monkeypatch.setattr(app.services, 'load_tagger', lambda *args: pytest.fail('downloaded weights invalidated resume'))
    run_model(context)
    assert context.stats['reused'] == 2 and not context.stats['processed']


def test_content_replacement_invalidates_resume_and_hashes_once_per_run(model_run, monkeypatch):
    import hashlib
    import os
    context = model_run
    context.config.model_dir.mkdir()
    weight = context.config.model_dir/'model.onnx'
    weight.write_bytes(b'AAAA')
    calls, original = [], hashlib.file_digest
    def counted(stream, algorithm):
        calls.append(stream.name)
        return original(stream, algorithm)
    monkeypatch.setattr(hashlib, 'file_digest', counted)
    run_model(context)
    assert context.stats['processed'] == 2 and calls == [str(weight)]
    context.args.resume = True
    context.stats.clear()
    run_model(context)
    assert context.stats['reused'] == 2 and len(calls) == 2
    stamp = weight.stat()
    weight.write_bytes(b'BBBB')
    os.utime(weight, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    context.stats.clear()
    run_model(context)
    assert context.stats['processed'] == 2 and not context.stats['reused']
    assert len(context.pipeline.loaded) == 2 and len(calls) == 3
