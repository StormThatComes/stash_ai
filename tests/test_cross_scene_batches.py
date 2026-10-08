"""Scene-boundary overlap with owner-thread SQLite and bounded speculation."""
from threading import Event, current_thread
from types import SimpleNamespace

import pytest

from stash_ai.inference import checkpoints, prediction, results, sprites
from stash_ai.runtime.profile import ProfileMetrics
from tests.helpers.inference import sprite_frames
from tests.helpers.runtime import runtime as app
from tests.test_merged_model import model_run, run_model, assert_inference_workers_closed


pytestmark = pytest.mark.usefixtures('reset_progress')


def instrument(context, monkeypatch, *, predict=None, await_ready=True, score_index=lambda index: index):
    """Identify each scene in CPU inputs without depending on timing or network."""
    for scene, current in context.scenes.items():
        current['paths']['vtt'] = '/vtt/' + scene
    ready, gpu_next = Event(), Event()
    calls, cpu_threads, gpu_threads = [], set(), set()
    load = app.services.load_tagger

    def frames(media, paths, cues):
        for cue, image in sprite_frames(cues):
            image.info['scene'] = paths['vtt'].rsplit('/', 1)[-1]
            yield cue, image

    def load_model(*args):
        model = load(*args)
        original = model.predict_batch

        def prepare(image):
            cpu_threads.add(current_thread())
            return image.info['scene'], image.getpixel((0, 0))

        def collate(images):
            if images[0][0] == '2':
                ready.set()
            return tuple(images)

        def forward(images, count):
            scene = images[0][0]
            if scene == '1' and await_ready:
                assert ready.wait(5), 'Next scene CPU batches waited for the scene boundary'
            gpu_threads.add(current_thread())
            calls.append((scene, tuple(index for _, index in images)))
            if scene == '2':
                gpu_next.set()
            if predict is not None:
                predict(scene, images)
            return original([score_index(index) for _, index in images])

        model.prepare, model.collate_batch, model.predict_collated_batch = prepare, collate, forward
        return model

    monkeypatch.setattr(sprites, 'iter_sprite_frames', frames)
    monkeypatch.setattr(app.services, 'load_tagger', load_model)
    return ready, gpu_next, calls, cpu_threads, gpu_threads


@pytest.mark.parametrize('stage', ['finalize', 'capture', 'commit'])
def test_next_gpu_overlaps_scene_finalization_and_matches_sequential_results(model_run, monkeypatch, stage):
    context = model_run
    context.args.prefetch_batches, context.args.prepare_workers = 16, 8
    context.args.report.profile = ProfileMetrics()
    monkeypatch.setattr(results, 'now', lambda: '2026-10-08T12:00:00+00:00')
    owner, checked = current_thread(), []
    sql_threads = []
    context.state.connection.set_trace_callback(lambda _: sql_threads.append(current_thread()))
    with monkeypatch.context() as patch:
        ready, gpu_next, calls, cpu_threads, gpu_threads = instrument(context, patch)
        if stage == 'finalize':
            target, name = prediction, '_finalize_predictions'
        elif stage == 'capture':
            target, name = results, 'capture_outputs'
        else:
            target, name = checkpoints, 'finish'
        original = getattr(target, name)

        def finish(*args, **kwargs):
            assert current_thread() is owner
            if not checked:
                checked.append(True)
                assert ready.is_set()
                assert gpu_next.wait(5), 'Next inference waited for scene finalization/commit'
                assert context.state.records('1', 'scene') == {}
                assert context.state.records('2', 'scene') == {}
            return original(*args, **kwargs)

        patch.setattr(target, name, finish)
        run_model(context)
        assert context.stats['processed'] == 2 and not context.stats['errors']
        assert len(cpu_threads) <= 8 and owner not in cpu_threads
        assert len(gpu_threads) == 1 and owner not in gpu_threads
        assert calls == [(scene, tuple(range(start, start + 2)))
                         for scene in ('1', '2') for start in range(0, 10, 2)]
        assert_inference_workers_closed()
    assert set(sql_threads) == {owner}
    optimized = {(scene, kind): context.state.records(scene, kind)
                 for scene in context.scenes for kind in ('scene', 'markers')}
    context.args.prefetch_batches = 0
    run_model(context)
    assert optimized == {(scene, kind): context.state.records(scene, kind)
                         for scene in context.scenes for kind in ('scene', 'markers')}
    metrics = context.args.report.profile.export()
    assert sum(row['calls'] for row in metrics if row['name'] == 'scene_handoff') == 1


@pytest.mark.parametrize('result_prefetch', [1, 2, 4])
def test_next_gpu_results_are_bounded_while_previous_scene_is_finalized(model_run, monkeypatch, result_prefetch):
    context = model_run
    context.args.prefetch_batches = 16
    context.args.result_prefetch_batches = result_prefetch
    full = Event()
    next_calls = []

    def predict(scene, images):
        if scene == '2':
            next_calls.append(images)
            if len(next_calls) == result_prefetch + 1:
                full.set()

    instrument(context, monkeypatch, predict=predict)
    original = prediction._finalize_predictions
    finalized = []

    def finalize(*args, **kwargs):
        if not finalized:
            finalized.append(True)
            assert full.wait(5)
            # N queued results and one waiting to be enqueued. No result of
            # the next scene is consumed during this scene's finalization.
            assert len(next_calls) == result_prefetch + 1
        return original(*args, **kwargs)

    monkeypatch.setattr(prediction, '_finalize_predictions', finalize)
    run_model(context)
    assert len(next_calls) == 5
    assert_inference_workers_closed()


def test_next_scene_can_finish_inference_during_previous_commit(model_run, monkeypatch):
    context = model_run
    context.args.prefetch_batches = 16
    assert context.args.result_prefetch_batches == 32
    next_finished = Event()

    def predict(scene, images):
        if scene == '2' and images[0][1] == 8:
            next_finished.set()

    _, _, calls, _, gpu_threads = instrument(context, monkeypatch, predict=predict)
    original = checkpoints.finish
    owner = current_thread()
    sql_threads = []
    context.state.connection.set_trace_callback(lambda _: sql_threads.append(current_thread()))

    def commit(store, scene, *args):
        assert current_thread() is owner
        if scene == '1':
            assert next_finished.wait(5), 'CPU result queue blocked scene 2 during scene 1 commit'
            assert len([value for value in calls if value[0] == '2']) == 5
            assert not store.records('2', 'scene')
        return original(store, scene, *args)

    monkeypatch.setattr(checkpoints, 'finish', commit)
    run_model(context)
    assert context.stats['processed'] == 2 and not context.stats['errors']
    assert len(gpu_threads) == 1 and set(sql_threads) == {owner}
    assert context.state.records('2', 'scene')
    assert_inference_workers_closed()


@pytest.mark.parametrize('prefetch', [1, 16, 128])
def test_handoff_continues_across_three_scenes(model_run, monkeypatch, prefetch):
    context = model_run
    context.args.prefetch_batches, context.args.prepare_workers = prefetch, 8
    # Keep scene 2 running until its owner starts prefetching scene 3.
    context.args.result_prefetch_batches = 2
    context.scenes['3'] = dict(paths=dict(vtt='/vtt/3', sprite='/sprite'))
    cpu_third, gpu_third = Event(), Event()

    def predict(scene, images):
        if scene == '2' and images[0][1] == 8:
            assert cpu_third.wait(5), 'Third scene CPU queue waited for second finalization'
        if scene == '3':
            gpu_third.set()

    _, _, calls, cpu_threads, gpu_threads = instrument(context, monkeypatch, predict=predict)
    load, finish = app.services.load_tagger, checkpoints.finish

    def load_model(*args):
        model = load(*args)
        collate = model.collate_batch

        def prepare_batch(images):
            value = collate(images)
            if images[0][0] == '3':
                cpu_third.set()
            return value

        model.collate_batch = prepare_batch
        return model

    def commit(store, scene, *args):
        if scene == '2':
            assert gpu_third.wait(5), 'Third scene GPU queue waited for second commit'
        return finish(store, scene, *args)

    monkeypatch.setattr(app.services, 'load_tagger', load_model)
    monkeypatch.setattr(checkpoints, 'finish', commit)
    run_model(context)
    assert context.stats['processed'] == 3 and not context.stats['errors']
    assert [scene for scene, _ in calls] == ['1'] * 5 + ['2'] * 5 + ['3'] * 5
    assert len(cpu_threads) <= 8 and len(gpu_threads) == 1
    assert all(context.state.records(scene, 'scene') for scene in context.scenes)
    assert_inference_workers_closed()


@pytest.mark.parametrize('prefetch', [1, 16, 128])
def test_next_input_and_cpu_batches_start_while_long_current_queue_is_full(model_run, monkeypatch, prefetch):
    from stash_ai.domain import Cue
    from stash_ai.inference.input_identity import cue_manifest, inference_signature
    context = model_run
    context.args.prefetch_batches, context.args.prepare_workers = prefetch, 8
    context.args.result_prefetch_batches = 2
    context.scenes['3'] = dict(paths=dict(vtt='/vtt/3', sprite='/sprite'))
    third_verified, third_prepared = Event(), Event()
    inspect = context.inputs.inspect

    def inspect_scene(*args):
        cues, media, signature = inspect(*args)
        if args[1]['vtt'] == '/vtt/2':
            cues = [Cue(index, index + 1, 'sprite', (0, 0, 1, 1))
                    for index in range(2 * (prefetch + 8))]
            signature = inference_signature(args[2], cue_manifest(cues), media.hashes)
        if args[1]['vtt'] == '/vtt/3':
            third_verified.set()
        return cues, media, signature

    def predict(scene, images):
        if scene == '2' and images[0][1] == 8:
            assert third_verified.wait(5), 'Full current CPU queue blocked next input verification'
            assert third_prepared.wait(5), 'Full current CPU queue blocked next scene preparation'

    _, _, calls, cpu_threads, gpu_threads = instrument(context, monkeypatch, predict=predict,
                                                       score_index=lambda index: index % 10)
    load = app.services.load_tagger

    def load_model(*args):
        model = load(*args)
        collate = model.collate_batch

        def collate_batch(images):
            value = collate(images)
            if images[0][0] == '3':
                third_prepared.set()
            return value

        model.collate_batch = collate_batch
        return model

    monkeypatch.setattr(context.inputs, 'inspect', inspect_scene)
    monkeypatch.setattr(app.services, 'load_tagger', load_model)
    run_model(context)
    assert context.stats['processed'] == 3 and not context.stats['errors']
    assert len([scene for scene, _ in calls if scene == '2']) == prefetch + 8
    assert len(cpu_threads) <= 8 and len(gpu_threads) == 1
    assert_inference_workers_closed()


@pytest.mark.parametrize('error', [RuntimeError, KeyboardInterrupt])
def test_future_gpu_error_waits_for_its_scene_and_preserves_processed_prefix(model_run, monkeypatch, error):
    context = model_run
    context.args.prefetch_batches = 2
    failed = Event()

    def predict(scene, images):
        if scene == '2' and images[0][1] == 4:
            failed.set()
            raise error('future GPU failed')

    instrument(context, monkeypatch, predict=predict)
    original = checkpoints.finish

    def finish(store, scene, *args):
        if scene == '1':
            assert failed.wait(5), 'Future GPU did not start before current commit'
            assert not context.stats['errors']
        return original(store, scene, *args)

    monkeypatch.setattr(checkpoints, 'finish', finish)
    if error is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt, match='future GPU failed'):
            run_model(context)
    else:
        run_model(context)
        assert context.stats['errors'] == 1
    assert context.stats['processed'] == 1
    assert context.state.records('1', 'scene') and not context.state.records('2', 'scene')
    assert context.state.connection.execute('SELECT scene,processed FROM checkpoints').fetchall() == [('2', 4)]
    assert_inference_workers_closed()


def test_interrupt_during_previous_finalization_discards_future_results(model_run, monkeypatch):
    context = model_run
    context.args.prefetch_batches = 16
    _, gpu_next, _, _, _ = instrument(context, monkeypatch)

    def interrupt(*args, **kwargs):
        assert gpu_next.wait(5)
        raise KeyboardInterrupt('finalization interrupted')

    monkeypatch.setattr(prediction, '_finalize_predictions', interrupt)
    with pytest.raises(KeyboardInterrupt, match='finalization interrupted'):
        run_model(context)
    assert not context.state.records('1', 'scene') and not context.state.records('2', 'scene')
    assert context.state.connection.execute('SELECT scene,processed FROM checkpoints').fetchall() == [('1', 10)]
    from stash_ai.storage.progress import pending_refreshes
    assert set(pending_refreshes(context.state, ['scene', 'markers'])) == {'1', '2'}
    assert context.pipeline.loaded[0].closed
    assert_inference_workers_closed()


def test_next_scene_checkpoint_is_restored_before_speculative_gpu_work(model_run, monkeypatch):
    context = model_run
    context.args.prefetch_batches = 2

    def fail_later(scene, images):
        if scene == '2' and images[0][1] == 4:
            raise RuntimeError('create partial checkpoint')

    with monkeypatch.context() as patch:
        instrument(context, patch, predict=fail_later)
        run_model(context)
    assert context.state.connection.execute('SELECT scene,processed FROM checkpoints').fetchall() == [('2', 4)]
    context.args.resume = True
    context.stats.clear()
    from stash_ai.storage.progress import request_refresh
    request_refresh(context.state, ['1'], ['wd'], ['scene', 'markers'])
    _, _, calls, _, _ = instrument(context, monkeypatch)
    run_model(context)
    assert [indexes for scene, indexes in calls if scene == '2'] == [(4, 5), (6, 7), (8, 9)]
    assert context.stats['processed'] == 2 and not context.stats['errors']
    assert context.state.records('1', 'scene') and context.state.records('2', 'scene')
    assert context.state.connection.execute('SELECT 1 FROM checkpoints').fetchall() == []
    assert_inference_workers_closed()


def test_slow_next_input_does_not_delay_previous_scene_finalization(model_run, monkeypatch):
    context = model_run
    context.args.prefetch_batches = 2
    context.scenes['2']['paths']['vtt'] = '/vtt/2'
    started, released, finished = Event(), Event(), Event()
    inspect, load = context.inputs.inspect, app.services.load_tagger
    finalize, commit = prediction._finalize_predictions, checkpoints.finish

    def inspect_next(*args):
        if args[1]['vtt'] == '/vtt/2':
            started.set()
            assert released.wait(5), 'Finalization waited for a slow next input'
            result = inspect(*args)
            finished.set()
            return result
        return inspect(*args)

    def load_model(*args):
        model = load(*args)
        predict = model.predict_batch

        def forward(images):
            assert started.wait(5)
            return predict(images)

        model.predict_batch = forward
        return model

    def finalize_current(*args, **kwargs):
        if not released.is_set():
            assert not finished.is_set()
        return finalize(*args, **kwargs)

    def commit_current(store, scene, *args):
        if scene == '1':
            assert not finished.is_set()
            result = commit(store, scene, *args)
            released.set()
            return result
        return commit(store, scene, *args)

    monkeypatch.setattr(context.inputs, 'inspect', inspect_next)
    monkeypatch.setattr(app.services, 'load_tagger', load_model)
    monkeypatch.setattr(prediction, '_finalize_predictions', finalize_current)
    monkeypatch.setattr(checkpoints, 'finish', commit_current)
    run_model(context)
    assert context.stats['processed'] == 2 and not context.stats['errors']
    assert_inference_workers_closed()


def test_input_finishing_during_finalization_can_start_gpu_before_capture(model_run, monkeypatch):
    context = model_run
    context.args.prefetch_batches = 2
    started, released = Event(), Event()
    inspect, finalize, capture = context.inputs.inspect, prediction._finalize_predictions, results.capture_outputs

    def inspect_next(*args):
        if args[1]['vtt'] == '/vtt/2':
            started.set()
            assert released.wait(5)
        return inspect(*args)

    def predict(scene, images):
        if scene == '1':
            assert started.wait(5)

    ready, gpu_next, _, _, _ = instrument(context, monkeypatch, predict=predict, await_ready=False)

    def finalize_first(*args, **kwargs):
        if not released.is_set():
            released.set()
            assert ready.wait(5)
        return finalize(*args, **kwargs)

    def capture_first(*args, **kwargs):
        assert gpu_next.wait(5), 'Input ready during finalization waited until the next scene'
        return capture(*args, **kwargs)

    monkeypatch.setattr(context.inputs, 'inspect', inspect_next)
    monkeypatch.setattr(prediction, '_finalize_predictions', finalize_first)
    monkeypatch.setattr(results, 'capture_outputs', capture_first)
    run_model(context)
    assert context.stats['processed'] == 2 and not context.stats['errors']
    assert_inference_workers_closed()


def test_future_checkpoint_failure_does_not_prevent_previous_commit(model_run, monkeypatch):
    context = model_run
    context.args.prefetch_batches = 2
    instrument(context, monkeypatch)
    original = checkpoints.Checkpoint.restore

    def restore(checkpoint):
        if checkpoint.scene == '2':
            raise ValueError('future checkpoint is invalid')
        return original(checkpoint)

    monkeypatch.setattr(checkpoints.Checkpoint, 'restore', restore)
    run_model(context)
    assert context.stats['processed'] == context.stats['errors'] == 1
    assert context.state.records('1', 'scene') and not context.state.records('2', 'scene')
    assert_inference_workers_closed()


def test_future_decode_failure_keeps_valid_prefix_and_previous_scene(model_run, monkeypatch):
    context = model_run
    context.args.prefetch_batches = 2
    instrument(context, monkeypatch)
    frames = sprites.iter_sprite_frames

    def read(media, paths, cues):
        for cue, image in frames(media, paths, cues):
            if paths['vtt'] == '/vtt/2' and cue.start == 4:
                raise OSError('future sprite decode failed')
            yield cue, image

    monkeypatch.setattr(sprites, 'iter_sprite_frames', read)
    run_model(context)
    assert context.stats['processed'] == context.stats['errors'] == 1
    assert context.state.records('1', 'scene') and not context.state.records('2', 'scene')
    assert context.state.connection.execute('SELECT scene,processed FROM checkpoints').fetchall() == [('2', 4)]
    assert_inference_workers_closed()


@pytest.mark.parametrize('interrupt', [False, True])
def test_history_keeps_decisions_and_proofs_on_the_correct_scene(model_run, monkeypatch, tmp_path, interrupt):
    from stash_ai.history.store import HistoryStore
    from stash_ai.signatures import digest
    context = model_run
    context.args.prefetch_batches = 2
    owner = current_thread()
    _, gpu_next, _, _, _ = instrument(context, monkeypatch)
    with HistoryStore(tmp_path / 'history.sqlite3') as history:
        history.start('run', 'fixture', 'inference', '2026-10-08T12:00:00+00:00', {})

        def call(operation, *args):
            assert current_thread() is owner
            return getattr(history, operation)(*args)

        context.args.report.history = SimpleNamespace(store=history, identifier='run', call=call)
        if interrupt:
            def stop(*args, **kwargs):
                assert gpu_next.wait(5)
                raise KeyboardInterrupt('stop during finalization')

            monkeypatch.setattr(prediction, '_finalize_predictions', stop)
            with pytest.raises(KeyboardInterrupt, match='stop during finalization'):
                run_model(context)
        else:
            run_model(context)
        history.flush()
        assert {row['scene']: row['status'] for row in history.decisions('fixture')} == {
            '1': 'interrupted' if interrupt else 'finished',
            '2': 'interrupted' if interrupt else 'finished',
        }
        if not interrupt:
            for scene in context.scenes:
                for kind in ('scene', 'markers'):
                    signature = context.state.records(scene, kind)['wd']['meta']['inference_signature']
                    proof = history.get_proof('fixture', scene, 'wd', kind, signature)
                    assert proof['vtt']['location'] == digest(app.STASH_URL.rstrip('/') + '/vtt/' + scene)
    assert_inference_workers_closed()
