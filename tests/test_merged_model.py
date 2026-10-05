"""Merged model lifecycle against real input signatures and SQLite checkpoints."""
from collections import Counter
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


def run_model(context):
    MergedModelRun(
        app, context.args, context.state, context.config, context.pipeline.vocab,
        stats=context.stats, report=context.args.report,
    ).run(context.scenes, ['scene', 'markers'], context.inputs)


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
    assert len(context.pipeline.loaded) == 1
    assert context.pipeline.loaded[0].closed
    # The borrowed database remains usable after the model's finally block.
    assert context.state.connection.execute('SELECT 1').fetchone() == (1,)


def test_loading_failure_stops_remaining_scenes_without_retrying(model_run, monkeypatch):
    context = model_run
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


@pytest.mark.parametrize('reason', ['missing-input', 'completed'])
def test_skipped_and_completed_scenes_do_not_load_a_model(model_run, monkeypatch, reason):
    context = model_run
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
    _, _, expected = inspect_input(app, context.pipeline.stash, context.scenes['1']['paths'], after)
    for scene_id in context.scenes:
        for kind in ('scene', 'markers'):
            assert context.state.records(scene_id, kind)['wd']['meta']['inference_signature'] == expected

    context.args.resume = True
    context.stats.clear()
    monkeypatch.setattr(app.services, 'load_tagger', lambda *args: pytest.fail('downloaded weights invalidated resume'))
    run_model(context)
    assert context.stats['reused'] == 2 and not context.stats['processed']

