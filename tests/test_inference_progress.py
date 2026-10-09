"""Frame progress follows consumed results, durable resume and scene ownership."""
from io import StringIO
from threading import Event, current_thread

import pytest
from tqdm import tqdm

from stash_ai.inference import checkpoints
from stash_ai.runs import merged_model
from tests.helpers.runtime import runtime as app
from tests.test_cross_scene_batches import instrument
from tests.test_merged_model import model_run, run_model, assert_inference_workers_closed


pytestmark = pytest.mark.usefixtures('reset_progress')


class Terminal(StringIO):
    def isatty(self):
        return True


@pytest.fixture
def bars(monkeypatch):
    captured = []
    owner = current_thread()

    class RecordedBar(tqdm):
        def __init__(self, *args, **kwargs):
            self.updates, self.closed = [], False
            kwargs['file'] = Terminal()
            super().__init__(*args, **kwargs)
            captured.append(self)

        def update(self, count=1):
            assert current_thread() is owner, 'A background producer updated UI progress'
            self.updates.append(count)
            return super().update(count)

        def close(self):
            self.closed = True
            return super().close()

    monkeypatch.setattr(merged_model, 'tqdm', RecordedBar)
    return captured


def frame_bars(bars):
    return [bar for bar in bars if getattr(bar, 'unit', None) == 'frame']


@pytest.mark.parametrize('prefetch', [0, 2])
def test_frame_bar_counts_partial_batches_even_without_retained_tags(pipeline, bars, prefetch):
    pipeline.scores['wd'][:] = 0
    assert app.main([*pipeline.args, '--prefetch-batches', str(prefetch)]) == 0
    scene_bar, frame_bar = bars[:2]
    assert scene_bar.pos == 0 and scene_bar.n == scene_bar.total == 1
    assert frame_bar.pos == -1 and frame_bar.desc == 'wd scene 1'
    assert frame_bar.initial == 0 and frame_bar.n == frame_bar.total == 10
    assert frame_bar.updates == [4, 4, 2]
    assert [bar.pos for bar in bars[2:]] == [-2, -3]
    assert bars[2].desc.startswith('Buffers: in=')
    assert bars[3].desc.startswith('Cache:') and 'checkpoint interval=0/8' in bars[3].desc
    assert not frame_bar.leave and all(bar.closed for bar in bars)


def test_interruption_resume_and_cache_reuse_keep_frame_progress_accurate(pipeline, bars):
    args = [*pipeline.args, '--prefetch-batches', '2']
    pipeline.fail['wd'] = 4
    with pytest.raises(KeyboardInterrupt):
        app.main(args)
    interrupted = frame_bars(bars)[0]
    assert interrupted.n == 4 and interrupted.updates == [4]
    assert all(bar.closed for bar in bars)
    pipeline.fail.clear()
    assert app.main(args) == 0
    resumed = frame_bars(bars)[1]
    assert resumed.initial == 4 and resumed.n == resumed.total == 10
    assert resumed.updates == [4, 2]
    assert pipeline.loaded[-1].seen == list(range(4, 10))
    assert app.main(args) == 0
    assert len(frame_bars(bars)) == 2  # Completed cache does not fabricate frame work.
    assert all(bar.closed for bar in bars)
    assert_inference_workers_closed()


def test_fully_restored_checkpoint_starts_at_complete_frame_count(pipeline, bars, monkeypatch):
    finish = checkpoints.finish

    def fail_commit(*args):
        raise RuntimeError('commit failed after frame processing')

    with monkeypatch.context() as patch:
        patch.setattr(checkpoints, 'finish', fail_commit)
        assert app.main(pipeline.args) == 1
    assert checkpoints.finish is finish
    assert app.main(pipeline.args) == 0
    resumed = frame_bars(bars)[1]
    assert resumed.initial == resumed.n == resumed.total == 10
    assert resumed.updates == [] and pipeline.loaded[-1].seen == []
    assert all(bar.closed for bar in bars)


def test_next_scene_gpu_work_does_not_advance_current_frame_bar(model_run, bars, monkeypatch):
    context = model_run
    context.args.prefetch_batches = 16
    next_finished = Event()

    def predict(scene, images):
        if scene == '2' and images[0][1] == 8:
            next_finished.set()

    instrument(context, monkeypatch, predict=predict)
    finish = checkpoints.finish

    def commit(store, scene, *args):
        if scene == '1':
            assert next_finished.wait(5), 'Next scene did not run during the current commit'
            current = frame_bars(bars)
            assert len(current) == 1 and current[0].desc == 'wd scene 1'
            assert current[0].n == 10 and current[0].updates == [2] * 5
            assert not current[0].closed
        return finish(store, scene, *args)

    monkeypatch.setattr(checkpoints, 'finish', commit)
    run_model(context)
    assert context.stats['processed'] == 2 and not context.stats['errors']
    assert [bar.desc for bar in frame_bars(bars)] == ['wd scene 1', 'wd scene 2']
    assert all(bar.n == bar.total == 10 for bar in frame_bars(bars))
    assert all(bar.closed for bar in bars)
    assert_inference_workers_closed()


def test_frame_bar_is_hidden_when_output_is_redirected(pipeline, monkeypatch):
    captured = []
    output = StringIO()

    def redirected_bar(*args, **kwargs):
        kwargs['file'] = output
        bar = tqdm(*args, **kwargs)
        captured.append(bar)
        return bar

    monkeypatch.setattr(merged_model, 'tqdm', redirected_bar)
    monkeypatch.setattr(merged_model.MergedModelRun, '_buffer_snapshot',
                        lambda *args: pytest.fail('Redirected output sampled buffer state'))
    assert app.main(pipeline.args) == 0
    assert len(captured) == 2 and captured[1].disable
    assert 'wd inference' in output.getvalue()
    assert 'wd scene 1' not in output.getvalue() and '\x1b[A' not in output.getvalue()
