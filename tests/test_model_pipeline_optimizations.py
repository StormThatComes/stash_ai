"""Exact preprocessing, probability floors, max resume and optional backend traces."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from stash_ai.backends.onnx_preprocessing import prepare_wd
from stash_ai.backends import profiling
from stash_ai.inference import prediction, checkpoints
from stash_ai.models import ModelSettings, WD
from stash_ai.runtime.profile import ProfileMetrics
from stash_ai.runtime.report import RunReport
from stash_ai.storage.store import StateStore
from tests.helpers.inference import IndexedTagger
from tests.helpers.runtime import runtime as app

pytestmark = pytest.mark.usefixtures('reset_progress')


def original_wd(image, size):
    rgba = image.convert('RGBA')
    background = Image.new('RGBA', rgba.size, (255,255,255,255))
    background.alpha_composite(rgba)
    rgb = background.convert('RGB')
    side = max(rgb.size)
    padded = Image.new('RGB', (side,side), (255,255,255))
    padded.paste(rgb, ((side-rgb.width)//2, (side-rgb.height)//2))
    resized = padded.resize((size,size), Image.Resampling.BICUBIC) if side != size else padded
    result = np.ascontiguousarray(np.asarray(resized,dtype=np.float32)[:,:,::-1])
    for value in (rgba,background,rgb,padded,resized): value.close()
    return result


@pytest.mark.parametrize('mode', ['RGB','RGBA','L','LA','P'])
@pytest.mark.parametrize('dimensions', [(7,13),(13,7),(8,8)])
def test_wd_fast_path_preserves_all_modes_and_borrowed_image(mode, dimensions):
    rng = np.random.default_rng(17)
    with Image.fromarray(rng.integers(0,256,(*dimensions,4),dtype=np.uint8),'RGBA').convert(mode) as image:
        if mode == 'P': image.info['transparency'] = 0
        expected = original_wd(image, 8)
        np.testing.assert_array_equal(prepare_wd(image, 8), expected)
        image.getpixel((0,0))  # The RGB fast path must not close caller-owned input.


@pytest.mark.parametrize('dtype', [np.float16,np.float32,np.float64])
def test_threshold_cutoffs_match_widened_reference_at_adjacent_values(dtype):
    rng = np.random.default_rng(22)
    scores = np.concatenate((np.array([0.,1.,.4,.8],dtype=dtype),rng.random(1000).astype(dtype)))
    floors = [0.,1.,.123456789,.4,.8]
    floors += [float(value) for value in scores[:30]]
    floors += [np.nextafter(value, -np.inf) for value in floors[:]]
    floors += [np.nextafter(value, np.inf) for value in floors[:]]
    for floor in floors:
        reference = [(int(i),float(scores[i])) for i in np.flatnonzero(scores.astype(np.float64)>=floor)]
        assert prediction.selected_scores(scores,floor) == reference


@pytest.mark.parametrize('batch', [1,4,10])
def test_max_aggregation_matches_uninterrupted_and_resumed_runs(pipeline, tmp_path, monkeypatch, batch):
    monkeypatch.setattr(app,'AGGREGATION','max')
    config = ModelSettings(WD,Path('missing'),.4,.65,batch)
    reference = IndexedTagger(config,pipeline.scores['wd'],pipeline.vocab.labels)
    expected = prediction.predict_scene(app, reference,pipeline.stash,{},pipeline.cues)
    with StateStore(tmp_path/'state.db','s') as store:
        checkpoint = checkpoints.Checkpoint(store,'1','wd','sig',10,2,every=1)
        failed = IndexedTagger(config,pipeline.scores['wd'],pipeline.vocab.labels, fail_at=batch if batch < 10 else 0)
        with pytest.raises(KeyboardInterrupt):
            prediction.predict_scene(app,failed,pipeline.stash,{},pipeline.cues,checkpoint=checkpoint)
        resumed = IndexedTagger(config,pipeline.scores['wd'],pipeline.vocab.labels)
        checkpoint = checkpoints.Checkpoint(store,'1','wd','sig',10,2,every=1)
        assert prediction.predict_scene(app,resumed,pipeline.stash,{},pipeline.cues,checkpoint=checkpoint) == expected
        np.testing.assert_array_equal(checkpoint.top_scores,np.max(pipeline.scores['wd'],axis=0,keepdims=True))


def test_trace_reports_observed_node_providers_without_names_or_configuration(tmp_path):
    path=tmp_path/'trace.json'
    path.write_text(json.dumps([
        dict(cat='Node',name='private graph label',dur=120,args=dict(provider='CPUExecutionProvider')),
        dict(cat='Node',dur=80,args=dict(provider='CUDAExecutionProvider')),
        dict(cat='Node',dur=20,args=dict(provider='CUDAExecutionProvider')),
        dict(cat='Session',dur=99,args=dict(provider='CUDAExecutionProvider'))]))
    report=RunReport(write=lambda _:None)
    report.profile=ProfileMetrics()
    profiling.record_onnx(report,SimpleNamespace(end_profiling=lambda:str(path)),'wd')
    observed=report.profile.models['wd']['observed_node_providers']
    assert observed['node_events']=={'CPUExecutionProvider':1,'CUDAExecutionProvider':2}
    assert observed['seconds']=={'CPUExecutionProvider':.00012,'CUDAExecutionProvider':.0001}
    assert 'private' not in json.dumps(observed)


def test_gpu_events_need_no_extra_synchronize_and_profile_failure_is_optional():
    report=RunReport(write=lambda _:None)
    report.profile=ProfileMetrics()
    events=[SimpleNamespace(query=lambda:True,elapsed_time=lambda end:2.) for _ in range(3)]
    profiling.record_gpu(report,events,'joytag','fresh',4)
    assert {r['name'] for r in report.profile.export()}=={'gpu_input_transfer','gpu_model_compute'}
    assert all(r['units']==4 and r['inclusive_seconds']==.002 for r in report.profile.export())
    events[0].elapsed_time=lambda _: (_ for _ in ()).throw(RuntimeError('unavailable'))
    profiling.record_gpu(report,events,'joytag','fresh',4)
    assert report.counts['backend_gpu_profile_failures']==1
    assert profiling.gpu_events(None,SimpleNamespace()) is None


def test_native_subclasses_cannot_bypass_the_checked_boundary():
    from stash_ai.backends.onnx import OnnxTagger
    from stash_ai.backends.predictions import checked_prediction
    class External(OnnxTagger):
        def predict_batch(self, inputs): return np.array([[np.nan]],np.float32)
    tagger=External.__new__(External)
    tagger.labels=['tag'];tagger.settings=SimpleNamespace(source=WD)
    with pytest.raises(RuntimeError,match='invalid probabilities'):
        checked_prediction(tagger,[object()],1,collated=False)


def test_onnx_profile_setup_failure_falls_back_to_normal_session(monkeypatch):
    import sys
    from stash_ai.backends import onnx
    monkeypatch.setitem(sys.modules, 'onnxruntime', SimpleNamespace())
    monkeypatch.setattr(onnx, 'ensure_files', lambda *args: None)
    attempts = []
    session = SimpleNamespace()
    def open_session(ort, settings, **kwargs):
        attempts.append(bool(kwargs))
        if kwargs:
            raise OSError('profiling unavailable')
        return session, 'cpu'
    monkeypatch.setattr(onnx, '_open_session', open_session)
    monkeypatch.setattr(onnx, '_inspect_model', lambda *args: SimpleNamespace(
        input_name='input',output_name='output',mask_name=None,fixed_batch=None,
        output_count=1,output_indices=None,size=8,batch_size=1))
    report = RunReport(write=lambda _: None)
    report.profile = ProfileMetrics()
    settings = SimpleNamespace(source=WD)
    with profiling.use(report, enabled=True):
        tagger = onnx.OnnxTagger(settings, SimpleNamespace(labels=['tag']))
    assert attempts == [True, False]
    assert not tagger.node_profile_enabled and tagger.profile_directory is None
    assert report.counts['backend_onnx_profile_setup_failures'] == 1
    tagger.close()
