"""Inference failures and output modes must preserve publication dependencies."""
from collections import Counter
from types import SimpleNamespace
import pytest
from tests.helpers.runtime import runtime as app
from tests.helpers.evidence import inference_record, empty_policies, segment
from tests.helpers.cli import runtime_args
from stash_ai.storage.store import StateStore, scope_for
from stash_ai.evidence import pipeline

pytestmark = pytest.mark.usefixtures('reset_progress')


def test_fresh_markers_only_retains_occurrence_evidence(pipeline):
    p = pipeline
    p.scores['wd'][:] = 0
    p.scores['wd'][3, 0] = .95
    assert app.main([*p.args, '--temporal-policy', str(app.INSTALL_ROOT/'temporal_policy.yml'), '--markers-only']) == 0
    with StateStore(p.path, scope_for(app.STASH_URL), True) as state:
        assert state.records('1', 'scene')['wd']['items']
    assert not p.stash.scene['tags']
    assert len(p.stash.scene['scene_markers']) == 1
    assert p.stash.scene['scene_markers'][0]['seconds'] == 3



def test_input_error_blocks_fresh_stale_mix(pipeline, monkeypatch):
    from stash_ai.inference.inputs import RunInputs
    from stash_ai.storage.progress import pending_refreshes
    from stash_ai.domain import Cue
    p = pipeline
    args = ['--models', 'wd', 'dtq', *p.args[2:]]
    assert app.main(args) == 0
    with StateStore(p.path, scope_for(app.STASH_URL), True) as state:
        old = state.records('1', 'scene')['wd']['meta']['frame_input_signature']
    p.cues = [Cue(c.start, c.end-.1, c.image, c.rect) for c in p.cues]
    p.scores['dtq'][:] = 0
    original = RunInputs.inspect
    def inspect(self, runtime, paths, identity):
        if identity['source'] == 'wd':
            raise OSError('VTT fetch failed before identity inspection')
        return original(self, runtime, paths, identity)
    monkeypatch.setattr(RunInputs, 'inspect', inspect)
    p.stash.writes.clear()
    assert app.main(args) == 1
    assert not p.stash.writes
    with StateStore(p.path, scope_for(app.STASH_URL), True) as state:
        from stash_ai.storage.input_checks import pending
        assert pending_refreshes(state, ['scene', 'markers']) == {}
        assert pending(state, '1') == {'wd': 'read_error'}
        assert state.records('1', 'scene')['dtq']['meta']['frame_input_signature'] != old
        published = state.snapshot('1', 'scene')[0]
        assert any(h['source'] == 'wd' and h['meta']['frame_input_signature'] == old for h in published['contributions'])
    monkeypatch.setattr(RunInputs, 'inspect', original)
    loaded = len(p.loaded)
    assert app.main(args) == 0
    assert [model.settings.source.key for model in p.loaded[loaded:]] == ['wd']


def test_model_load_failure_blocks_all_remaining_scenes(tmp_path, monkeypatch):
    from stash_ai.models import ModelSettings, Vocabulary, WD
    from stash_ai.runs.merged_model import MergedModelRun
    from stash_ai.storage.progress import pending_refreshes
    from stash_ai.domain import Cue
    config = ModelSettings(WD, tmp_path/'weights', .4, .7, 1)
    cues = [Cue(0, 1, 'sprite', (0,0,1,1))]
    inputs = SimpleNamespace(inspect=lambda *a:(cues, SimpleNamespace(hashes={}), 'new-input'))
    monkeypatch.setattr(app.services, 'load_tagger', lambda *a: (_ for _ in ()).throw(OSError('missing weights')))
    with StateStore(tmp_path/'state.sqlite3', 's') as state:
        record = inference_record(mode='frames', items=[dict(start=0, end=1, tags=[['core',.9]])])
        for scene in ['1', '2']:
            state.put(scene, 'wd', {'markers': record})
        args = runtime_args(resume=False, restart=False)
        MergedModelRun(app, args, state, config, Vocabulary(['core'], {}), stats=Counter(), report=args.report).run(
            {s:{'paths':{'vtt':'/v','sprite':'/s'}} for s in ['1','2']}, ['markers'], inputs)
        assert pending_refreshes(state, ['markers']) == {s:[('wd','markers')] for s in ['1', '2']}


def test_scene_only_does_not_read_unneeded_marker_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(app,'CREATE_MARKERS',False)
    with StateStore(tmp_path/'s.db','s') as state:
        state.put('1','wd',{'scene':inference_record(mode='scene',items=[{'tag':'a','score':.9}])})
        bad = segment('wd',[[('a',.9)]])
        bad['meta']['analysis_format'] = 4
        with state.connection:
            state.connection.execute('INSERT INTO results VALUES (?,?,?,?,?)', ('s','1','wd','markers',state.pack(bad)))
        hits, spans = pipeline.evidence(app,state,'1',empty_policies(),empty_policies())
        assert [hit['tag'] for hit in hits] == ['a']
        assert spans == []
