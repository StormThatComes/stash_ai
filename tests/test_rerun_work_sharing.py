"""Count actual reads/work and verify ownership across cached/stale passes."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import sqlite3

import pytest

from stash_ai.inference.inputs import RunInputs
from stash_ai.runs.merged_model import MergedModelRun
from stash_ai.storage.store import StateStore, scope_for
from tests.helpers.runtime import runtime as app
from tests.test_cached_input_reuse import media_requests, model_args
from tests.test_result_thresholds import outputs

pytestmark = pytest.mark.usefixtures('reset_progress')


def test_ready_pair_is_planned_once_and_reads_provenance_as_one_batch(pipeline, monkeypatch):
    requests = media_requests(pipeline, monkeypatch)
    args = model_args(pipeline, 'all')
    assert app.main(args) == 0
    calls, statements = Counter(), []
    plan = MergedModelRun.cached_plan
    def counted(self, scene, paths, kinds):
        calls[scene, self.key] += 1
        return plan(self, scene, paths, kinds)
    monkeypatch.setattr(MergedModelRun, 'cached_plan', counted)
    old_enter = StateStore.__enter__
    def traced(self):
        result = old_enter(self)
        self.connection.set_trace_callback(statements.append)
        return result
    monkeypatch.setattr(StateStore, '__enter__', traced)
    requests.clear()
    assert app.main(args) == 0
    assert calls == {('1', model): 1 for model in ('wd', 'dtq', 'joytag', 'oppai')}
    assert not requests
    assert len([sql for sql in statements if sql.startswith('WITH wanted') and
                'inference_identities' in sql]) == 1
    assert not any(sql.startswith('SELECT data FROM inference_identities') for sql in statements)


def test_one_manifest_snapshot_is_shared_by_four_models_and_released_before_merge(pipeline, monkeypatch):
    import stash_ai.inference.inputs as inputs_module
    from stash_ai.runs.merged import MergedRun
    requests = media_requests(pipeline, monkeypatch)
    snapshots, builds, closed = [], [], []
    manifest, inspect, close, merge = inputs_module.cue_manifest, RunInputs.inspect, RunInputs.close, MergedRun._merge_scenes
    def counted(cues):
        builds.append(True)
        return manifest(cues)
    def observed(self, *args):
        result = inspect(self, *args)
        snapshots.append(result.snapshot)
        return result
    def released(self):
        close(self)
        assert not self.identities and not self.media.hashes
        assert not any(key[0] in ('input', 'media') and key[1] in (self.token, self.media.token)
                       for key in self.cache.entries if isinstance(key, tuple) and len(key) > 1)
        closed.append(True)
    def checked_merge(self):
        assert closed == [True]
        return merge(self)
    monkeypatch.setattr(inputs_module, 'cue_manifest', counted)
    monkeypatch.setattr(RunInputs, 'inspect', observed)
    monkeypatch.setattr(RunInputs, 'close', released)
    monkeypatch.setattr(MergedRun, '_merge_scenes', checked_merge)
    assert app.main(model_args(pipeline, 'all')) == 0
    assert len(snapshots) == 4 and all(s is snapshots[0] for s in snapshots)
    assert len(builds) == 1 and len(requests) == 2
    with pytest.raises(TypeError):
        snapshots[0].manifest[0][0] = 99
    with pytest.raises(TypeError):
        snapshots[0].hashes.clear()


def test_release_waits_for_stale_peer_revisit(pipeline, monkeypatch):
    media_requests(pipeline, monkeypatch)
    args = model_args(pipeline, 'wd', 'dtq')
    assert app.main(args) == 0
    with sqlite3.connect(pipeline.path) as db:
        db.execute("DELETE FROM results WHERE model='dtq'")
        db.execute("DELETE FROM result_index WHERE model='dtq'")
    pipeline.cues = [replace(c, end=c.end-.1) for c in pipeline.cues]
    events = []
    inspect, close = RunInputs.inspect, RunInputs.close
    def observed(self, runtime, paths, identity):
        assert not events or events[-1] != 'closed'
        events.append(identity['source'])
        return inspect(self, runtime, paths, identity)
    def closed(self):
        events.append('closed')
        close(self)
    monkeypatch.setattr(RunInputs, 'inspect', observed)
    monkeypatch.setattr(RunInputs, 'close', closed)
    assert app.main(args) == 0
    assert events == ['dtq', 'wd', 'closed']


@pytest.mark.parametrize('corrupt', [False, True])
def test_metadata_backfill_refreshes_batch_and_unpacks_once(tmp_path, monkeypatch, corrupt):
    with StateStore(tmp_path / 'state.sqlite3', 'scope') as store:
        store.put('1', 'wd', outputs())
        with store.connection:
            if corrupt:
                store.connection.execute("UPDATE result_metadata SET data='bad' WHERE kind='scene'")
            else:
                store.connection.execute("DELETE FROM result_metadata WHERE kind='scene'")
        unpack = store.unpack
        calls = []
        def counted(data):
            calls.append(True)
            return unpack(data)
        monkeypatch.setattr(store, 'unpack', counted)
        with store.metadata.batch([('1', 'wd')]):
            first = store.result_metadata('1', 'scene', 'wd')
            first['meta']['scene_threshold'] = .99
            second = store.result_metadata('1', 'scene', 'wd')
        assert len(calls) == 1
        assert second['meta']['scene_threshold'] != .99


def test_source_preload_limits_kinds_scenes_and_caches_missing_keys(tmp_path):
    with StateStore(tmp_path / 'state.sqlite3', 'scope', cache_bytes=2**20) as store:
        store.put('1', 'wd', outputs())
        store.put('2', 'wd', outputs())
        store.cache.clear()
        store.preload_sources(['1', 'missing'], kinds=('markers',))
        assert store.cached(('sources', '1', 'markers')) == {'wd'}
        assert store.cached(('sources', 'missing', 'markers')) == set()
        assert store.cached(('sources', '1', 'scene')) is None
        assert store.cached(('sources', '2', 'markers')) is None


def test_download_bytes_are_hashed_once_for_four_verified_model_passes(monkeypatch):
    from types import SimpleNamespace
    from stash_ai.inference import inputs as inputs_module
    from stash_ai.runtime.cache import RamCache
    settings = SimpleNamespace(STASH_URL='http://stash.test', MAX_FRAMES_PER_SCENE=0)
    paths = {'vtt': '/vtt', 'sprite': '/sprite'}
    vtt = b'WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n/sprite#xywh=0,0,1,1\n'
    data = {'http://stash.test/vtt': vtt, 'http://stash.test/sprite': b'sprite payload'}
    counts = Counter()
    sha = inputs_module.hashlib.sha256
    def counted(value=b'', *args, **kwargs):
        if value in data.values():
            counts[value] += 1
        return sha(value, *args, **kwargs)
    monkeypatch.setattr(inputs_module.hashlib, 'sha256', counted)
    with RunInputs(SimpleNamespace(get_media=data.__getitem__), RamCache(100000)) as inputs:
        for model in range(4):
            result = inputs.inspect(settings, paths, model)
            assert result.media.get_media('http://stash.test/sprite') == b'sprite payload'
    assert counts == {vtt: 1, b'sprite payload': 1}
