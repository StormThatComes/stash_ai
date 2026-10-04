"""Input signatures and bounded, verified media sharing across model passes."""
from collections import Counter
from types import SimpleNamespace

import pytest

from tests.helpers.runtime import runtime as app
from stash_ai.inference import sprites
from stash_ai.inference.inputs import VerifiedMedia, RunInputs, inspect_input
from stash_ai.runtime.cache import RamCache


pytestmark = pytest.mark.usefixtures("reset_progress")


def test_input_signature_matches_existing_cache_format():
    settings = SimpleNamespace(STASH_URL='http://stash.test/root/', MAX_FRAMES_PER_SCENE=0)
    paths = {'vtt': '/vtt', 'sprite': '/sprite'}
    data = {
        'http://stash.test/vtt': (
            b'WEBVTT\n\n00:00:01.250 --> 00:00:03.500\nsprite.jpg#xywh=1,2,3,4\n'),
        'http://stash.test/sprite': b'sprite bytes',
    }
    identity = {'source': 'wd', 'labels': ['żółć', '猫'], 'threshold': .4}
    stash = SimpleNamespace(get_media=data.__getitem__)
    # Captured from progress.inspect_input before extracting the input module.
    expected = 'd2af957c8bde207ad87cf8953ad2db20fffdea34f0f61cfba8ef1c9ed58ad501'
    assert inspect_input(settings, stash, paths, identity)[2] == expected
    inputs = RunInputs(stash, RamCache(1000000))
    assert inputs.inspect(settings, paths, identity)[2] == expected
    assert inputs.inspect(settings, paths, identity)[2] == expected


def test_verified_media_ram_limit_and_input_consistency():
    stash=SimpleNamespace(get_media=lambda url:(url*8).encode())
    media=VerifiedMedia(stash,limit=16)
    for url in ["a","b","c","d"]:media.get_media(url)
    assert media.size<=16 and len(media.cache)<=2
    stash.get_media=lambda url:b"changed"
    with pytest.raises(RuntimeError,match="changed"):media.get_media("a")


def test_media_is_fetched_once_for_four_models_with_identical_signatures(pipeline, monkeypatch):
    p = pipeline
    calls = Counter()
    def get_media(url):
        calls[url] += 1
        return url.encode()
    monkeypatch.setattr(p.stash, 'get_media', get_media)
    def cues(app, media, paths):
        media.get_media(paths['vtt'])
        return p.cues, len(p.cues)
    monkeypatch.setattr(sprites, 'fetch_cues', cues)
    paths = p.stash.scene['paths']
    reference = [inspect_input(app, p.stash, paths, model)[2] for model in range(4)]
    calls.clear()
    inputs = RunInputs(p.stash, RamCache(1000000))
    for model in range(4):
        out, media, signature = inputs.inspect(app, paths, model)
        assert out == p.cues and signature == reference[model]
        assert media.get_media('sprite') == b'sprite'
    assert calls == {'/vtt': 1, 'sprite': 1}


def test_media_eviction_revalidates_and_new_run_accepts_new_bytes(pipeline, monkeypatch):
    p = pipeline
    cache = RamCache(1000000)
    inputs = RunInputs(p.stash, cache)
    paths = p.stash.scene['paths']
    original = inputs.inspect(app, paths, 'model')[2]
    cache.clear()
    monkeypatch.setattr(p.stash, 'get_media', lambda url: b'new input')
    with pytest.raises(RuntimeError, match='media changed'):
        inputs.inspect(app, paths, 'model')
    assert RunInputs(p.stash, cache).inspect(app, paths, 'model')[2] != original


