from types import SimpleNamespace

import pytest

from stash_ai.inference.sprites import fetch_cues
from stash_ai.recognition.config import load
from stash_ai.recognition.media import SpriteInput
from stash_ai.vtt import parse_vtt_cues
from tests.helpers.recognition import Response


def vtt(image):
    return f'WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n{image}#xywh=0,0,32,32\n'


@pytest.mark.parametrize('image', ['', ' ', '\t'])
@pytest.mark.parametrize('override', [False, True])
def test_blank_image_rejected_before_any_sheet_download(tmp_path, image, override):
    paths = {'vtt': '/vtt'} | ({'sprite': '/sprite'} if override else {})
    calls = []
    def get_media(url):
        calls.append(url)
        assert url == 'http://stash/vtt'
        return vtt(image).encode()
    client = SimpleNamespace(url='http://stash', get_media=get_media,
                             media=SimpleNamespace(get=lambda url, **kw: Response(get_media(url))))
    with pytest.raises(ValueError, match='nonempty sprite image'):
        parse_vtt_cues(vtt(image))
    with pytest.raises(ValueError, match='nonempty sprite image'):
        fetch_cues(SimpleNamespace(STASH_URL=client.url, MAX_FRAMES_PER_SCENE=0), client, paths)
    with pytest.raises(ValueError, match='nonempty sprite image'):
        SpriteInput(client, {'paths': paths}, load(root=tmp_path))
    assert calls == ['http://stash/vtt', 'http://stash/vtt']


@pytest.mark.parametrize('multiple', [False, True])
@pytest.mark.parametrize('override', [False, True])
def test_normal_single_and_multiple_sheet_references_keep_resolution(multiple, override):
    text = vtt('sprite1.jpg')
    if multiple:
        text += '\n00:00:01.000 --> 00:00:02.000\nsprite2.jpg#xywh=pixel:32,0,32,32\n'
    paths = {'vtt': '/media/thumbs.vtt'} | ({'sprite': '/sprite'} if override else {})
    client = SimpleNamespace(get_media=lambda url: text.encode())
    cues, total = fetch_cues(SimpleNamespace(STASH_URL='http://stash', MAX_FRAMES_PER_SCENE=0), client, paths)
    assert total == (2 if multiple else 1)
    assert cues[0].image == ('http://stash/sprite' if override and not multiple else 'http://stash/media/sprite1.jpg')
    if multiple:
        assert cues[1].image == 'http://stash/media/sprite2.jpg' and cues[1].start == 1
