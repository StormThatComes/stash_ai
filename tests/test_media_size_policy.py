"""Native-size Stash media and quiet Pillow policy shared by both workflows."""
from io import BytesIO
from pathlib import Path
import struct
from types import SimpleNamespace
import warnings
import zlib

from PIL import Image
import pytest

from stash_ai.domain import Cue
from stash_ai.inference.image_io import open_image
from stash_ai.inference.sprites import iter_sprite_frames
from stash_ai.recognition.config import load
from stash_ai.recognition.media import SpriteInput
from stash_ai.vtt import format_time
from tests.helpers.recognition import MediaClient, Response


RETIRED = {'max_image_pixels', 'max_media_bytes', 'max_vtt_bytes', 'max_sheets'}


@pytest.fixture(autouse=True)
def pillow_default(monkeypatch):
    # Exercise the transition from Pillow defaults independently of test order.
    monkeypatch.setattr(Image, 'MAX_IMAGE_PIXELS', 89478485)


def png_header(width, height):
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', b'') + chunk(b'IEND', b''))


def test_defaults_and_template_have_no_size_limits(tmp_path):
    baseline = load(root=tmp_path)
    bundled = load(Path(__file__).resolve().parents[1] / 'facial_recognition.yml')
    for config in (baseline, bundled):
        assert not RETIRED.intersection(config.data['execution'])
    assert all(key + ':' not in bundled.snapshot()['source'] for key in RETIRED)


@pytest.mark.parametrize('key', [*sorted(RETIRED), 'max_vtt_bytez'])
def test_removed_and_unknown_keys_fail_before_creating_state(tmp_path, key):
    path = tmp_path / 'old.yml'
    path.write_text(f'version: 1\nexecution: {{{key}: 1}}\n')
    with pytest.raises(ValueError, match='unknown keys'):
        load(path)
    assert not (tmp_path / 'faces_state.sqlite3').exists()


def test_header_above_previous_and_pillow_thresholds_opens_without_loading_or_warning(monkeypatch):
    from PIL.PngImagePlugin import PngImageFile
    data = png_header(20000, 20000)  # 400M pixels; only the header is allocated.
    with pytest.raises(Image.DecompressionBombError):
        Image.open(BytesIO(data))
    monkeypatch.setattr(PngImageFile, 'load', lambda self: pytest.fail('header open decoded pixels'))
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter('always')
        with open_image(BytesIO(data)) as image:
            assert image.size == (20000, 20000)
        assert Image.MAX_IMAGE_PIXELS is None
    assert not emitted


def test_tagging_recognition_and_later_crop_are_quiet_and_keep_native_pixels(tmp_path, monkeypatch):
    output = BytesIO()
    with Image.new('RGB', (640, 320), 'red') as image:
        image.save(output, format='PNG')
    data = {'http://stash/sprite': output.getvalue(), 'http://stash/vtt':
            b'WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nsprite.png#xywh=0,0,640,320\n'}
    client = SimpleNamespace(url='http://stash', get_media=data.__getitem__,
                             media=SimpleNamespace(get=lambda url, **kw: Response(data[url])))
    monkeypatch.setattr(Image, 'MAX_IMAGE_PIXELS', 1024)
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter('always')
        for cue, frame in iter_sprite_frames(client, {}, [Cue(0, 1, 'http://stash/sprite', (0, 0, 640, 320))]):
            assert frame.size == (640, 320) and frame.getpixel((0, 0)) == (255, 0, 0)
        monkeypatch.setattr(Image, 'MAX_IMAGE_PIXELS', 1024)
        with SpriteInput(client, {'paths': {'vtt': '/vtt', 'sprite': '/sprite'}}, load(root=tmp_path)) as source:
            for index, cue, frame in source.frames():
                assert frame.size == (640, 320) and frame.getpixel((0, 0)) == (255, 0, 0)
            with source.thumbnail(0) as thumbnail:
                assert thumbnail.size == (640, 320) and thumbnail.getpixel((0, 0)) == (255, 0, 0)
        warnings.warn('unrelated warning remains visible', UserWarning)
    assert len(emitted) == 1 and emitted[0].category is UserWarning


@pytest.mark.parametrize('selected', [0, 2])
def test_vtt_above_one_mib_and_ten_thousand_cues_keeps_optional_sampling(tmp_path, selected):
    client = MediaClient()
    cues = '\n\n'.join(f'{format_time(i)} --> {format_time(i+1)}\nsprite.jpg#xywh=0,0,160,160'
                       for i in range(10001))
    client.data['http://stash/vtt/1.vtt'] = ('WEBVTT\n\nNOTE ' + 'x' * 1048576 + '\n\n' + cues).encode()
    path = tmp_path / 'config.yml'
    path.write_text(f'version: 1\nexecution: {{max_cues: {selected}}}\n')
    with SpriteInput(client, client.scene, load(path)) as source:
        assert source.coverage == {'total_cues': 10001, 'selected_cues': selected or 10001,
                                   'complete_sampling': selected == 0}
        assert source.cues[0].start == 0 and source.cues[-1].start == 10000


def test_sprite_above_sixteen_mib_downloads_and_decodes(tmp_path):
    client = MediaClient()
    original = client.data['http://stash/sprite/1.jpg']
    client.data['http://stash/sprite/1.jpg'] = original + b'\0' * (16777217 - len(original))
    with SpriteInput(client, client.scene, load(root=tmp_path)) as source:
        assert len(source.cache[source.cues[0].image]) == 16777217
        assert len(list(source.frames())) == 3


def test_invalid_images_still_fail_without_suppressing_general_validation():
    with pytest.raises(Image.UnidentifiedImageError):
        open_image(BytesIO(b'not an image'))


def test_more_than_256_sheets_are_verified_with_bounded_cache_and_correct_resolution(tmp_path):
    output = BytesIO()
    with Image.new('RGB', (1, 1), 'red') as image:
        image.save(output, format='PNG')
    sprite = output.getvalue()
    text = 'WEBVTT\n\n' + '\n\n'.join(
        f'{format_time(i)} --> {format_time(i+1)}\nsheet{i}.png#xywh=0,0,1,1' for i in range(257))
    calls = []
    def get(url, **kwargs):
        calls.append(url)
        if url == 'http://stash/vtt/list.vtt':
            return Response(text.encode())
        assert url.startswith('http://stash/vtt/sheet') and url.endswith('.png')
        return Response(sprite)
    client = SimpleNamespace(url='http://stash', media=SimpleNamespace(get=get))
    # A single-sheet override must not replace these distinct VTT sources.
    scene = {'paths': {'vtt': '/vtt/list.vtt', 'sprite': '/unused.jpg'}}
    with SpriteInput(client, scene, load(root=tmp_path)) as source:
        assert len(source.hashes) == 257 and len(source.manifest['sheets']) == 257
        assert len(source.cache) == 2 and len(calls) == 258
        assert source.cues[0].image == 'http://stash/vtt/sheet0.png'
        assert source.cues[-1].image == 'http://stash/vtt/sheet256.png'
        count = 0
        for index, cue, image in source.frames():
            assert image.getpixel((0, 0)) == (255, 0, 0)
            count += 1
        assert count == 257 and not source.cache
