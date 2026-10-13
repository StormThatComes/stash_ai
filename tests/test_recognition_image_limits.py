from io import BytesIO
from pathlib import Path
import struct
import zlib

from PIL import Image
import pytest

from stash_ai.recognition.config import load
from stash_ai.recognition.images import open_image


def png_header(width, height):
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', b'') + chunk(b'IEND', b''))


def test_shipped_and_missing_yaml_defaults_allow_requested_pixels_and_keep_media_bytes(tmp_path):
    for config in (load(root=tmp_path), load(Path(__file__).resolve().parents[1] / 'facial_recognition.yml')):
        assert config.data['execution']['max_image_pixels'] == 196000000
        assert config.data['execution']['max_media_bytes'] == 16777216
    path = tmp_path / 'custom.yml'
    path.write_text('version: 1\nexecution: {max_image_pixels: 196000000, max_media_bytes: 16777216}\n')
    assert load(path).data['execution']['max_image_pixels'] == 196000000
    path.write_text('version: 1\nexecution: {max_image_pixels: 196000001}\n')
    with pytest.raises(ValueError, match='max_image_pixels'):
        load(path)


def test_196_million_pixel_header_opens_without_loading_or_changing_pillow_default(monkeypatch):
    original = Image.MAX_IMAGE_PIXELS
    data = png_header(14000, 14000)
    with pytest.raises(Image.DecompressionBombError):
        Image.open(BytesIO(data))
    monkeypatch.setattr(Image.Image, 'load', lambda self: pytest.fail('header open decoded pixels'))
    with open_image(BytesIO(data), 196000000) as image:
        assert image.size == (14000, 14000)
        assert Image.MAX_IMAGE_PIXELS == original
    assert Image.MAX_IMAGE_PIXELS == original
    with pytest.raises(Image.DecompressionBombError):
        Image.open(BytesIO(data))


@pytest.mark.parametrize('width,height,limit', [(14001, 14000, 196000000), (200, 200, 1024)])
def test_over_limit_headers_are_rejected_before_loading_and_restore_default(width, height, limit, monkeypatch):
    original = Image.MAX_IMAGE_PIXELS
    monkeypatch.setattr(Image.Image, 'load', lambda self: pytest.fail('oversized header decoded pixels'))
    with pytest.raises(ValueError, match='pixel limit'):
        open_image(BytesIO(png_header(width, height)), limit)
    assert Image.MAX_IMAGE_PIXELS == original


def test_invalid_image_restores_pillow_default_and_valid_decode_still_works():
    original = Image.MAX_IMAGE_PIXELS
    with pytest.raises(Image.UnidentifiedImageError):
        open_image(BytesIO(b'not an image'), 196000000)
    assert Image.MAX_IMAGE_PIXELS == original
    output = BytesIO()
    with Image.new('RGB', (32, 32), 'red') as image:
        image.save(output, format='PNG')
    with open_image(BytesIO(output.getvalue()), 1024) as image:
        with image.convert('RGB') as decoded:
            assert decoded.getpixel((0, 0)) == (255, 0, 0)
    assert Image.MAX_IMAGE_PIXELS == original
