from io import BytesIO
import json

import numpy as np
from PIL import Image
import pytest
import yaml

from stash_ai.recognition.config import load
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.media import SpriteInput
from stash_ai.recognition.models import checked_detection
from stash_ai.recognition.run import Services
from tests.helpers.recognition import Backend, Response, Stash
from tests.test_recognition_run import run


FACE = checked_detection([180, 100, 100, 100, 205, 135, 255, 135, 230, 150, 211, 170, 249, 170, .99])


class ThumbnailStash(Stash):
    def __init__(self):
        super().__init__()
        pixels = np.random.default_rng(4).integers(30, 225, (350, 500, 3), dtype='uint8')
        with Image.fromarray(pixels) as image:
            output = BytesIO()
            image.save(output, format='PNG')
            self.sprite = output.getvalue()
        self.vtt = ('WEBVTT\n\n' + '\n\n'.join(
            f'00:00:{i:02d}.000 --> 00:00:{i+1:02d}.000\nsprite.jpg#xywh=0,0,500,350'
            for i in range(3))).encode()

    def get(self, url, **kwargs):
        if '/vtt/' in url or '/sprite/' in url:
            self.media_calls.append(url)
            return Response(self.vtt if '/vtt/' in url else self.sprite)
        return super().get(url, **kwargs)


def setup(tmp_path, monkeypatch, **portraits):
    monkeypatch.setenv('STASH_URL', 'http://stash/')
    path = tmp_path / 'facial_recognition.yml'
    write_config(path, **portraits)
    stash, backends = ThumbnailStash(), []
    def backend(*args):
        value = Backend()
        value.detect = lambda image: [FACE]
        backends.append(value)
        return value
    services = Services(client=lambda **kwargs: stash, assets=lambda *a, **k: {}, backend=backend,
                        decoder=lambda *args: pytest.fail('VTT portraits must not initialize FFmpeg/video'))
    return load(path), stash, services, backends


def write_config(path, **portraits):
    path.write_text(yaml.safe_dump({'version': 1, 'detection': {'min_sharpness': 0},
                                  'portraits': {'min_sharpness': 0, 'max_candidates': 2, **portraits}}))


def image_size(data):
    with Image.open(BytesIO(data)) as image:
        return image.size


def image_writes(stash):
    return [write for write in stash.writes if write[0] == 'metadata']


def test_default_vtt_portrait_is_wider_than_recognition_crop_without_extra_inference(tmp_path, monkeypatch):
    config, stash, services, backends = setup(tmp_path, monkeypatch)
    assert run(config, services, '1') == 0
    performer = stash.links['1'][-1]
    assert image_size(stash.images[performer]) == (170, 170)
    assert len(backends) == 1 and backends[0].calls == 3 and backends[0].closed
    assert all('/scene/' not in url and '/2.' not in url for url in stash.media_calls)
    assert stash.media_calls.count('http://stash/sprite/1.jpg') == 2  # one scan, one shared portrait source
    with FaceDatabase(config.database) as db:
        observations = db.connection.execute('SELECT * FROM observations').fetchall()
        assert all(image_size(row['crop']) == (130, 130) for row in observations)
        portrait = db.connection.execute("SELECT * FROM portraits WHERE status='published'").fetchone()
        metadata = json.loads(portrait['metadata'])
        observation = next(row for row in observations if row['id'] == metadata['observation'])
        assert metadata['source_type'] == 'vtt' and metadata['cue_seconds'] == observation['seconds']
        assert metadata['geometry'] == json.loads(observation['geometry'])
        assert metadata['crop'] == [145, 65, 315, 235]
        assert metadata['input_sha'] == db.connection.execute('SELECT input_sha FROM scans').fetchone()[0]
        assert 'actual_seconds' not in metadata
    assert run(config, services, '1') == 0
    assert len(image_writes(stash)) == 1 and len(backends) == 1


def test_yaml_framing_and_quality_refresh_owned_image_without_rescanning(tmp_path, monkeypatch):
    config, stash, services, backends = setup(tmp_path, monkeypatch)
    assert run(config, services, '1') == 0
    performer, initial = stash.links['1'][-1], dict(stash.images)
    write_config(config.source_path, margin=.5, aspect_ratio=.8, jpeg_quality=75)
    changed = load(config.source_path)
    assert run(changed, services, '1') == 0
    assert image_size(stash.images[performer]) == (200, 250)
    assert stash.images != initial and len(image_writes(stash)) == 2 and len(backends) == 1
    with FaceDatabase(config.database) as db:
        assert db.connection.execute('SELECT COUNT(*) FROM scans').fetchone()[0] == 1
        row = db.connection.execute("SELECT summary FROM runs ORDER BY started DESC LIMIT 1").fetchone()
        assert json.loads(row[0])['counts']['portraits_updated'] == 1
    assert run(changed, services, '1') == 0
    assert len(image_writes(stash)) == 2
    write_config(config.source_path, margin=.5, aspect_ratio=.8, jpeg_quality=60)
    assert run(load(config.source_path), services, '1') == 0
    assert len(image_writes(stash)) == 3 and len(backends) == 1


def test_manual_image_replacement_with_same_url_is_preserved_after_yaml_change(tmp_path, monkeypatch):
    config, stash, services, _ = setup(tmp_path, monkeypatch)
    assert run(config, services, '1') == 0
    performer = stash.links['1'][-1]
    stash.images[performer] = b'manual replacement at the same Stash URL'
    write_config(config.source_path, margin=.5)
    assert run(load(config.source_path), services, '1') == 0
    assert stash.images[performer] == b'manual replacement at the same Stash URL'
    assert len(image_writes(stash)) == 1


def test_manual_image_during_vtt_refresh_is_preserved(tmp_path, monkeypatch):
    config, stash, services, _ = setup(tmp_path, monkeypatch)
    assert run(config, services, '1') == 0
    performer = stash.links['1'][-1]
    class ConcurrentSource(SpriteInput):
        def thumbnail(self, index):
            stash.images[performer] = b'manual concurrent replacement'
            return super().thumbnail(index)
    services.source = ConcurrentSource
    write_config(config.source_path, margin=.5)
    assert run(load(config.source_path), services, '1') == 0
    assert stash.images[performer] == b'manual concurrent replacement'
    assert len(image_writes(stash)) == 1


@pytest.mark.parametrize('applied', [True, False])
def test_uncertain_portrait_refresh_is_reconciled_or_withheld_without_retry(tmp_path, monkeypatch, applied):
    config, stash, services, _ = setup(tmp_path, monkeypatch)
    assert run(config, services, '1') == 0
    original = stash.update_person
    calls = []
    def lost(*args, **kwargs):
        calls.append(1)
        if applied:
            original(*args, **kwargs)
        raise TimeoutError('lost portrait update response')
    stash.update_person = lost
    write_config(config.source_path, margin=.5)
    changed = load(config.source_path)
    assert run(changed, services, '1') == 1
    assert run(changed, services, '1') == (0 if applied else 1)
    assert len(calls) == 1
    with FaceDatabase(config.database) as db:
        assert db.connection.execute("SELECT status FROM publications WHERE kind='image'").fetchone()[0] == (
            'observed' if applied else 'unknown')


def test_changed_sprite_between_recognition_and_portraits_is_withheld(tmp_path, monkeypatch):
    config, stash, services, _ = setup(tmp_path, monkeypatch)
    class ChangedSource(SpriteInput):
        def __enter__(self):
            if stash.links['1'][-1] != '77':
                self.fingerprint = 'changed after recognition'
            return self
    services.source = ChangedSource
    assert run(config, services, '1') == 0
    assert len(stash.links['1']) == 2 and not stash.images
    with FaceDatabase(config.database) as db:
        assert all(json.loads(row[0])['reason'] == 'vtt_source_unavailable_or_changed'
                   for row in db.connection.execute('SELECT metadata FROM portraits'))


def test_stricter_yaml_quality_keeps_existing_photo_when_no_replacement_qualifies(tmp_path, monkeypatch):
    config, stash, services, _ = setup(tmp_path, monkeypatch)
    assert run(config, services, '1') == 0
    initial = dict(stash.images)
    write_config(config.source_path, min_face_pixels=200)
    assert run(load(config.source_path), services, '1') == 0
    assert stash.images == initial and len(image_writes(stash)) == 1


def test_selected_scene_vtt_shortlist_never_reads_historical_other_scene(tmp_path, monkeypatch):
    config, stash, services, _ = setup(tmp_path, monkeypatch, enabled=False)
    assert run(config, services, '1', '2') == 0
    stash.media_calls.clear()
    write_config(config.source_path)
    assert run(load(config.source_path), services, '1') == 0
    assert all('/2.' not in url for url in stash.media_calls)
    assert stash.images


def test_thumbnail_pixel_limits_and_changed_sheet_are_enforced(tmp_path, monkeypatch):
    config, stash, _, _ = setup(tmp_path, monkeypatch)
    with SpriteInput(stash, stash.scene('1'), config) as source:
        with source.thumbnail(0) as image:
            assert image.size == (500, 350)
        source.hashes[source.cues[0].image] = 'changed'
        with pytest.raises(ValueError, match='changed'):
            source.thumbnail(1)
    write_config(config.source_path)
    text = config.source_path.read_text() + 'execution: {max_image_pixels: 1024}\n'
    config.source_path.write_text(text)
    with SpriteInput(stash, stash.scene('1'), load(config.source_path)) as source:
        with pytest.raises(ValueError, match='pixel limit'):
            source.thumbnail(0)


def test_switching_portrait_sources_refreshes_the_same_owned_performer(tmp_path, monkeypatch):
    config, stash, services, backends = setup(tmp_path, monkeypatch)
    assert run(config, services, '1') == 0
    performer = stash.links['1'][-1]
    decoders = []
    class Decoder:
        url = 'http://stash/scene/1/stream'

        def __init__(self, *args):
            self.closed = False
            decoders.append(self)

        def __enter__(self):
            return self

        def frame(self, seconds):
            with Image.open(BytesIO(stash.sprite)) as image:
                return image.convert('RGB'), {'requested_seconds': seconds, 'actual_seconds': seconds}

        def close(self):
            self.closed = True
    services.decoder = Decoder
    write_config(config.source_path, source='video', margin=.15, time_offsets=[0])
    assert run(load(config.source_path), services, '1') == 0
    assert image_size(stash.images[performer]) == (130, 130)
    assert len(backends) == 2 and all(decoder.closed for decoder in decoders)
    write_config(config.source_path)
    assert run(load(config.source_path), services, '1') == 0
    assert image_size(stash.images[performer]) == (170, 170)
    assert len(backends) == 2 and len(image_writes(stash)) == 3
    assert sum(write[0] == 'create' for write in stash.writes) == 1


def test_vtt_identity_ambiguity_is_withheld_without_reading_portrait_source(tmp_path, monkeypatch):
    import stash_ai.recognition.vtt_portraits as vtt
    from stash_ai.recognition.clustering import Group
    config, stash, services, _ = setup(tmp_path, monkeypatch, enabled=False)
    assert run(config, services, '1') == 0
    original = vtt._groups
    def ambiguous(*args):
        groups = original(*args)
        return groups + [Group('competing-person', groups[0].samples, True)]
    monkeypatch.setattr(vtt, '_groups', ambiguous)
    write_config(config.source_path)
    assert run(load(config.source_path), services, '1') == 0
    assert not stash.images
    assert stash.media_calls.count('http://stash/sprite/1.jpg') == 2  # scans only
    with FaceDatabase(config.database) as db:
        assert all(json.loads(row[0])['reason'] == 'portrait_identity_ambiguous'
                   for row in db.connection.execute('SELECT metadata FROM portraits'))


def test_default_framing_at_thumbnail_edge_keeps_face_and_never_upscales(tmp_path, monkeypatch):
    from stash_ai.recognition.quality import portrait_bytes
    config, stash, _, _ = setup(tmp_path, monkeypatch)
    edge = checked_detection([0, 0, 100, 100, 25, 35, 75, 35, 50, 50, 31, 70, 69, 70, .99])
    with Image.open(BytesIO(stash.sprite)) as image:
        data, bounds = portrait_bytes(image, edge, config.data['portraits'], limit=1048576)
        assert bounds == [0, 0, 170, 170] and image_size(data) == (170, 170)
