"""Configured portrait ratios are output constraints, including body winners."""
from io import BytesIO
import hashlib
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from stash_ai.recognition.config import defaults
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.models import checked_detection
from stash_ai.recognition.portrait_availability import fit_cached_bytes
from stash_ai.recognition.portrait_framing import person_bounds, aspect_size, matches_aspect
from stash_ai.recognition.quality import portrait_bytes, encode_portrait
from tests.test_recognition_body_integration import seeded, crop


@pytest.mark.parametrize('ratio', [.1, .8, 1., 16/9, 10.])
@pytest.mark.parametrize('position', [0, 20, 40])
def test_face_ratio_survives_source_edges_and_oversized_face(ratio, position):
    policy = dict(defaults()['portraits'], aspect_ratio=ratio)
    face = checked_detection([position, position, 60, 60, 20, 20, 50, 20, 35, 35, 25, 50, 45, 50, .99])
    with Image.new('RGB', (100, 100)) as image:
        data, bounds = portrait_bytes(image, face, policy, limit=1048576)
    with Image.open(BytesIO(data)) as portrait:
        assert portrait.width / portrait.height == pytest.approx(ratio)
        assert portrait.size == (bounds[2]-bounds[0], bounds[3]-bounds[1])
    assert 0 <= bounds[0] < bounds[2] <= 100
    assert 0 <= bounds[1] < bounds[3] <= 100


@pytest.mark.parametrize('ratio', [.1, .8, 1., 16/9, 10.])
def test_byte_budget_preserves_ratio(ratio):
    policy = dict(defaults()['portraits'], aspect_ratio=ratio, jpeg_quality=100)
    pixels = np.random.default_rng(23).integers(0, 256, (500, 500, 3), dtype='uint8')
    face = checked_detection([50, 50, 400, 400, 150, 150, 350, 150, 250, 250, 170, 350, 330, 350, .99])
    details = {}
    with Image.fromarray(pixels) as image:
        data, bounds = portrait_bytes(image, face, policy, limit=1024, details=details)
    assert len(data) <= 1024 and details['downscaled']
    with Image.open(BytesIO(data)) as portrait:
        assert portrait.width / portrait.height == pytest.approx(ratio)
        assert portrait.size == tuple(details['pixels'])
        assert portrait.width <= bounds[2]-bounds[0] and portrait.height <= bounds[3]-bounds[1]


@pytest.mark.parametrize('size,head', [
    ((120, 240), (60, 20)),       # upright
    ((120, 240), (60, 210)),      # inverted / bent down
    ((240, 120), (210, 60)),      # lying on the right
    ((240, 120), (20, 60)),       # lying on the left
])
def test_body_crop_keeps_measured_head_with_clearance(size, head):
    x, y = head
    # Nonzero source origin makes accidental use of absolute points visible.
    geometry = dict(bounds=[300, 500, 300+size[0], 500+size[1]],
                    keypoints=[[300+x+dx, 500+y+dy] for dx, dy in [(0, 0), (-5, -3), (5, -3), (-10, 0), (10, 0)]]
                    + [[-1, -1]]*12)
    left, top, right, bottom = person_bounds(size, geometry, .8)
    assert (right-left)/(bottom-top) == .8
    assert left <= x-19 and right >= x+19
    assert top <= y-12 and bottom >= y+9
    assert 0 <= left < right <= size[0] and 0 <= top < bottom <= size[1]


@pytest.mark.parametrize('points', [[], [[-1, -1]]*17, [[float('nan'), 5]]*17, [[999, 999]]*17])
def test_body_without_usable_head_points_keeps_top(points):
    geometry = dict(bounds=[0, 0, 120, 240], keypoints=points)
    assert person_bounds((120, 240), geometry, .8) == [0, 0, 120, 150]
    assert person_bounds((120, 240), geometry, None) == [0, 0, 120, 240]


def test_head_features_take_priority_when_clearance_cannot_fit():
    geometry = dict(bounds=[0, 0, 120, 100], keypoints=[[60, 50], [110, 50]] + [[-1, -1]]*15)
    left, top, right, bottom = person_bounds((120, 100), geometry, .5)
    assert left <= 60 and right >= 110
    assert (right-left)/(bottom-top) == .5


@pytest.mark.parametrize('kind', ['auto', 'person'])
@pytest.mark.parametrize('ratio,expected', [(None, (120, 240)), (.8, (120, 150)), (1., (120, 120)), (2., (120, 60))])
def test_body_winner_uses_ratio_and_retained_coordinates(tmp_path, monkeypatch, kind, ratio, expected):
    from stash_ai.recognition.body_portraits import choose_kind
    from stash_ai.recognition.portrait_refresh import retained_targets
    c, selected, ids, people = seeded(tmp_path, vectors=[[1, 0]])
    c.data['portraits'].update(kind=kind, aspect_ratio=ratio)
    monkeypatch.setattr('stash_ai.recognition.portraits.choose', lambda *a, **k: None)
    with FaceDatabase(c.database) as state:
        crop(state.connection, ids[0])
        original = list(state.connection.execute('SELECT id,crop,geometry FROM body_observations'))
        target = retained_targets(state, c, 'scope')[people[0]]
        winner = choose_kind(state, None, c, 'scope', target, {}, None, None, None)
        metadata = json.loads(winner['metadata'])
        with Image.open(BytesIO(winner['image'])) as image:
            assert image.size == expected
        assert metadata['portrait_kind'] == 'person'
        assert metadata['crop'] == [0, 0, *expected]
        assert metadata['crop_coordinate_system'] == 'retained_person_crop'
        assert list(state.connection.execute('SELECT id,crop,geometry FROM body_observations')) == original


def test_cached_wrong_ratio_is_rejected_even_under_the_byte_budget():
    config = SimpleNamespace(data=defaults())
    config.data['portraits']['aspect_ratio'] = .8
    with Image.new('RGB', (120, 240)) as image:
        data = encode_portrait(image, {'jpeg_quality': 90}, limit=1048576)
    with pytest.raises(ValueError, match='aspect ratio'):
        fit_cached_bytes(data, hashlib.sha256(data).hexdigest(), config, {})


@pytest.mark.parametrize('ratio', [.1, .8, 1., 16/9, 10., 3.617179932252463])
@pytest.mark.parametrize('size', [(1, 1), (3, 4), (101, 99), (334, 759)])
def test_integer_ratio_rounding_is_accepted_by_cache(size, ratio):
    framed = aspect_size(*size, ratio)
    assert 1 <= framed[0] <= size[0] and 1 <= framed[1] <= size[1]
    assert matches_aspect(framed, ratio)


def test_body_owned_portrait_reframes_once_after_config_change(tmp_path):
    from stash_ai.recognition.portraits import fill_portrait
    from stash_ai.recognition.portrait_refresh import retained_targets
    from tests.helpers.recognition import Stash
    c, selected, ids, people = seeded(tmp_path, vectors=[[1, 0]])
    c.data['portraits']['kind'] = 'person'
    client = Stash()
    with FaceDatabase(c.database) as state:
        crop(state.connection, ids[0])
        person = people[0]
        token = state.connection.execute('SELECT token FROM people WHERE id=?', (person,)).fetchone()[0]
        client.profiles['900'] = dict(id='900', name='Body', details=token, image_path='')
        target = retained_targets(state, c, 'scope')[person]
        def fill():
            return fill_portrait(state, client, None, c, 'scope', target, {}, None, None)
        assert fill()['portraits_filled'] == 1
        c.data['portraits']['aspect_ratio'] = .8
        assert fill()['portraits_updated'] == 1
        with Image.open(BytesIO(client.images['900'])) as image:
            assert image.size == (120, 150)
        writes = list(client.writes)
        assert fill()['portraits_preserved'] == 1
        assert client.writes == writes
        client.images['900'] = b'manual replacement'
        c.data['portraits']['aspect_ratio'] = 1.
        assert fill()['portraits_preserved'] == 1
        assert client.images['900'] == b'manual replacement' and client.writes == writes


def test_legacy_vtt_jpeg_with_requested_but_unmet_ratio_is_rebuilt(tmp_path, monkeypatch):
    from tests.test_recognition_vtt_portraits import setup, image_size
    from tests.test_recognition_portrait_removal import change
    from tests.test_recognition_run import run
    config, client, services, _ = setup(tmp_path, monkeypatch)
    assert run(config, services, '1') == 0
    performer = client.links['1'][-1]
    with FaceDatabase(config.database) as state:
        # Old framing silently ignored extreme ratios while retaining them in
        # metadata. Simulate that cache under an earlier policy implementation.
        state.connection.execute("UPDATE portraits SET metadata=json_set(metadata,"
                                 "'$.policy.aspect_ratio',0.1,'$.policy_sha','legacy')")
        state.connection.commit()
    config = change(config, aspect_ratio=.1)
    assert run(config, services, '1') == 0
    width, height = image_size(client.images[performer])
    assert width/height == .1
    writes = list(client.writes)
    assert run(config, services, '1') == 0
    assert client.writes == writes
