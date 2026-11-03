"""Paired portrait pixels follow the associated pose, including cache/fallbacks."""
from io import BytesIO
import hashlib
import json
import uuid

import numpy as np
from PIL import Image
import pytest

from stash_ai.recognition.database import encoded
from stash_ai.recognition.media import SpriteInput
from stash_ai.recognition.models import checked_detection
from stash_ai.recognition.person_signal import contract, save
from stash_ai.recognition.portraits import fill_portrait
from stash_ai.recognition.quality import portrait_bytes
from tests.helpers.recognition import Visibility, textured_image
from tests.test_recognition_portrait_precedence import evidence, choose, forbidden


SMALL_FACE = checked_detection([50, 15, 32, 32, 58, 26, 74, 26, 66, 33, 60, 40, 72, 40, .99])
PERSON = dict(bounds=[20, 5, 120, 155],
              keypoints=[[66, 33], [58, 26], [74, 26], [52, 28], [80, 28]] + [[-1, -1]]*12)


def paired(evidence, geometry=PERSON, *, retained=True, index=0):
    state, config, _, _, ids = evidence
    blob = None
    if retained:
        with textured_image() as image, image.crop(geometry['bounds']) as crop:
            output = BytesIO()
            crop.save(output, 'PNG')
            blob = output.getvalue()
    save(state.connection, ids[index], contract(config), {'person': geometry}, np.ones(4096), blob)
    state.connection.execute('UPDATE observations SET geometry=? WHERE id=?', (encoded(SMALL_FACE.row), ids[index]))
    state.connection.commit()


class NoPairedThumbnail(SpriteInput):
    def thumbnail(self, index):
        if index == 0:
            raise OSError('paired thumbnail unavailable')
        return super().thumbnail(index)


@pytest.mark.parametrize('prefetch', [False, True])
@pytest.mark.parametrize('ratio,expected', [(None, [20, 5, 120, 155]), (.8, [20, 5, 120, 130])])
def test_paired_frames_person_in_thumbnail_coordinates(evidence, prefetch, ratio, expected):
    paired(evidence)
    # Two candidates exercise the bounded JPEG worker as well as the serial path.
    paired(evidence, index=2)
    config = evidence[1]
    config.data['execution']['prefetch_portraits'] = prefetch
    config.data['portraits']['aspect_ratio'] = ratio
    original = [tuple(r) for r in evidence[0].connection.execute('SELECT * FROM members')]
    winner = choose(evidence)
    metadata = json.loads(winner['metadata'])
    assert metadata['selection_category'] == 'face+pose'
    assert metadata['crop_coordinate_system'] == 'thumbnail'
    assert metadata['crop'] == expected
    assert metadata['framing']['geometry'] == PERSON
    with Image.open(BytesIO(winner['image'])) as image:
        assert image.size == (expected[2]-expected[0], expected[3]-expected[1])
        assert image.height > SMALL_FACE.box[3]*2
    assert [tuple(r) for r in evidence[0].connection.execute('SELECT * FROM members')] == original


def test_pose_ratio_keeps_inverted_head(evidence):
    geometry = dict(PERSON, keypoints=[[66, 140], [58, 134], [74, 134], [52, 136], [80, 136]] + [[-1, -1]]*12)
    paired(evidence, geometry)
    evidence[1].data['portraits']['aspect_ratio'] = 1.
    metadata = json.loads(choose(evidence)['metadata'])
    assert metadata['crop'][1] > 5
    assert metadata['crop'][3] >= 150
    assert metadata['crop'][2]-metadata['crop'][0] == metadata['crop'][3]-metadata['crop'][1]


def test_retained_pose_pixels_match_live_pose_pixels(evidence):
    paired(evidence)
    state, config, _, _, _ = evidence
    config.data['portraits']['aspect_ratio'] = .8
    offline = choose(evidence, NoPairedThumbnail)
    metadata = json.loads(offline['metadata'])
    assert metadata['source_type'] == 'person_crop'
    assert metadata['selection_category'] == 'face+pose'
    assert metadata['crop'] == [0, 0, 100, 125]
    assert metadata['person_crop_bounds'] == PERSON['bounds']
    assert metadata['crop_coordinate_system'] == 'retained_person_crop'
    live = choose(evidence)
    assert live['image'] == offline['image']
    assert json.loads(live['metadata'])['source_type'] == 'vtt'


@pytest.mark.parametrize('bad_crop', [None, b'corrupt', 'wrong_size'])
def test_missing_or_invalid_person_crop_never_substitutes_retained_face(evidence, bad_crop):
    paired(evidence, retained=False)
    state, _, _, _, ids = evidence
    if bad_crop == 'wrong_size':
        with Image.new('RGB', (32, 32)) as image:
            output = BytesIO()
            image.save(output, 'PNG')
            bad_crop = output.getvalue()
    state.connection.execute('UPDATE person_evidence SET crop=? WHERE observation=?', (bad_crop, ids[0]))
    state.connection.commit()
    assert state.connection.execute('SELECT crop FROM observations WHERE id=?', (ids[0],)).fetchone()[0]
    metadata = json.loads(choose(evidence, NoPairedThumbnail)['metadata'])
    assert metadata['selection_category'] == 'face'
    assert metadata['observation'] != ids[0]


@pytest.mark.parametrize('cross_policy', [False, True])
def test_legacy_face_framing_is_not_reused_for_paired_portrait(evidence, cross_policy):
    paired(evidence)
    state, config, _, _, _ = evidence
    winner = choose(evidence)
    metadata = json.loads(winner['metadata'])
    with textured_image() as image:
        old_image, bounds = portrait_bytes(image, SMALL_FACE, config.data['portraits'], limit=1048576)
    metadata.pop('framing')
    metadata['crop'] = bounds
    if cross_policy:
        metadata['policy_sha'] = 'legacy-face-framing'
    state.connection.execute('UPDATE portraits SET id=?,metadata=?,image=?,digest=? WHERE id=?',
        (str(uuid.uuid4()) if cross_policy else winner['id'], encoded(metadata), old_image,
         hashlib.sha256(old_image).hexdigest(), winner['id']))
    state.connection.commit()
    current = choose(evidence)
    assert current['image'] != old_image
    assert json.loads(current['metadata'])['crop'] == PERSON['bounds']


def test_changed_pose_geometry_invalidates_same_id_and_historical_cache(evidence):
    paired(evidence)
    previous = choose(evidence)
    paired(evidence, dict(PERSON, bounds=[10, 0, 130, 160]))
    current = choose(evidence)
    assert current['id'] == previous['id']
    assert current['image'] != previous['image']
    assert json.loads(current['metadata'])['crop'] == [10, 0, 130, 160]


def test_unchanged_pose_cache_needs_no_source_pixels(evidence):
    paired(evidence)
    previous = choose(evidence)
    assert choose(evidence, forbidden)['image'] == previous['image']


def test_paired_encoding_respects_byte_budget_and_aspect(evidence):
    paired(evidence)
    config = evidence[1]
    config.data['portraits'].update(aspect_ratio=.8, jpeg_quality=100, margin=10.)
    config.data['storage']['max_crop_bytes'] = 1024
    winner = choose(evidence)
    metadata = json.loads(winner['metadata'])
    assert metadata['selection_category'] == 'face+pose'
    assert metadata['crop'] == [20, 5, 120, 130]  # Face margin never expands the pose.
    assert len(winner['image']) <= 1024
    with Image.open(BytesIO(winner['image'])) as image:
        assert image.width / image.height == .8
        assert image.width <= 100 and image.height <= 125


def test_explicit_face_kind_keeps_face_framing(evidence):
    paired(evidence)
    state, config, _, _, ids = evidence
    state.connection.execute('DELETE FROM members WHERE observation!=?', (ids[0],))
    state.connection.commit()
    config.data['portraits']['kind'] = 'face'
    metadata = json.loads(choose(evidence)['metadata'])
    assert metadata['selection_category'] == 'face'
    assert metadata['framing']['kind'] == 'face'
    assert metadata['crop'][3]-metadata['crop'][1] < 70


def test_owned_legacy_paired_portrait_updates_once(evidence):
    paired(evidence)
    state, config, client, target, _ = evidence
    def refresh():
        return fill_portrait(state, client, None, config, 'scope', target, {}, forbidden, forbidden,
                             visibility=Visibility(), expression=object())
    assert refresh()['portraits_filled'] == 1
    with textured_image() as image:
        old_image, bounds = portrait_bytes(image, SMALL_FACE, config.data['portraits'], limit=1048576)
    old_digest = hashlib.sha256(old_image).hexdigest()
    portrait = state.connection.execute("SELECT * FROM portraits WHERE status='published'").fetchone()
    metadata = json.loads(portrait['metadata'])
    metadata.pop('framing')
    metadata.update(crop=bounds, policy_sha='legacy-face-framing')
    state.connection.execute('UPDATE portraits SET image=?,digest=?,metadata=? WHERE id=?',
                             (old_image, old_digest, encoded(metadata), portrait['id']))
    state.connection.execute("UPDATE publications SET payload=json_set(payload,'$.sha256',?) WHERE kind='image'", (old_digest,))
    state.connection.commit()
    client.images['900'] = old_image
    assert refresh()['portraits_updated'] == 1
    assert client.images['900'] != old_image
    assert refresh()['portraits_preserved'] == 1


def test_invalid_pose_geometry_skips_to_lower_tier(evidence):
    state, _, _, _, _ = evidence
    state.connection.execute("UPDATE person_evidence SET metadata=json_set(metadata,'$.person.bounds',json('[20,5,10,155]'))")
    state.connection.commit()
    assert json.loads(choose(evidence)['metadata'])['selection_category'] == 'face'
