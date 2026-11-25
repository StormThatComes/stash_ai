import json

import numpy as np
import pytest

from stash_ai.recognition import inspection_face_visibility as inspector
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.inspection_crops import VttPreviews
from stash_ai.recognition.visibility import CONTRACT_ID, FaceVisibility
from tests.helpers.recognition import FACE
from tests.test_recognition_identity_inspector import vtt_fixture


@pytest.mark.parametrize('roll,visible,unmarked', [
    (0, (60, 80), (100, 80)), (45, (65, 65), (95, 95)),
    (90, (80, 60), (80, 100)), (180, (100, 80), (60, 80)),
])
def test_replay_uses_selected_geometry_and_the_recognition_mask_on_verified_pixels(
        tmp_path, monkeypatch, roll, visible, unmarked):
    import torch
    from stash_ai.recognition.models import checked_detection

    config, reader, client, _, face, backend, _ = vtt_fixture(tmp_path)
    angle = np.deg2rad(roll)
    c, s = np.cos(angle), np.sin(angle)
    points = (np.array(FACE.landmarks) - 80) @ [[c, s], [-s, c]] + 80
    geometry = checked_detection([*FACE.box, *points.ravel(), FACE.score])
    with FaceDatabase(config.database) as db:
        db.connection.execute('UPDATE observations SET geometry=?,rejected=?,'
                              "quality=json_set(quality,'$.visible_face_fraction',0.23,'$.visibility.contract',?) "
                              'WHERE id=?', (json.dumps(geometry.row), 'face_occluded', CONTRACT_ID, face))
        db.connection.commit()
    before = config.database.read_bytes()

    model = object.__new__(FaceVisibility)
    model.logit_threshold = 0.0
    model.device = 'cpu'
    model.minimum_source_fraction = config.data['detection']['min_face_source_fraction']
    def split(inputs):
        values = torch.ones_like(inputs[:, :1])
        values[..., 128:] = 0  # Threshold is strictly positive.
        return values
    model.model = split
    monkeypatch.setattr(inspector, 'FaceVisibility', lambda *_: model)
    def assets(config, *, keys):
        assert keys == ['faceocc']
        return {'faceocc': None}
    monkeypatch.setattr(inspector, 'ensure_assets', assets)
    previews = VttPreviews(config, client_factory=lambda: client)
    try:
        replay = inspector.replay(previews, face, reader.scan['scope'])
        original = replay.image
        overlay, caption = replay.render(0.0)
        assert model.model is None  # The inspector released its model.
        model.model = split
        expected = model.measure(original, geometry)['visibility']
        assert f"({expected['visible_pixels']:,} visible / {expected['face_pixels']:,} ellipse pixels)" in caption
        assert 'Recorded 0.230 at logit 0 · Replay 0.500' in caption and 'Source coverage 100.0%' in caption
        assert original.size == overlay.size == (160, 160)
        assert overlay.getpixel(visible) != original.getpixel(visible)
        assert overlay.getpixel(unmarked) == original.getpixel(unmarked)
        assert overlay.getpixel((5, 5)) == original.getpixel((5, 5))
        # The outline follows the measured face, including oblique rolls;
        # its top is no longer fixed at the old vertical box ellipse.
        assert overlay.getpixel(tuple(int(v) for v in replay.outline[0])) == (255, 215, 0)
        loose, status = replay.render(-0.2)
        assert 'Replay 1.000' in status and loose.getpixel(unmarked) != original.getpixel(unmarked)
        _, status = replay.render(1.)
        assert 'Replay 0.000' in status  # Equal logits stay outside.
        assert config.database.read_bytes() == before and backend.calls == 0
    finally:
        previews.close()


@pytest.mark.parametrize('damage', ['changed_source', 'clipped', 'eyes', 'missing_model'])
def test_replay_explains_unavailable_masks_without_guessing(tmp_path, monkeypatch, damage):
    config, reader, client, media, face, _, _ = vtt_fixture(tmp_path)
    if damage == 'changed_source':
        key = client.url + 'vtt/1.vtt'
        media.data[key] = media.data[key].replace(b'00:00:00.000', b'00:00:00.100')
    else:
        geometry = list(FACE.row)
        if damage == 'clipped':
            for i in (0, 4, 6, 8, 10, 12):
                geometry[i] -= 80
        if damage == 'eyes':
            geometry[6:8] = geometry[4:6]
        with FaceDatabase(config.database) as db:
            db.connection.execute('UPDATE observations SET geometry=? WHERE id=?', (json.dumps(geometry), face))
            db.connection.commit()

    def assets(*args, **kwargs):
        if damage != 'missing_model':
            pytest.fail('invalid source or alignment must be checked before loading FaceOcc')
        raise FileNotFoundError
    monkeypatch.setattr(inspector, 'ensure_assets', assets)
    previews = VttPreviews(config, client_factory=lambda: client)
    try:
        replay = inspector.replay(previews, face, reader.scan['scope'])
        original = replay.image
        overlay, caption = replay.render(0.0)
        if damage == 'changed_source':
            assert original is overlay is None and 'VTT source changed' in caption
        else:
            assert original.size == overlay.size and 'Recorded fraction unavailable' in caption
            expected = {'clipped': 'Insufficient face ellipse coverage', 'eyes': 'distinct eye', 'missing_model': 'weights missing'}
            assert expected[damage] in caption and 'Replay 0.' not in caption
            if damage == 'clipped':
                import re
                fraction = float(re.search(r'Source coverage ([\d.]+)%',caption).group(1))
                assert 45 < fraction < 55
    finally:
        previews.close()
