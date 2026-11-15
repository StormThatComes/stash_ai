"""FaceOcc pixel policy is shared by scan, portrait, SQL admission and inspection."""
from dataclasses import replace
import json
import sqlite3

import numpy as np
import pytest

from stash_ai.recognition.config import digest, load, plain
from stash_ai.recognition.models import namespace
from stash_ai.recognition.pose_policy import compact_quality, visibility_sql
from stash_ai.recognition.portrait_admission import measurement_quality
from stash_ai.recognition.portraits import policy_identity
from stash_ai.recognition.scanning import scan_identity
from stash_ai.recognition.visibility import CONTRACT_ID, cached, device_measurements, measurement


def configuration(tmp_path, threshold):
    path = tmp_path / 'config.yml'
    path.write_text(f'version: 2\nface:\n  quality: {{faceocc_logit_threshold: {threshold}}}\n')
    return load(path, device='cpu')


@pytest.mark.parametrize('token', ['.nan', '.inf', '-.inf', 'true', 'null', '"-0.2"', '1.0e+50'])
def test_reject_invalid_pixel_threshold(tmp_path, token):
    with pytest.raises(ValueError, match='face.quality.faceocc_logit_threshold'):
        configuration(tmp_path, token)


def test_default_can_read_existing_scans_and_tuning_changes_measurement_keys(tmp_path):
    default = configuration(tmp_path, 0)
    tuned = configuration(tmp_path, -0.2)
    ns = namespace(default)
    legacy = plain(default.data['detection'])
    del legacy['faceocc_logit_threshold']
    legacy_sha = digest(dict(namespace=ns, detection=legacy, max_cues=default.data['execution']['max_cues'],
                             quality_format=5, visibility_contract=CONTRACT_ID))
    assert scan_identity(default, ns) == legacy_sha != scan_identity(tuned, ns)
    assert policy_identity(default) != policy_identity(tuned)
    for config in (default, tuned):
        data = plain(config.data)
        data['detection'].update(min_visible_face_fraction=0., min_sharpness=0.)
        disabled = replace(config, data=data)
        if config is default:
            unmeasured = scan_identity(disabled, ns)
        else:
            assert scan_identity(disabled, ns) == unmeasured
    old = dict(visible_face_fraction=.23, visibility={'contract': CONTRACT_ID}, sharpness_method='faceocc_mask',
               sharpness=75, measurement_basis={'visibility': {'source_type': 'vtt'}})
    assert measurement_quality(old, default) is old and cached(old, 0.)
    detached = measurement_quality(old, tuned)
    assert not cached(detached) and detached['sharpness_method'] == 'unavailable' and not detached['measurement_basis']
    assert old['sharpness'] == 75 and not cached(old, -0.2)


@pytest.mark.parametrize('threshold, expected', [(-0.2, 2/3), (0, 1/3), (0.25, 0.)])
def test_mask_counts_sharpness_and_role_eligibility_match_for_numpy_and_torch(threshold, expected):
    import torch
    scores = np.zeros((1, 1, 256, 256), dtype='float32')
    scores[..., :64] = -0.2
    scores[..., 128:] = 0.25
    region = np.zeros((256, 256), dtype=bool)
    region[:, :192] = True
    blob = np.random.default_rng(3).normal(size=(1, 3, 256, 256)).astype('float32')
    summary = measurement(scores, region, 'cpu', blob=blob, threshold=threshold)
    other, = device_measurements(torch.from_numpy(scores), [region], 'cpu', inputs=torch.from_numpy(blob),
                                threshold=threshold)
    assert summary['visible_face_fraction'] == other['visible_face_fraction'] == expected
    assert summary['visibility'] == other['visibility'] and summary['visibility']['logit_threshold'] == threshold
    assert summary['sharpness_pixels'] == other['sharpness_pixels']
    assert summary['sharpness'] == pytest.approx(other['sharpness'], rel=1e-4, abs=1e-5)
    assert cached(json.loads(compact_quality(json.dumps(summary))), threshold)
    with sqlite3.connect(':memory:') as db:
        query = f'SELECT {visibility_sql("o", 0.25)} FROM (SELECT ? AS quality) o'
        assert bool(db.execute(query, (json.dumps(summary),)).fetchone()[0]) == (expected >= .25)


def test_yaml_changes_scan_admission_and_keeps_evidence_for_each_cutoff(tmp_path, monkeypatch):
    import torch
    import safetensors.torch
    from stash_ai.recognition import faceocc, visibility
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.media import SpriteInput
    from stash_ai.recognition.scanning import scan_scene
    from tests.helpers.recognition import Backend, MediaClient

    class PixelModel(torch.nn.Module):
        def forward(self, inputs):
            return torch.full_like(inputs[:, :1], -0.1)

    monkeypatch.setattr(faceocc, 'build_model', PixelModel)
    monkeypatch.setattr(visibility, 'verified_bytes', lambda *_: b'weights')
    monkeypatch.setattr(safetensors.torch, 'load', lambda _: {})
    media = MediaClient(count=1)
    ids = []
    for cutoff, rejected, fraction in [(0, 'face_occluded', 0.), (-0.2, None, 1.), (0, 'face_occluded', 0.)]:
        config = configuration(tmp_path, cutoff)
        model = visibility.FaceVisibility(config, {'faceocc': None})
        try:
            with FaceDatabase(config.database) as db, SpriteInput(media, media.scene, config) as source:
                scan, reused = scan_scene(db, 'scope', media.scene, config, namespace(config), source,
                                          Backend, visibility=model)
                row = db.connection.execute('SELECT rejected,quality FROM observations WHERE scan=?', (scan,)).fetchone()
                quality = json.loads(row['quality'])
                assert row['rejected'] == rejected and quality['visible_face_fraction'] == fraction
                assert cached(quality, cutoff)
                assert bool(reused) == (scan in ids)
                ids.append(scan)
        finally:
            model.close()
    assert ids[0] == ids[2] != ids[1]


def test_retuning_portrait_only_masks_reuses_scan_and_updates_optional_pixels(tmp_path, monkeypatch):
    import yaml
    from stash_ai.recognition.database import FaceDatabase
    from tests.helpers.recognition import Visibility
    from tests.test_recognition_vtt_portraits import setup, run

    config, _, services, backends = setup(tmp_path, monkeypatch)
    assert run(config, services, '1') == 0
    data = yaml.safe_load(config.source_path.read_text())
    data['detection']['faceocc_logit_threshold'] = -0.2
    config.source_path.write_text(yaml.safe_dump(data))
    tuned = load(config.source_path)
    runs = []

    class PortraitMask(Visibility):
        def measure(self, image, face):
            cutoff = tuned.data['detection']['faceocc_logit_threshold']
            runs.append(cutoff)
            return measurement(np.zeros((1, 1, 256, 256), dtype='float32'), np.ones((256, 256), dtype=bool),
                               'cpu', threshold=cutoff)

    services.visibility = PortraitMask
    assert run(tuned, services, '1') == 0
    assert len(backends) == 1 and runs == [-0.2]*3
    with FaceDatabase(config.database) as state:
        assert state.connection.execute('SELECT COUNT(*) FROM scans').fetchone()[0] == 1
        rows = state.connection.execute('SELECT quality FROM observations').fetchall()
        assert len(rows) == 3 and all(cached(json.loads(r[0]), -0.2) for r in rows)
    assert run(tuned, services, '1') == 0 and len(runs) == 3
