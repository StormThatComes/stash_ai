"""Retained report overlays map source geometry without touching evidence pixels."""
from io import BytesIO
import json
import os
import sqlite3

import pytest
from PIL import Image

from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.identity_report import export
from stash_ai.recognition.person_audit import BOX_COLOR, FACE_COLORS, POSE_COLOR
from stash_ai.recognition.person_signal import contract, save
from stash_ai.recognition.retained_annotations import export as annotate
from tests.test_recognition_identity_report import seed


GEOMETRY = [110, 220, 50, 50, 120, 230, 150, 230, 135, 245, 125, 260, 145, 260, .99]


def test_report_maps_downscaled_face_and_offset_person_points_offline(tmp_path, monkeypatch):
    config, person, face_blob = seed(tmp_path, count=1)
    with Image.new('RGB', (120, 200), (10, 20, 30)) as image:
        buffer = BytesIO()
        image.save(buffer, format='PNG')
    points = [[90, 150], [-1, -1], [50, 150], [180, 150], [90, 300]] + [[-1, -1]] * 12
    with FaceDatabase(config.database) as state:
        db = state.connection
        observation = db.execute('SELECT id FROM observations').fetchone()[0]
        db.execute("UPDATE observations SET geometry=?,quality=json_set(quality,'$.crop_bounds',json(?))",
                   (json.dumps(GEOMETRY), json.dumps([100, 200, 174, 286])))
        save(db, observation, contract(config), dict(identity_rejection=None,
             person=dict(bounds=[60, 100, 180, 300], keypoints=points)), [1.] * 4096, buffer.getvalue())
        db.commit()
    def forbidden(*args, **kwargs):
        pytest.fail('Retained overlays must not load models or media')
    monkeypatch.setattr('stash_ai.recognition.models.load_backend', forbidden)
    monkeypatch.setattr('stash_ai.recognition.models.ensure_assets', forbidden)
    monkeypatch.setattr('stash_ai.recognition.run.Services', forbidden)
    # An enabled collector must not be instantiated by an offline export either.
    monkeypatch.setattr('stash_ai.recognition.person_collection.PersonCollector', forbidden)
    before = config.database.read_bytes()
    target = tmp_path / 'report'
    report = export(config, target, performer_id='900')
    assert config.database.read_bytes() == before
    assert json.loads((target / 'manifest.json').read_text()) == report
    sample = report['samples'][0]
    face, body = sample['image'], sample['person_evidence']['image']
    assert face['annotations']['source_to_crop_scale'] == [.5, .5]
    assert face['annotations']['drawn_indices'] == list(range(5))
    assert body['annotations']['drawn_indices'] == [0, 17, 18, 19, 20, 21]
    assert body['annotations']['pose_indices'] == list(range(17))
    assert body['annotations']['face_indices'] == list(range(17, 22))
    with Image.open(BytesIO(face_blob)) as original, Image.open(target / face['file']) as clean:
        assert clean.tobytes() == original.tobytes()
    with Image.open(target / body['file']) as clean:
        assert clean.getpixel((30, 50)) == (10, 20, 30)
    with Image.open(target / face['annotations']['file']) as annotated:
        assert annotated.size == (37, 43)
        assert annotated.getpixel((10, 15)) == FACE_COLORS[0]
        assert annotated.getpixel((5, 30)) == BOX_COLOR
    with Image.open(target / body['annotations']['file']) as annotated:
        assert annotated.getpixel((30, 50)) == POSE_COLOR
        assert annotated.getpixel((60, 130)) == FACE_COLORS[0]
        assert annotated.getpixel((0, 100)) == BOX_COLOR
        # Outside/missing points never get clamped into border markers.
        assert annotated.getpixel((119, 50)) == BOX_COLOR
        assert annotated.getpixel((30, 199)) == BOX_COLOR
    assert len(list(target.glob('*.png'))) == 6
    if os.name == 'posix':
        assert all(path.stat().st_mode & 0o777 == 0o600 for path in target.iterdir())


@pytest.mark.parametrize('bounds,status', [
    (None, 'missing_crop_bounds'),
    ([0, 0, 0, 20], 'invalid_crop_bounds'),
    ([20, 0, 10, 20], 'invalid_crop_bounds'),
    ([0, 0, 'invalid', 20], 'invalid_crop_bounds'),
])
def test_unavailable_crop_geometry_keeps_clean_export(tmp_path, bounds, status):
    config, person, _ = seed(tmp_path, count=1)
    with sqlite3.connect(config.database) as db:
        quality = json.loads(db.execute('SELECT quality FROM observations').fetchone()[0])
        quality['crop_bounds'] = bounds
        db.execute('UPDATE observations SET quality=?', (json.dumps(quality),))
    report = export(config, tmp_path / 'report', identity_id=person)
    image = report['samples'][0]['image']
    assert image['status'] == 'exported'
    assert image['annotations'] == {'status': status, 'file': None}
    assert report['summary']['complete']
    assert len(list((tmp_path / 'report').glob('*.png'))) == 2


@pytest.mark.parametrize('keypoints,status', [
    (None, 'missing_pose_geometry'),
    ([[1, 2]] * 16, 'missing_pose_geometry'),
    ([[1]] * 17, 'invalid_pose_geometry'),
    ([['bad', 2]] * 17, 'invalid_geometry'),
])
def test_missing_or_invalid_pose_geometry_never_invents_points(tmp_path, keypoints, status):
    with Image.new('RGB', (120, 200)) as image:
        result = annotate(image, tmp_path / 'person.png', GEOMETRY, [60, 100, 180, 300],
                          pose=True, keypoints=keypoints)
    assert result == {'status': status, 'file': None}
    assert not list(tmp_path.iterdir())


def test_nonfinite_and_outside_points_are_omitted_without_mutating_input(tmp_path):
    points = [[float('nan'), 150], [90, float('inf')], [60, 100], [179, 299]] + [[-1, -1]] * 13
    with Image.new('RGB', (120, 200), (10, 20, 30)) as image:
        before = image.tobytes()
        result = annotate(image, tmp_path / 'person.png', GEOMETRY, [60, 100, 180, 300],
                          pose=True, keypoints=points)
        assert image.tobytes() == before
    assert result['drawn_indices'] == [2, 3, 17, 18, 19, 20, 21]


def test_face_overlay_omits_landmarks_outside_a_clipped_crop(tmp_path):
    # Only the left eye, nose and left mouth corner remain in this source region.
    with Image.new('RGB', (15, 25)) as image:
        result = annotate(image, tmp_path / 'face.png', GEOMETRY, [110, 220, 140, 270])
    assert result['drawn_indices'] == [0, 2, 3]
    assert result['source_to_crop_scale'] == [.5, .5]


def test_annotation_write_error_is_fatal(tmp_path, monkeypatch):
    config, person, _ = seed(tmp_path, count=1)
    def fail(*args):
        raise OSError('synthetic annotation disk full')
    monkeypatch.setattr('stash_ai.recognition.retained_annotations._png', fail)
    target = tmp_path / 'report'
    with pytest.raises(OSError, match='annotation disk full'):
        export(config, target, identity_id=person)
    assert not (target / 'manifest.json').exists()
