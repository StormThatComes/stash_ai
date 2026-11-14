"""Roll correction preserves occlusion gates, source coverage and cache refresh."""
from dataclasses import replace
from io import BytesIO
import json

import numpy as np
from PIL import Image, ImageDraw
import pytest

from stash_ai.recognition.models import checked_detection
from stash_ai.recognition.visibility import (
    CONTRACT, CONTRACT_ID, FaceVisibility, VisibilityGeometryError, cached, prepare,
)
from tests.helpers.recognition import FACE, Backend, MediaClient, textured_image


def quarter_turn(image, face, turns):
    """Rotate pixels, box edges and semantic landmarks counterclockwise together."""
    result = image.copy()
    for _ in range(turns % 4):
        width = result.width
        rotated = result.transpose(Image.Transpose.ROTATE_90)
        result.close()
        result = rotated
        x, y, w, h = face.box
        points = [(py, width - 1 - px) for px, py in face.landmarks]
        face = checked_detection([y, width - x - w, h, w,
                                  *[v for point in points for v in point], face.score])
    return result, face


@pytest.mark.parametrize('turns', [1, 2, 3])
def test_quarter_turns_restore_pixels_without_mirroring_or_rescaling(turns):
    face = checked_detection([56, 40, 128, 160, 90, 100, 150, 100,
                              120, 130, 96, 164, 144, 164, .99])
    pixels = np.random.default_rng(11).integers(0, 256, (240, 280, 3), dtype=np.uint8)
    with Image.fromarray(pixels) as image:
        expected, _ = prepare(image, face)
        rotated, detection = quarter_turn(image, face, turns)
        with rotated:
            actual, region = prepare(rotated, detection)
    np.testing.assert_array_equal(actual, expected)
    assert region.any() and region.dtype == np.bool_


@pytest.mark.parametrize('angle', [-135, -45, 30, 90, 135, 180, 270])
def test_oblique_roll_restores_eye_order_and_keeps_mouth_below_eyes(angle):
    with Image.new('RGB', (160, 160)) as image:
        draw = ImageDraw.Draw(image)
        for point, color in [(FACE.landmarks[0], 'red'), (FACE.landmarks[1], 'lime'), ((80, 108), 'blue')]:
            x, y = point
            draw.ellipse((x-5, y-5, x+5, y+5), fill=color)
        # Rotation around the same image/box center; keeping a square ROI here
        # isolates alignment from detector box changes at oblique angles.
        radians = np.deg2rad(-angle)
        c, s = np.cos(radians), np.sin(radians)
        points = (np.asarray(FACE.landmarks)-79.5) @ np.array([[c, s], [-s, c]]) + 79.5
        face = checked_detection([*FACE.box, *points.ravel(), FACE.score])
        with image.rotate(angle, resample=Image.Resampling.BILINEAR) as rotated:
            blob, _ = prepare(rotated, face)
    rgb = blob[0].transpose(1, 2, 0)*[.229, .224, .225]+[.485, .456, .406]
    centers = [np.argwhere((rgb[..., channel] > .7) & (rgb[..., (channel+1) % 3] < .2)).mean(axis=0)
               for channel in range(3)]
    left, right, mouth = centers
    assert abs(left[0]-right[0]) < 1.5
    assert left[1] < mouth[1] < right[1]
    assert mouth[0] > max(left[0], right[0])+40


@pytest.mark.parametrize('turns', [0, 1, 2, 3])
def test_rotated_source_coverage_cannot_count_padding_as_visible(turns):
    row = list(FACE.row)
    for index in (0, 4, 6, 8, 10, 12):
        row[index] -= 40
    face = checked_detection(row)
    with Image.new('RGB', (160, 160), 'white') as image:
        rotated, detection = quarter_turn(image, face, turns)
        with rotated:
            with pytest.raises(VisibilityGeometryError, match='real source'):
                prepare(rotated, detection)
            blob, region = prepare(rotated, detection, allow_partial=True)
            rgb = blob[0].transpose(1, 2, 0)*[.229, .224, .225]+[.485, .456, .406]
            assert region.any()
            assert np.all(rgb[region] > .99)


@pytest.mark.parametrize('turns', [0, 1, 2, 3])
def test_partial_ellipse_real_foreground_full_denominator_on_both_reductions(turns):
    import torch
    from stash_ai.recognition.visibility import prepare_measurement, measurement, device_measurements
    row = list(FACE.row)
    for index in (0, 4, 6, 8, 10, 12):
        row[index] -= 40
    with Image.new('RGB', (160, 160), 'white') as image:
        rotated, face = quarter_turn(image, checked_detection(row), turns)
        with rotated:
            blob, real, total = prepare_measurement(rotated, face, minimum_source_fraction=0.0)
    logits = np.ones((1, 1, 256, 256), dtype='float32')
    logits[..., 128:] = -1
    expected = np.count_nonzero(real[:, :128]) / total
    cpu = measurement(logits, real, 'cpu', blob=blob, face_pixels=total)
    device = device_measurements(torch.from_numpy(logits), [real], 'cpu', inputs=torch.from_numpy(blob),
                                 face_pixels=[total])[0]
    assert 0 < expected < real.sum() / total < 1
    assert cpu['visible_face_fraction'] == device['visible_face_fraction'] == expected
    assert cpu['visibility'] == device['visibility']
    assert cpu['sharpness_pixels'] == device['sharpness_pixels']


def test_configured_source_coverage_endpoints_and_exact_boundary():
    from stash_ai.recognition.visibility import prepare_measurement
    row = list(FACE.row)
    for index in (0, 4, 6, 8, 10, 12):
        row[index] -= 40
    face = checked_detection(row)
    with textured_image() as image:
        _, region, total = prepare_measurement(image, face, 0.)
        coverage = int(region.sum()) / total
        assert 0 < coverage < 1
        prepare_measurement(image, face, coverage)
        with pytest.raises(VisibilityGeometryError):
            prepare_measurement(image, face, np.nextafter(coverage, 1.))
        with pytest.raises(VisibilityGeometryError):
            prepare_measurement(image, face, 1.)
        prepare_measurement(image, FACE, 1.)
    with Image.new('RGB', (1, 1)) as image, pytest.raises(VisibilityGeometryError):
        prepare_measurement(image, FACE, 0.)


def test_degenerate_eye_line_is_an_observation_local_geometry_error():
    row = list(FACE.row)
    row[6:8] = row[4:6]
    with textured_image() as image:
        with pytest.raises(VisibilityGeometryError, match='distinct eye'):
            prepare(image, checked_detection(row))


@pytest.mark.parametrize('angle', [-45, 30, 135])
def test_oblique_crop_padding_does_not_reject_a_complete_source_face(angle):
    radians = np.deg2rad(angle)
    c, s = np.cos(radians), np.sin(radians)
    points = (np.asarray(FACE.landmarks)-80) @ np.array([[c, s], [-s, c]])+80
    face = checked_detection([*FACE.box, *points.ravel(), FACE.score])
    with Image.new('RGB', (160, 160), 'white') as image:
        blob, region = prepare(image, face)
    rgb = blob[0].transpose(1, 2, 0)*[.229, .224, .225]+[.485, .456, .406]
    assert np.any(rgb < .01)  # The square model crop extends outside the image.
    assert np.all(rgb[region] > .99)  # Its measured ellipse is fully real.


def upright_mask_model():
    """Orientation-sensitive test double; real weights are covered by model smoke."""
    import torch
    class Model:
        def __call__(self, blob):
            rgb = blob*blob.new_tensor([.229, .224, .225])[None, :, None, None]
            rgb += blob.new_tensor([.485, .456, .406])[None, :, None, None]
            upright = (rgb[:, 0, 101, 87] > .8) & (rgb[:, 1, 101, 169] > .8)
            foreground = rgb.amax(dim=1) > .05
            return torch.where(foreground & upright[:, None, None], 1., -1.)[:, None]
    model = object.__new__(FaceVisibility)
    model.device, model.model = 'cpu', Model()
    model.minimum_source_fraction = 1.0
    return model


@pytest.mark.parametrize('batch_size', [1, 4])
def test_scan_keeps_rolled_faces_but_rejects_real_occlusion(tmp_path, batch_size):
    from stash_ai.recognition.config import plain
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.media import SpriteInput
    from stash_ai.recognition.scanning import scan_scene
    from tests.test_recognition_detection_quality import settings
    config = settings(tmp_path, min_visible_face_fraction=.8, min_sharpness=1)
    data = plain(config.data)
    data['execution'].update(scan_cpu_workers=2, recognizer_batch_size=batch_size)
    config = replace(config, data=data)
    faces = []
    client = MediaClient(count=8)
    with Image.new('RGB', (1280, 160)) as sprite:
        for index in range(8):
            with textured_image() as image:
                draw = ImageDraw.Draw(image)
                draw.ellipse((44, 55, 54, 65), fill='red')
                draw.ellipse((106, 55, 116, 65), fill='lime')
                if index >= 4:
                    draw.rectangle((0, 80, 159, 159), fill='black')
                rotated, face = quarter_turn(image, FACE, index % 4)
                with rotated:
                    sprite.paste(rotated, (160*index, 0))
                faces.append(face)
        stream = BytesIO()
        sprite.save(stream, format='PNG')
        client.data['http://stash/sprite/1.jpg'] = stream.getvalue()
    client.data['http://stash/vtt/1.vtt'] = ('WEBVTT\n\n'+'\n\n'.join(
        f'00:00:{i:02d}.000 --> 00:00:{i+1:02d}.000\nsprite.jpg#xywh={i*160},0,160,160'
        for i in range(8))).encode()
    backend = Backend()
    detections = iter(faces)
    backend.detect = lambda image: [next(detections)]
    model = upright_mask_model()
    try:
        with FaceDatabase(config.database) as db, SpriteInput(client, client.scene, config) as source:
            scan, reused = scan_scene(db, 'scope', client.scene, config, 'ns', source,
                                      lambda: backend, visibility=model)
            rows = db.connection.execute('SELECT rejected,embedding,quality FROM observations WHERE scan=? ORDER BY cue',
                                         (scan,)).fetchall()
            assert not reused and backend.calls == 4
            assert [row['rejected'] for row in rows] == [None]*4+['face_occluded']*4
            assert all(row['embedding'] is not None for row in rows[:4])
            assert all(row['embedding'] is None for row in rows[4:])
            assert all(cached(json.loads(row['quality'])) for row in rows)
            repeated, reused = scan_scene(db, 'scope', client.scene, config, 'ns', source,
                lambda: pytest.fail('current scan must reuse its measurements'))
            assert reused and repeated == scan
    finally:
        model.close()


@pytest.mark.parametrize('visibility_floor', [0, .8])
def test_old_completed_rejected_scan_is_reprocessed_once(tmp_path, monkeypatch, visibility_floor):
    from stash_ai.recognition import scanning
    from stash_ai.recognition.config import digest
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.media import SpriteInput
    from tests.helpers.recognition import Visibility
    from tests.test_recognition_detection_quality import settings
    config = settings(tmp_path, min_visible_face_fraction=visibility_floor)
    old_contract = dict(CONTRACT, version=2, crop='square-face-box-padding-0.25',
                        region='face-box-ellipse-0.45w-0.48h')
    old_id = digest(old_contract)
    assert old_id != CONTRACT_ID
    assert not cached(dict(visible_face_fraction=1., visibility=dict(contract=old_id)))
    def old_policy(config, namespace):
        return digest(dict(namespace=namespace, detection=config.data['detection'],
            max_cues=config.data['execution']['max_cues'], quality_format=4,
            visibility_contract=old_id if visibility_floor else None))
    client = MediaClient(count=1)
    with FaceDatabase(config.database) as db, SpriteInput(client, client.scene, config) as source:
        with monkeypatch.context() as patch:
            patch.setattr(scanning, 'scan_identity', old_policy)
            previous, _ = scanning.scan_scene(db, 'scope', client.scene, config, 'ns', source,
                                              Backend, visibility=Visibility())
        with db.connection:
            db.connection.execute("UPDATE observations SET rejected='face_occluded',embedding=NULL WHERE scan=?", (previous,))
        current, reused = scanning.scan_scene(db, 'scope', client.scene, config, 'ns', source,
                                              Backend, visibility=Visibility())
        assert not reused and current != previous
        row = db.connection.execute('SELECT rejected,embedding FROM observations WHERE scan=?', (current,)).fetchone()
        assert row['rejected'] is None and row['embedding'] is not None
        again, reused = scanning.scan_scene(db, 'scope', client.scene, config, 'ns', source,
            lambda: pytest.fail('updated scan must be reused'))
        assert reused and again == current
