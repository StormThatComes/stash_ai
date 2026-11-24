"""Failed crop-only measurements must never revive full-frame landmarks."""
from contextlib import nullcontext
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from stash_ai.recognition import face_landmark_repair as repair, landmark_repair_audit as audit
from stash_ai.recognition.models import checked_detection
from stash_ai.recognition.quality import alignment_measurements
from tests.test_recognition_facebox_landmarks import CROP, model_for
from tests.test_recognition_landmark_repair import PROFILE, project, shifted


@pytest.mark.parametrize('raw', [PROFILE, shifted(PROFILE, -70), PROFILE.without_landmarks()])
def test_original_points_do_not_choose_crop_rotations(raw):
    pixels = np.arange(300*300*3, dtype=np.uint8).reshape(300, 300, 3)
    upright = pixels[35:185, 55:145]
    model, calls = model_for('scrfd_10g', [raw],
        lambda img: [project(PROFILE, CROP, 0)] if np.array_equal(img, upright) else [], rotation=True)
    with audit.capture() as reports:
        assert model.detect(pixels) == [PROFILE]
    trace = reports[0]['faces'][0]
    assert trace['estimated_turn_degrees'] is None
    assert trace['tried_turn_degrees'] == [0, 90, 180, 270]
    assert trace['selected_turn_degrees'] == 0
    assert sum(map(len, calls[1:])) == 4


@pytest.mark.parametrize('good_original', [False, True])
def test_original_fit_error_cannot_relax_crop_acceptance(good_original):
    target = checked_detection([*PROFILE.box, 98,85,100,86,90,150,98,65,102,65,.99])
    def translated(face):
        row = list(face.row)
        for i in (0, 1, *range(4, 14)):
            row[i] += 900
        return checked_detection(row)
    target = translated(target)
    original = translated(PROFILE) if good_original else target
    pixels = np.zeros((2000, 2000, 3), np.uint8)
    quality = alignment_measurements(target, SimpleNamespace(width=2000, height=2000))
    assert quality['alignment_valid'] and quality['alignment_error'] > .20
    model, _ = model_for('scrfd_10g', [original],
        [project(target, tuple(n+900 for n in CROP), 0)])
    with audit.capture() as reports:
        assert model.detect(pixels) == [original.without_landmarks()]
    candidate = reports[0]['faces'][0]['attempts'][0]['candidates'][0]
    assert candidate['rejection'] == 'alignment_error'


def test_empty_clipped_crop_leaves_box_without_points():
    outside = checked_detection([400,400,60,100,*PROFILE.row[4:]])
    model, calls = model_for('scrfd_10g', [outside], [])
    with audit.capture() as reports:
        assert model.detect(np.zeros((300,300,3), np.uint8)) == [outside.without_landmarks()]
    assert len(calls) == 1
    assert reports[0]['faces'][0]['status'] == 'empty_crop'


def test_disagreeing_crop_rotations_do_not_restore_original_points():
    from tests.test_recognition_landmark_repair import OBSERVED, OBSERVED_CROP, synthetic_retry
    pixels = np.arange(252*448*3, dtype=np.uint8).reshape(252,448,3)
    choices = {turn: project(synthetic_retry(turn), OBSERVED_CROP, turn) for turn in (1,3)}
    def answers(img):
        return [answer for turn, answer in choices.items()
                if np.array_equal(img, np.rot90(pixels[22:223,0:337], turn))]
    model, _ = model_for('scrfd_10g', [OBSERVED], answers, rotation=True)
    with audit.capture() as reports:
        assert model.detect(pixels) == [OBSERVED.without_landmarks()]
    assert reports[0]['faces'][0]['status'] == 'conflicting_orientations'


@pytest.mark.parametrize('turn', [0, 1, 2, 3])
def test_restoring_crop_preserves_missing_sentinel_but_transforms_real_negative_points(turn):
    missing = PROFILE.without_landmarks()
    restored = repair.restore(missing, CROP, turn)
    assert not restored.landmarks_available and restored.landmarks == ((-1.,-1.),)*5
    raw = checked_detection([*PROFILE.box, -1,-1, *PROFILE.row[6:]])
    assert raw.landmarks_available
    # A single real point outside the crop is still a coordinate, not absence.
    expected = [(54,34), (145,34), (145,185), (54,185)][turn]
    assert repair.restore(raw, CROP, turn).landmarks[0] == expected


def test_policy_upgrade_invalidates_only_enabled_face_and_scan_caches(tmp_path, monkeypatch):
    from stash_ai.recognition.config import load, plain
    from stash_ai.recognition.inference_contracts import face, detector, pose
    from stash_ai.recognition.scanning import scan_identity
    off = load(root=tmp_path, device='cpu')
    data = plain(off.data)
    data['detection']['landmarks_from_facebox'] = True
    on = replace(off, data=data)
    def keys(config):
        return face(config), scan_identity(config, 'ns'), detector(config), pose(config)
    current_on, current_off = keys(on), keys(off)
    monkeypatch.setattr(repair, 'FACEBOX_CONTRACT', 'facebox-landmarks-v1')
    previous_on = keys(on)
    assert current_on[0] != previous_on[0] and current_on[1] != previous_on[1]
    assert current_on[2:] == previous_on[2:] and current_off == keys(off)


@pytest.mark.parametrize('visibility_enabled', [False, True])
def test_scan_retains_box_and_rejects_missing_crop_points_before_embedding(tmp_path, visibility_enabled):
    from stash_ai.recognition.config import plain
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.media import SpriteInput
    from stash_ai.recognition.scanning import scan_scene
    from tests.helpers.recognition import Backend, FACE, MediaClient
    from tests.test_recognition_scanning import config
    settings = config(tmp_path)
    data = plain(settings.data)
    data['detection']['landmarks_from_facebox'] = True
    data['detection']['min_visible_face_fraction'] = .5 if visibility_enabled else 0
    settings = replace(settings, data=data)
    crop_detector = SimpleNamespace(config=settings, _raw_many=lambda images: [[] for _ in images])
    backend = Backend()
    backend.detect = lambda image: repair.repair(crop_detector, [np.asarray(image)], [[FACE]])[0]
    backend.embed = lambda *args: pytest.fail('missing crop landmarks reached the face encoder')
    visibility = SimpleNamespace(measure=lambda *args: pytest.fail('missing crop landmarks reached FaceOcc'))
    client = MediaClient()
    with FaceDatabase(settings.database) as state, SpriteInput(client, client.scene, settings) as source:
        scan_scene(state, 'scope', client.scene, settings, 'ns', source, lambda: backend, visibility=visibility)
        rows = state.connection.execute('SELECT * FROM observations').fetchall()
        assert len(rows) == len(source.cues)
        for row in rows:
            assert row['rejected'] == 'face_landmarks_invalid'
            assert row['embedding'] is None and row['crop'] is None
            assert json.loads(row['geometry']) == list(FACE.without_landmarks().row)
            assert not json.loads(row['quality'])['alignment_valid']


def test_tuner_shows_box_and_missing_label_without_ghost_points():
    from stash_ai.recognition import detection_audit
    from stash_ai.recognition.detection_associations import face_label
    model, _ = model_for('scrfd_10g', [PROFILE], [])
    backend = SimpleNamespace(prepared_frame=lambda image: nullcontext(), frame_people=[], face_matches=[None],
        detect=lambda image, **kwargs: model.detect(np.asarray(image)))
    cue = SimpleNamespace(start=0., end=1., rect=(0,0,300,300))
    with Image.new('RGB', (300,300)) as image, detection_audit.render_frame(
            image, backend, '1', 0, cue) as (_, record):
        face = record['faces'][0]
        assert face['box'] == list(PROFILE.box)
        assert not face['landmarks_available'] and face['drawn_indices'] == []
        assert face['display_landmarks'] == [[-1.,-1.]]*5
        assert 'landmarks unavailable' in face_label(0, PROFILE.without_landmarks(), None, face['association'])


def test_inspector_draws_box_but_no_missing_landmarks():
    from stash_ai.recognition.inspection_overlays import COLORS, render
    geometry = dict(bounds=[50,30,150,190], notes=[], shapes=[
        dict(kind='face', selected=True, geometry=list(PROFILE.without_landmarks().row))])
    with Image.new('RGB', (100,160)) as image:
        annotated, caption = render(image, geometry)
        with annotated:
            assert '0 points' in caption and 'landmarks unavailable' in caption
            assert annotated.getpixel((20,30)) == COLORS['face']
            assert annotated.getpixel((48,55)) == (0,0,0)  # Former eye.
