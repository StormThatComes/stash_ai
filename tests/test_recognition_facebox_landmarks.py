"""Crop-only measurements preserve full-frame boxes, never full-frame points."""
from contextlib import closing
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from stash_ai.recognition.config import defaults, load
from stash_ai.recognition.detection_parameters import apply_edits
from stash_ai.recognition.models import checked_detection, YuNetDetector
from stash_ai.recognition.scrfd import ScrfdDetector
from stash_ai.recognition.retinaface import RetinaFaceDetector
from stash_ai.recognition import landmark_repair_audit as audit
from tests.test_recognition_landmark_repair import PROFILE, detector, project, shifted


KEY = 'face.detector.landmarks_from_facebox'
CROP = (55, 35, 145, 185)


def model_for(key, faces, answer, *, enabled=True, rotation=False):
    stub, calls = detector(faces, answer, rotation=rotation)
    cls = YuNetDetector if key == 'yunet' else RetinaFaceDetector if key == 'retinaface_r50' else ScrfdDetector
    model = object.__new__(cls)
    model.config = stub.config
    model.config.data['models']['detector'] = key
    model.config.data['detection']['landmarks_from_facebox'] = enabled
    if not rotation:
        model.config.data['sapiensid']['enabled'] = False
        model.config.data['sapiensid']['collection']['enabled'] = False
    model._raw_many = stub._raw_many
    model._raw_facebox_many = stub._raw_many
    model.ort_detector = None
    return model, calls


@pytest.mark.parametrize('key', ['scrfd_10g', 'scrfd_2_5g', 'retinaface_r50', 'yunet'])
@pytest.mark.parametrize('enabled', [False, True])
def test_all_backends_refine_valid_upright_points_without_body_or_error_trigger(key, enabled):
    original = shifted(PROFILE, -7)
    # Crop predicts a slightly different box and score. Only its points belong
    # in the final result; unchanged shape residual must not suppress the mode.
    cropped = checked_detection([72, 62, 56, 96, *PROFILE.row[4:14], .97])
    model, calls = model_for(key, [original], [project(cropped, CROP, 0)], enabled=enabled)
    pixels = np.arange(230*300*3, dtype=np.uint8).reshape(230, 300, 3)
    with audit.capture() as reports, Image.fromarray(pixels[:, :, ::-1]) as image:
        result = model.detect(image if key == 'yunet' else pixels)
    assert result[0].box == original.box and result[0].score == original.score
    assert result[0].landmarks == (PROFILE.landmarks if enabled else original.landmarks)
    assert len(calls) == (2 if enabled else 1)
    if enabled:
        np.testing.assert_array_equal(calls[1][0], pixels[35:185, 55:145])
        assert reports[0]['landmarks_from_facebox'] is True
        trace = reports[0]['faces'][0]
        assert trace['status'] == 'refined_from_facebox'
        assert trace['result']['box'] == list(original.box)
        assert trace['attempts'][0]['candidates'][0]['source_detection']['box'] == list(cropped.box)


@pytest.mark.parametrize('turn', [0, 1, 2, 3])
def test_crop_and_rotation_restore_only_points_and_bound_retries(turn):
    expected = project(PROFILE, (0, 0, 300, 300), turn)
    wrong = shifted(expected, -70)
    x, y, w, h = wrong.box
    crop = tuple(map(int, (x-.25*w, y-.25*h, x+1.25*w, y+1.25*h)))
    pixels = np.arange(300*300*3, dtype=np.uint8).reshape(300, 300, 3)
    upright_turn = (-turn) % 4
    upright = np.rot90(pixels[crop[1]:crop[3], crop[0]:crop[2]], upright_turn)
    answer = project(expected, crop, upright_turn)
    model, calls = model_for('scrfd_10g', [wrong],
        lambda img: [answer] if np.array_equal(img, upright) else [], rotation=True)
    with audit.capture() as reports:
        result = model.detect(pixels)
    np.testing.assert_allclose(result[0].row, expected.row, atol=1e-6)
    assert sum(map(len, calls[1:])) == 4
    assert all(len(batch) <= 2 for batch in calls[1:])
    assert reports[0]['faces'][0]['selected_turn_degrees'] == upright_turn*90


def test_clipped_non_square_crop_uses_actual_origin_not_unclipped_margin():
    row = list(PROFILE.row)
    row[0] -= 60; row[1] -= 50
    for i in range(4, 14, 2):
        row[i] -= 60; row[i+1] -= 50
    expected = checked_detection(row)
    crop = (0, 0, 85, 135)
    model, calls = model_for('scrfd_10g', [shifted(expected, -7)], [project(expected, crop, 0)])
    pixels = np.arange(180*240*3, dtype=np.uint8).reshape(180, 240, 3)
    result = model.detect(pixels)
    assert result[0].row == expected.row
    np.testing.assert_array_equal(calls[1][0], pixels[:135, :85])


@pytest.mark.parametrize('shape', [(253, 448), (448, 253)])
def test_scrfd_facebox_content_keeps_fixed_anchors_and_exact_source_scale(shape):
    pixels = np.full((*shape, 3), [10, 80, 150], np.uint8)
    blob, sx, sy = ScrfdDetector._blob(pixels, content_size=ScrfdDetector.FACEBOX_CONTENT_SIZE)
    width, height = int(shape[1]*320/max(shape)), int(shape[0]*320/max(shape))
    assert blob.shape == (1, 3, 640, 640)
    np.testing.assert_allclose(blob[0, :, height-1, width-1], (np.array([150,80,10])-127.5)/128)
    np.testing.assert_allclose(blob[0, :, height, width], [-127.5/128]*3)
    assert sx == width/shape[1] and sy == height/shape[0]


@pytest.mark.parametrize('failure', ['empty', 'different_head', 'competing_head', 'ambiguous', 'weak', 'invalid'])
@pytest.mark.parametrize('key', ['scrfd_10g', 'scrfd_2_5g', 'retinaface_r50', 'yunet'])
def test_failed_or_ambiguous_crop_clears_points_and_records_reason(failure, key):
    original = shifted(PROFILE, -7)
    target = PROFILE
    faces = [original]
    if failure == 'different_head':
        row = list(PROFILE.row); row[0] += 50
        for i in range(4, 14, 2): row[i] += 50
        target = checked_detection(row)
    if failure == 'competing_head': faces.append(PROFILE)
    if failure == 'weak': target = checked_detection([*PROFILE.row[:-1], .7])
    if failure == 'invalid': target = shifted(PROFILE, -70)
    answers = [] if failure == 'empty' else [project(target, CROP, 0)]
    if failure == 'ambiguous': answers *= 2
    model, _ = model_for(key, faces, answers)
    with audit.capture() as reports, Image.new('RGB', (300, 300)) as image:
        result = model.detect(image if key == 'yunet' else np.asarray(image))
    assert result == [face.without_landmarks() for face in faces]
    assert all(not face.landmarks_available for face in result)
    trace = reports[0]['faces'][0]
    assert trace['status'] == 'no_eligible_candidate'
    assert trace['result']['landmarks'] == [[-1., -1.]] * 5
    assert trace['original']['landmarks'] == [list(p) for p in original.landmarks]
    assert trace['result']['box'] == trace['original']['box']


def test_crop_points_must_also_fit_preserved_original_box():
    # Both boxes refer to the same head, but moving the whole crop prediction
    # left would put an otherwise valid landmark outside the retained box.
    original = checked_detection([70,60,60,100,72,85,100,86,90,100,88,125,102,125,.99])
    target = checked_detection([62,60,60,100,64,85,92,86,82,100,80,125,94,125,.99])
    model, _ = model_for('scrfd_10g', [original], [project(target, CROP, 0)])
    with audit.capture() as reports:
        assert model.detect(np.zeros((300, 300, 3), np.uint8)) == [original.without_landmarks()]
    attempt = reports[0]['faces'][0]['attempts'][0]
    assert attempt['candidates'][0]['rejection'] == 'invalid_original_box_alignment'


@pytest.mark.parametrize('ort', [False, True])
def test_yunet_raw_crop_path_does_not_recurse_and_updates_input_size(ort):
    model = object.__new__(YuNetDetector)
    model.config = SimpleNamespace(data=defaults())
    model.config.data['models']['detector'] = 'yunet'
    model.config.data['detection']['landmarks_from_facebox'] = True
    calls, sizes = [], []
    original = shifted(PROFILE, -7)
    answer = project(PROFILE, CROP, 0)
    def raw(data):
        calls.append(data.shape)
        return [original if len(calls) == 1 else answer]
    if ort:
        model.ort_detector = SimpleNamespace(detect=raw)
    else:
        model.ort_detector = None
        model.detector = SimpleNamespace(setInputSize=sizes.append,
            detect=lambda data: (None, [f.row for f in raw(data)]))
    with Image.new('RGB', (300, 230)) as image:
        assert model.detect(image)[0] == PROFILE
    assert calls == [(230, 300, 3), (150, 90, 3)]
    assert sizes == ([] if ort else [(300, 230), (90, 150)])


def test_boolean_config_tuner_and_cache_contracts(tmp_path):
    from stash_ai.recognition.config import plain
    from stash_ai.recognition.inference_contracts import face, detector as person_detector, pose
    from stash_ai.recognition.scanning import scan_identity
    from stash_ai.recognition.tune_editor import Settings
    original = load(root=tmp_path, device='cpu')
    assert original.data['detection']['landmarks_from_facebox'] is False
    settings = Settings(original)
    settings.edit(KEY, 'true')
    enabled = settings.effective()
    assert enabled.data['detection']['landmarks_from_facebox'] is True
    assert face(enabled) != face(original)
    assert scan_identity(enabled, 'ns') != scan_identity(original, 'ns')
    assert person_detector(enabled) == person_detector(original) and pose(enabled) == pose(original)
    # Off must reuse the existing inference and scan identities.
    legacy = SimpleNamespace(data=plain(original.data))
    legacy.data['detection'].pop('landmarks_from_facebox')
    assert face(legacy) == face(original)
    assert scan_identity(legacy, 'ns') == scan_identity(original, 'ns')
    settings.edit(KEY, 'false')
    assert face(settings.effective()) == face(original)
    for version, text in ((1, 'detection: {landmarks_from_facebox: true}'),
                          (2, 'face: {detector: {landmarks_from_facebox: true}}')):
        path = tmp_path/f'v{version}.yml'
        path.write_text(f'version: {version}\n{text}\n')
        assert load(path).data['detection']['landmarks_from_facebox'] is True


@pytest.mark.parametrize('value', [0, 1, None, 'true', 'false'])
def test_crop_flag_requires_a_real_boolean(tmp_path, value):
    with pytest.raises(ValueError, match='landmarks_from_facebox'):
        apply_edits(load(root=tmp_path), {KEY: value}, detection_preview=True)


def test_switch_invalidates_cached_points_without_reloading_other_inference(tmp_path):
    from stash_ai.recognition.inference_cache import Store, using
    model = object.__new__(YuNetDetector)
    model.config = SimpleNamespace(data=defaults()); model.provenance = {}
    calls = []
    def detect(images):
        calls.append(len(images))
        result = PROFILE if model.config.data['detection']['landmarks_from_facebox'] else shifted(PROFILE, -7)
        return [[result] for _ in images]
    model._detect_faces_many = detect
    with closing(Store(tmp_path/'cache.sqlite', 1024**2)) as store, using(store), Image.new('RGB', (300, 230)) as image:
        before = model.detect(image)
        model.config.data['detection']['landmarks_from_facebox'] = True
        assert model.detect(image) == [PROFILE] and before != [PROFILE]
        assert model.detect(image) == [PROFILE]
        model.config.data['detection']['landmarks_from_facebox'] = False
        assert model.detect(image) == before
    assert calls == [1, 1]
