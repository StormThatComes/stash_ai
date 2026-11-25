"""Independent blur responses and identical admission on scans/replay/inspection."""
import json
import sqlite3
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from stash_ai.recognition.blur_quality import measure
from stash_ai.recognition.sharpness_gates import failure, sql
from tests.test_recognition_reblur_sharpness import scene


def test_focus_motion_and_mixed_blur_have_separate_responses():
    pixels = scene()
    sharp = measure(pixels)
    focus = measure(cv2.GaussianBlur(pixels, (0, 0), 2))
    motion = measure(cv2.blur(pixels, (9, 1)))
    mixed = measure(cv2.blur(cv2.GaussianBlur(pixels, (0, 0), 2), (9, 1)))
    assert focus['focus_sharpness'] < sharp['focus_sharpness']*.6
    assert focus['motion_sharpness'] > 80
    assert motion['motion_sharpness'] < sharp['motion_sharpness']*.6
    assert motion['focus_sharpness'] > 80
    assert mixed['focus_sharpness'] < 60 and mixed['motion_sharpness'] < 60
    assert all(q['sharpness_confidence'] >= .6 for q in (sharp, focus, motion, mixed))


@pytest.mark.parametrize('turns', [0, 1, 2, 3])
def test_motion_proxy_is_invariant_to_quarter_turns(turns):
    pixels = cv2.blur(scene(), (9, 1))
    before, after = measure(pixels), measure(np.rot90(pixels, turns))
    assert after['motion_sharpness'] == pytest.approx(before['motion_sharpness'], abs=8)
    assert after['focus_sharpness'] > 75


@pytest.mark.parametrize('image', [np.full((112,112),128,np.float32),
    np.tile(((np.arange(112)//8)%2).astype(np.float32)*200,(112,1)),
    np.random.default_rng(9).normal(128,5,(112,112)).astype(np.float32),
    np.full((14,14),128,np.float32)])
def test_ambiguous_inputs_do_not_invent_perfect_components(image):
    quality = measure(image)
    assert quality['focus_sharpness'] is None and quality['motion_sharpness'] is None
    assert quality['sharpness_split_status'] == 'unavailable'


def test_same_crop_with_compression_and_noise_does_not_become_perfect():
    pixels = cv2.blur(cv2.GaussianBlur(scene(), (0,0), 1), (9,1))
    noisy = pixels + np.random.default_rng(8).normal(0, 3, pixels.shape)
    _, jpg = cv2.imencode('.jpg', noisy.clip(0,255).astype('uint8'), [cv2.IMWRITE_JPEG_QUALITY, 70])
    q = measure(cv2.imdecode(jpg, cv2.IMREAD_GRAYSCALE))
    assert q['motion_sharpness'] is None or q['motion_sharpness'] < 65


@pytest.mark.parametrize('confidence', [0, .59, .6, 1, 2, None])
@pytest.mark.parametrize('motion', [None, 0., 30., 80.])
def test_sql_and_python_use_the_same_fallback(confidence, motion):
    quality = dict(sharpness=12., focus_sharpness=80., motion_sharpness=motion, sharpness_confidence=confidence)
    settings = dict(sharpness_mode='split', min_sharpness=15., min_focus_sharpness=40., min_motion_sharpness=30.)
    with sqlite3.connect(':memory:') as db:
        db.execute('CREATE TABLE observations(quality TEXT)')
        db.execute('INSERT INTO observations VALUES (?)', (json.dumps(quality),))
        passed = db.execute('SELECT '+sql('observations', settings)+' FROM observations').fetchone()[0]
    assert bool(passed) == (failure(quality, settings) is None)


def test_split_replaces_legacy_and_reports_exact_failed_component():
    from stash_ai.recognition.quality import blur_rejection
    from stash_ai.recognition.admission_report import sharpness_checks
    q = dict(sharpness=1., focus_sharpness=85., motion_sharpness=45., sharpness_confidence=.9)
    settings = dict(sharpness_mode='split', min_sharpness=90., min_focus_sharpness=70., min_motion_sharpness=40.)
    assert failure(q, settings) is None
    assert blur_rejection(q, dict(settings, min_motion_sharpness=50)) == 'face_motion_sharpness'
    rows = sharpness_checks(q, settings, 'detection.')
    assert all(row['status']=='pass' for row in rows)
    assert {r['parameter'] for r in rows} == {'detection.min_focus_sharpness','detection.min_motion_sharpness'}
    assert failure(dict(q, sharpness_confidence=0.), settings) == 'sharpness'


def test_retained_face_roles_and_body_anchor_use_split_measurements():
    from stash_ai.recognition.config import defaults
    from stash_ai.recognition.pose_policy import compact_quality, quality_allows
    from stash_ai.recognition.person_anchor import anchor_gate
    data = defaults()
    q = dict(sharpness=1., focus_sharpness=85., motion_sharpness=45., sharpness_confidence=.9, keypoints=10)
    data['clustering'].update(anchor_sharpness_mode='split', anchor_min_motion_sharpness=40.,
                              min_anchor_sharpness=90., min_auxiliary_sharpness=0.)
    compact = compact_quality(json.dumps(q))
    assert quality_allows(compact, data['clustering'], anchor=True)
    data['clustering']['anchor_min_motion_sharpness'] = 50.
    assert not quality_allows(compact, data['clustering'], anchor=True)
    settings = data['sapiensid']['identities']
    settings.update(anchor_sharpness_mode='split', anchor_min_motion_sharpness=50., anchor_min_sharpness=90.)
    assert anchor_gate(dict(quality=q), settings)['reason'] == 'body_anchor_motion_sharpness'
    settings['anchor_min_motion_sharpness'] = 40.
    assert anchor_gate(dict(quality=q), settings)['eligible']


def test_split_configuration_round_trip_validation_and_mask_requirements():
    from stash_ai.recognition.config import defaults, effective_data
    from stash_ai.recognition.config_layout import v2_document
    data = defaults()
    data['detection'].update(sharpness_mode='split', min_focus_sharpness=40., min_motion_sharpness=60.)
    _, loaded = effective_data(v2_document(data))
    assert loaded['detection'] == data['detection']
    for invalid in (True, float('nan'), -1, 101, None):
        document = v2_document(data)
        document['face']['quality']['min_motion_sharpness'] = invalid
        with pytest.raises(ValueError):effective_data(document)
    data['detection'].update(min_sharpness=0., min_visible_face_fraction=0.)
    with pytest.raises(ValueError, match='FaceOcc'):effective_data(v2_document(data))


def test_previous_body_contract_is_frozen_and_new_policy_invalidates_matching():
    from stash_ai.recognition.config import defaults, digest
    from stash_ai.recognition.person_signal import contract as person_contract
    from stash_ai.recognition.body_observations import contract, compatible_contracts
    from stash_ai.recognition.clustering import matching_policy
    data = defaults(); data['sapiensid']['enabled'] = True
    config = SimpleNamespace(data=data)
    old_collection = {k:v for k,v in data['sapiensid']['collection'].items()
                      if k not in ('enabled','sharpness_mode','min_focus_sharpness','min_motion_sharpness')}
    old = digest(['independent-person-v10-model-input-sharpness',
                  person_contract(config, legacy_split_sharpness=True), old_collection])
    assert old in compatible_contracts(config)[1:] and old != contract(config)
    before = matching_policy(config)
    data['sapiensid']['identities']['anchor_min_focus_sharpness'] = 60.
    assert before != matching_policy(config)


def test_tuner_edits_and_preview_expose_measurements_without_new_models(tmp_path):
    from PIL import Image
    from stash_ai.recognition.config import load
    from stash_ai.recognition.detection_parameters import apply_edits
    from stash_ai.recognition.tune_sharpness import preview, caption
    from tests.helpers.recognition import FACE
    path = tmp_path/'config.yml'; path.write_text('version: 2\n')
    config = load(path)
    config, _ = apply_edits(config, {'face.quality.sharpness_mode':'split',
                                    'face.quality.min_focus_sharpness':40.}, detection_preview=True)
    with Image.fromarray(scene().astype('uint8')).convert('RGB') as image:
        rows = preview(image, [FACE], [], config.data)
    assert len(rows)==1 and rows[0]['quality']['sharpness_input_size']==[112,112]
    assert 'FaceOcc not applied' in caption(rows)
    assert 'focus ' in caption(rows) and 'motion ' in caption(rows)
    legacy, _ = apply_edits(config, {'face.quality.sharpness_mode':'legacy'}, detection_preview=True)
    assert preview(None, [FACE], [], legacy.data) == []


def test_scan_does_not_reject_split_before_preparing_encoder_pixels():
    from stash_ai.recognition.config import defaults
    from stash_ai.recognition.scan_pipeline import prepare_record
    from stash_ai.recognition.quality import blur_rejection
    data = defaults()
    data['detection'].update(sharpness_mode='split', min_motion_sharpness=80., min_sharpness=90.)
    quality = dict(sharpness=0., focus_sharpness=10., motion_sharpness=0., sharpness_confidence=1.)
    row = prepare_record(None, (0, SimpleNamespace(score=1.)), SimpleNamespace(data=data), None,
                         lambda *_: quality.copy(), blur_rejection, lambda *_:'test')
    assert row['reason'] is None
    assert blur_rejection(quality, data['detection']) == 'face_motion_sharpness'


def test_check_config_recognizes_a_split_anchor_quality_band(tmp_path):
    from stash_ai.recognition.config import load
    from stash_ai.recognition.config_check import report
    path = tmp_path/'config.yml'
    path.write_text('version: 2\nmatching:\n  face:\n    anchors:\n      quality:\n'
                    '        sharpness_mode: split\n        min_motion_sharpness: 60\n')
    checks = {r['rule']:r for r in report(load(path))['checks']}
    assert checks['face-auxiliary-quality-band']['status'] == 'ok'
