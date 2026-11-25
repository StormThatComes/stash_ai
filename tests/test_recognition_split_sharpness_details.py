"""Natural-edge regressions and honest explanations of unavailable components."""
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from benchmarks.benchmark_split_sharpness import samples
from stash_ai.recognition.blur_quality import measure
from stash_ai.recognition.sharpness_details import describe, summary, gate_note, decision_text
from stash_ai.recognition.admission_report import sharpness_checks
from tests.test_recognition_reblur_sharpness import scene


@pytest.mark.parametrize('kind', ['face', 'body'])
def test_equal_length_profiles_resolve_natural_edges_with_the_original_fit_limits(kind):
    _, _, pixels, mask = next(row for row in samples({'attitude'}) if row[0] == 'attitude-0-'+kind)
    quality = measure(pixels, mask)
    assert quality['sharpness_confidence'] > .8
    diagnostic = quality['sharpness_split_diagnostics']
    assert diagnostic['fit_residual'] < .2
    assert min(diagnostic['edge_counts']) >= 8 and min(diagnostic['tile_counts']) >= 2
    assert quality['sharpness_split_reason'] == 'estimated'
    json.dumps(quality, allow_nan=False)


def test_natural_crop_still_responds_to_added_focus_motion_and_mixed_blur():
    _, _, pixels, mask = next(row for row in samples({'attitude'}) if row[0] == 'attitude-0-body')
    sharp = measure(pixels, mask)
    focus = measure(cv2.GaussianBlur(pixels, (0, 0), 2), mask)
    motion = measure(cv2.blur(pixels, (9, 1)), mask)
    mixed = measure(cv2.blur(cv2.GaussianBlur(pixels, (0, 0), 2), (9, 1)), mask)
    assert all(q['sharpness_confidence'] >= .6 for q in (sharp, focus, motion, mixed))
    assert focus['focus_sharpness'] < sharp['focus_sharpness']*.6
    assert focus['motion_sharpness'] > 85
    assert motion['motion_sharpness'] < sharp['motion_sharpness']*.75
    assert mixed['focus_sharpness'] < 50 and mixed['motion_sharpness'] < 75


def test_inconsistent_natural_edges_keep_fallback_and_explain_the_fit_failure():
    _, _, pixels, mask = next(row for row in samples({'beach-solo'}) if row[1] == 'body')
    quality = measure(pixels, mask)
    assert quality['sharpness_confidence'] == 0
    assert quality['sharpness_split_reason'] == 'inconsistent_edge_widths'
    assert quality['sharpness_split_diagnostics']['fit_residual'] >= .3
    assert 'disagree with the blur model' in gate_note(quality, dict(sharpness_mode='split'))


@pytest.mark.parametrize('pixels,mask,reason', [
    (np.zeros((14, 14)), None, 'input_too_small'),
    (scene(), np.zeros((128, 128), bool), 'no_foreground_interior'),
    (np.full((112, 112), 128), None, 'insufficient_edge_support'),
    (np.tile(((np.arange(112)//8) % 2)*200, (112, 1)), None, 'insufficient_edge_support'),
])
def test_unavailable_measurements_retain_the_actual_reason(pixels, mask, reason):
    quality = measure(pixels, mask, diagnostics=False)
    assert quality['sharpness_split_reason'] == reason
    assert quality['sharpness_confidence'] == 0
    assert quality['focus_sharpness'] is quality['motion_sharpness'] is None
    assert 'unavailable' in '\n'.join(describe(summary(quality)))
    json.dumps(quality, allow_nan=False)


def test_details_explain_zero_confidence_and_the_effective_mode_of_each_stage():
    quality = measure(np.full((112, 112), 128))
    settings = dict(sharpness_mode='split', min_sharpness=10,
                    min_focus_sharpness=40, min_motion_sharpness=50)
    rows = sharpness_checks(quality, settings, 'detection.')
    assert [row['parameter'] for row in rows] == ['detection.min_sharpness']
    assert rows[0]['status'] == 'fail'
    assert 'Legacy fallback (split requested)' in rows[0]['note']
    assert 'Too few reliable edges' in rows[0]['note']
    for measured in (quality, measure(scene())):
        note = gate_note(measured, dict(settings, sharpness_mode='legacy'))
        assert 'Legacy mode' in note and 'fallback' not in note
    text = '\n'.join(describe(summary(quality)))
    assert 'Focus sharpness (isotropic softness): unavailable' in text
    assert 'Split confidence: 0.000' in text
    assert 'not zero sharpness' in text
    assert 'requires >=8 edges and >=2 tiles' in text
    assert '0 deg: 0 / 0; 90 deg: 0 / 0; 45 deg: 0 / 0; 135 deg: 0 / 0' in text


def test_old_measurements_do_not_invent_the_cause_or_report_zero_components():
    text = '\n'.join(describe(summary(dict(sharpness=50, sharpness_confidence=0,
                                          sharpness_split_method='edge-spread-covariance-v1'))))
    assert 'Exact cause was not retained' in text
    assert 'Older or missing split measurement' in text
    assert 'Motion sharpness (directional smear): unavailable' in text
    assert 'Too few reliable edges' not in text


def test_split_components_remain_visible_when_a_stage_uses_legacy():
    quality = measure(scene())
    settings = dict(sharpness_mode='legacy', min_sharpness=10)
    rows = sharpness_checks(quality, settings, 'detection.')
    from stash_ai.recognition.identity_window import describe_diagnostics
    text = describe_diagnostics(dict(rejection=None, scan_status='complete', admission=dict(
        sharpness=summary(quality), groups=[dict(stage='face_detection', status='pass',
            configuration_source='current_configuration', checks=rows)])))
    assert 'Focus sharpness (isotropic softness): unavailable' not in text
    assert 'Motion sharpness (directional smear): unavailable' not in text
    assert 'Legacy mode' in text and 'Legacy fallback' not in text


@pytest.mark.parametrize('kind', ['face', 'body'])
def test_admission_and_export_retain_diagnostics_separately_from_advice_gates(tmp_path, kind):
    from stash_ai.recognition.config import load
    from stash_ai.recognition.admission_panel import export
    from tests.test_recognition_identity_inspector import advice_admission
    path = tmp_path/'config.yml'
    path.write_text('version: 2\n')
    config = load(path)
    quality = measure(scene())
    admission = advice_admission(config, kind, **quality)
    assert admission['sharpness']['sharpness_split_diagnostics'] == quality['sharpness_split_diagnostics']
    assert not any('confidence' in row['parameter'] or 'residual' in row['parameter']
                   for group in admission['groups'] for row in group['checks'])
    result = export(dict(status='missing_crop'), admission, tmp_path, 'sample.png',
                    observation='test', role='anchor', rejection=None)
    saved = json.loads((tmp_path/result['json_file']).read_text())
    assert saved['sharpness'] == admission['sharpness']
    text = decision_text(admission)
    assert 'Legacy mode' in text and 'Settings verified against retained policy' in text
    assert ('face.quality.min_sharpness' if kind == 'face' else 'body.collection.min_sharpness') in text
    assert 'required < 0.30' in text


def test_equal_span_contract_refreshes_unmasked_faces_and_preserves_v11_body_bank(monkeypatch):
    from stash_ai.recognition import scanning
    from stash_ai.recognition.config import defaults, digest
    from stash_ai.recognition.person_signal import contract as person_contract
    from stash_ai.recognition.body_observations import contract, compatible_contracts
    config = SimpleNamespace(data=defaults())
    config.data['sapiensid']['enabled'] = True
    collection = {k: v for k, v in config.data['sapiensid']['collection'].items() if k != 'enabled'}
    previous = digest(['independent-person-v11-split-sharpness',
                       person_contract(config, legacy_split_profiles=True), collection])
    assert previous in compatible_contracts(config)[1:] and previous != contract(config)
    monkeypatch.setattr(scanning, 'measurement_required', lambda *_: False)
    captured = []
    monkeypatch.setattr(scanning, 'digest', lambda payload: captured.append(payload))
    scanning.scan_identity(config, 'test')
    assert captured[0]['visibility_contract'] is None
    assert captured[0]['quality_format'] > 12
