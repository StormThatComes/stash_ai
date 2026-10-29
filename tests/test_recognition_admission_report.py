"""Admission boundaries, provenance and per-image companions remain honest offline."""
from dataclasses import replace
import json
import sqlite3

import pytest
from PIL import Image

from stash_ai.recognition.admission_report import check
from stash_ai.recognition.config import plain
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.identity_report import export
from stash_ai.recognition.models import namespace
from stash_ai.recognition.scanning import scan_identity
from tests.test_recognition_identity_report import seed


def checks(admission, stage):
    group = next(g for g in admission['groups'] if g['stage'] == stage)
    return group, {c['parameter']: c for c in group['checks']}


@pytest.mark.parametrize('operator,measured,threshold,status,required', [
    ('>', .91, .91, 'fail', '<'), ('>', .91, .9, 'pass', '<'),
    ('>=', 72, 72, 'pass', '<='), ('>=', 72, 73, 'fail', '<='),
    ('<=', .2, .2, 'pass', '>='), ('<=', .2, .19, 'fail', '>='),
])
def test_threshold_boundaries_match_strict_detector_and_inclusive_quality_gates(operator, measured, threshold, status, required):
    result = check('parameter', measured, threshold, operator)
    assert result['status'] == status
    assert result['parameter_requirement'] == dict(operator=required, boundary=measured)
    assert result['margin'] >= 0 if status == 'pass' else result['margin'] <= 0


@pytest.mark.parametrize('measured', [None, True, '0.9', float('nan'), float('inf'), -1])
def test_missing_invalid_measurements_are_unknown_not_zero_or_pass(measured):
    result = check('parameter', measured, .8)
    assert result['status'] == 'unknown'
    assert result['measured'] is result['margin'] is result['parameter_requirement'] is None
    assert check('disabled', measured, None, enabled=False)['status'] == 'disabled'


def test_every_image_has_readable_panel_exact_sidecar_and_current_fallback(tmp_path):
    config, person, _ = seed(tmp_path, count=1)
    data = plain(config.data)
    data['detection']['min_face_pixels'] = 56
    data['detection']['max_pose_deviation'] = None
    config = replace(config, data=data)
    before = config.database.read_bytes()
    target = tmp_path/'report'
    report = export(config, target, identity_id=person)
    sample = report['samples'][0]
    group, rows = checks(sample['admission'], 'face_detection')
    assert group['configuration_source'] == 'current_configuration_only_historical_unavailable'
    assert rows['detection.min_face_pixels']['measured'] == 32
    assert rows['detection.min_face_pixels']['status'] == 'fail'
    assert rows['detection.min_face_pixels']['parameter_requirement'] == dict(operator='<=', boundary=32)
    assert rows['alignment_valid']['status'] == 'unknown'
    assert rows['detection.max_pose_deviation']['status'] == 'disabled'
    panel = sample['image']['thresholds']
    data = json.loads((target/panel['json_file']).read_text())
    assert data['groups'] == sample['admission']['groups']
    assert data['recorded_role'] == sample['assignment']['role'] == 'anchor'
    with Image.open(target/panel['file']) as image, Image.open(target/sample['image']['annotations']['file']) as original:
        assert image.width >= original.width and image.height > original.height
        with image.crop((16, 16, 16+original.width, 16+original.height)) as preview:
            assert preview.tobytes() == original.tobytes()
    assert report['summary']['members'] == report['summary']['pngs_exported'] == 1
    assert config.database.read_bytes() == before


def test_hash_verified_historical_scan_thresholds_do_not_claim_current_settings(tmp_path):
    config, person, _ = seed(tmp_path, count=1)
    old_data = plain(config.data)
    old_data['detection'].update(min_face_pixels=24, min_sharpness=1.)
    old = replace(config, data=old_data)
    with FaceDatabase(config.database) as state:
        state.start_run('scope', ['2'], old.snapshot())
        state.start_run('scope', ['2'], {'effective': {'broken': True}})
        state.connection.execute('UPDATE scans SET policy_sha=?', (scan_identity(old, namespace(config)),))
        state.connection.commit()
    result = export(config, tmp_path/'report', identity_id=person)
    sample = result['samples'][0]
    group, rows = checks(sample['admission'], 'face_detection')
    assert not sample['detection_policy_matches_current']
    assert group['configuration_source'] == 'retained_configuration_hash_verified'
    assert rows['detection.min_face_pixels']['threshold'] == 24
    assert rows['detection.min_face_pixels']['status'] == 'pass'
    assert checks(sample['admission'], 'face_anchor_quality')[0]['configuration_source'] == 'current_configuration'


def test_verified_current_policy_and_separate_anchor_auxiliary_visibility(tmp_path):
    from stash_ai.recognition.visibility import CONTRACT_ID
    config, person, _ = seed(tmp_path, count=1)
    data = plain(config.data)
    data['clustering'].update(min_anchor_visible_face_fraction=.8, min_auxiliary_visible_face_fraction=.5,
                              min_anchor_face_pixels=96, min_auxiliary_face_pixels=64)
    config = replace(config, data=data)
    with sqlite3.connect(config.database) as db:
        db.execute('UPDATE scans SET policy_sha=?', (scan_identity(config, namespace(config)),))
        quality = json.loads(db.execute('SELECT quality FROM observations').fetchone()[0])
        quality.update(pixels=72, visible_face_fraction=.7, visibility={'contract': CONTRACT_ID})
        db.execute('UPDATE observations SET quality=?', (json.dumps(quality),))
    sample = export(config, tmp_path/'report', identity_id=person)['samples'][0]
    assert checks(sample['admission'], 'face_detection')[0]['configuration_source'] == 'current_configuration_hash_verified'
    _, anchor = checks(sample['admission'], 'face_anchor_quality')
    _, auxiliary = checks(sample['admission'], 'face_auxiliary_quality')
    assert anchor['clustering.min_anchor_visible_face_fraction']['status'] == 'fail'
    assert anchor['clustering.min_anchor_face_pixels']['status'] == 'fail'
    assert auxiliary['clustering.min_auxiliary_visible_face_fraction']['status'] == 'pass'
    assert auxiliary['clustering.min_auxiliary_face_pixels']['status'] == 'pass'
    assert sample['assignment']['role'] == 'anchor'  # Never relabel the recorded decision.


def test_body_only_members_get_collection_and_anchor_checks_without_face_gates(tmp_path):
    from tests.test_recognition_body_integration import seeded, crop
    config, _, ids, people = seeded(tmp_path, vectors=[[1, 0]])
    config.data['sapiensid']['identities']['anchor_min_person_pixels'] = 128
    with FaceDatabase(config.database) as state:
        crop(state.connection, ids[0])
        for oid in ids[0]:
            state.connection.execute('UPDATE body_observations SET quality=? WHERE id=?', (json.dumps(dict(
                person=dict(bounds=[0, 0, 120, 240], score=.75),
                quality=dict(width=120, height=240, sharpness=20, keypoints=5, overlap=.1))), oid))
        state.connection.commit()
    before = config.database.read_bytes()
    sample = export(config, tmp_path/'report', identity_id=people[0])['samples'][0]
    admission = sample['admission']
    assert admission['route'] == 'independent'
    assert not any(g['stage'].startswith('face') for g in admission['groups'])
    _, pose = checks(admission, 'person_detection')
    assert pose['sapiensid.pose.score_threshold']['status'] == 'pass'
    assert pose['sapiensid.pose.keypoint_score_threshold']['status'] == 'unknown'
    group, collection = checks(admission, 'independent_person_collection')
    assert group['configuration_source'] == 'current_configuration_hash_verified'
    assert collection['sapiensid.collection.min_person_pixels']['status'] == 'pass'
    _, anchor = checks(admission, 'person_anchor_quality')
    assert anchor['sapiensid.identities.anchor_min_person_pixels']['status'] == 'fail'
    assert anchor['sapiensid.identities.anchor_min_sharpness']['status'] == 'fail'
    assert anchor['sapiensid.identities.anchor_min_keypoints']['status'] == 'fail'
    assert config.database.read_bytes() == before


def test_face_associated_body_does_not_apply_independent_collection_floors(tmp_path):
    from stash_ai.recognition.person_signal import save, contract
    config, person, blob = seed(tmp_path, count=1)
    with FaceDatabase(config.database) as state:
        db = state.connection
        oid = db.execute('SELECT id FROM observations').fetchone()[0]
        save(db, oid, contract(config), dict(person=dict(bounds=[0, 0, 20, 40], score=.8),
             quality=dict(sharpness=0, keypoints=1, overlap=1), identity_rejection=None), [1.]*4096, blob)
        db.commit()
    sample = export(config, tmp_path/'report', identity_id=person)['samples'][0]
    evidence = sample['person_evidence']
    assert evidence['admission']['route'] == 'face_associated'
    assert {g['stage'] for g in evidence['admission']['groups']} == {
        'person_detection', 'face_person_association', 'person_anchor_quality'}
    assert evidence['image']['thresholds']['status'] == 'exported'


def test_body_historical_collection_settings_are_verified_independently_of_current_anchor_gates(tmp_path):
    from tests.test_recognition_body_integration import seeded
    config, _, _, people = seeded(tmp_path, vectors=[[1, 0]])
    old_floor = config.data['sapiensid']['collection']['min_person_pixels']
    with FaceDatabase(config.database) as state:
        state.start_run('scope', ['1'], config.snapshot())
    config.data['sapiensid']['collection']['min_person_pixels'] = old_floor+100
    config.data['sapiensid']['identities']['anchor_min_person_pixels'] = 200
    sample = export(config, tmp_path/'report', identity_id=people[0])['samples'][0]
    group, rows = checks(sample['admission'], 'independent_person_collection')
    assert not sample['contract_matches_current']
    assert group['configuration_source'] == 'retained_configuration_hash_verified'
    assert rows['sapiensid.collection.min_person_pixels']['threshold'] == old_floor
    group, rows = checks(sample['admission'], 'person_anchor_quality')
    assert group['configuration_source'] == 'current_configuration'
    assert rows['sapiensid.identities.anchor_min_person_pixels']['threshold'] == 200


def test_missing_crop_keeps_json_and_failed_panel_write_aborts_export(tmp_path, monkeypatch):
    config, person, _ = seed(tmp_path, count=1)
    with sqlite3.connect(config.database) as db:
        db.execute('UPDATE observations SET crop=NULL')
    result = export(config, tmp_path/'missing', identity_id=person)
    panel = result['samples'][0]['image']['thresholds']
    assert panel['file'] is None and panel['status'] == 'missing_crop'
    assert (tmp_path/'missing'/panel['json_file']).exists()
    assert not result['summary']['complete']
    # A sidecar failure must propagate too, rather than report success.
    def fail(*args, **kwargs):
        raise OSError('threshold disk full')
    monkeypatch.setattr('stash_ai.recognition.admission_panel._write_json', fail)
    with pytest.raises(OSError, match='threshold disk full'):
        export(config, tmp_path/'failed', identity_id=person)
    assert not (tmp_path/'failed'/'manifest.json').exists()
