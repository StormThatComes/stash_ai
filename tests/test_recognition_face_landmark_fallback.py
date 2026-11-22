"""Borrowed head points affect regional rules, never ownership or model inputs."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest
import yaml

from stash_ai.recognition.config import load, plain
from stash_ai.recognition.face_landmark_fallback import defaults, body_face_sql
from stash_ai.recognition.person_anchor import anchor_gate, body_gate
from tests.test_recognition_pose_anchor_rules import metadata, settings


def evidence(missing=('nose',)):
    value = metadata(missing)
    for index, name, xy in ((0, 'nose', [50, 26]), (1, 'left_eye', [60, 18]), (2, 'right_eye', [40, 18])):
        if name not in missing:
            value['person']['keypoints'][index] = xy
    face = dict(observation='face-1', geometry=[25, 5, 50, 40, 40, 18, 60, 18, 50, 26, 40, 34, 60, 34, .99],
                quality=dict(alignment_valid=True, face_profile=False), rejected=None)
    return value, face


def policy(points=('nose',)):
    result = settings({'all': [{'points': list(points), 'min': len(points)}], 'any': []})
    result['anchor_face_landmark_fallback'] = dict(defaults(), enabled=True)
    return result


@pytest.mark.parametrize('point', ['nose', 'left_eye', 'right_eye'])
def test_missing_point_is_borrowed_without_changing_pose_or_encoder_count(point):
    value, face = evidence((point,))
    original = deepcopy((value, face))
    config = policy((point,))
    before = anchor_gate(value, config)
    after = anchor_gate(value, config, face_evidence=face)
    assert before['reason'] == 'body_anchor_keypoint_rules'
    assert after['eligible'] and after['keypoints'] == before['keypoints'] == 16
    rule, = after['keypoint_rules']['all']
    assert rule['point_sources'] == {point: 'face_detector'}
    recovered, = after['keypoint_rules']['face_landmark_fallback']['recovered']
    assert recovered['point'] == point and recovered['visibility'] is None
    assert (value, face) == original
    config['anchor_min_keypoints'] = 17
    assert anchor_gate(value, config, face_evidence=face)['reason'] == 'body_anchor_keypoints'


def test_two_eyes_can_be_recovered_from_original_nose_and_frontal_face():
    value, face = evidence(('left_eye', 'right_eye'))
    gate = anchor_gate(value, policy(('left_eye', 'right_eye')), face_evidence=face)
    assert gate['eligible'] and gate['keypoints'] == 15
    assert gate['keypoint_rules']['all'][0]['present'] == ['left_eye', 'right_eye']


@pytest.mark.parametrize('case,reason', [
    ('no_original_head', 'no_valid_head_keypoints'),
    ('wrong_face', 'head_landmark_mismatch'),
    ('rejected_face', 'linked_face_rejected'),
    ('weak_face', 'linked_face_score'),
    ('invalid_alignment', 'linked_face_alignment_invalid'),
    ('invalid_geometry', 'invalid_face_evidence'),
])
def test_invalid_or_self_confirming_association_never_fills_nose(case, reason):
    value, face = evidence()
    if case == 'no_original_head':
        value['person']['keypoints'][:3] = [[-1, -1]]*3
    elif case == 'wrong_face':
        face['geometry'][4] += 15
    elif case == 'rejected_face':
        face['rejected'] = 'face_occluded'
    elif case == 'weak_face':
        face['geometry'][-1] = .8
    elif case == 'invalid_alignment':
        face['quality']['alignment_valid'] = False
    else:
        face['geometry'][8] = float('nan')
    gate = anchor_gate(value, policy(), face_evidence=face)
    assert not gate['eligible']
    assert gate['keypoint_rules']['face_landmark_fallback']['reason'] == reason


@pytest.mark.parametrize('missing', [('left_eye',), ('right_eye',), ('left_eye', 'right_eye')])
def test_profile_eye_coordinates_do_not_prove_eye_visibility(missing):
    value, face = evidence(missing)
    face['quality']['face_profile'] = True
    gate = anchor_gate(value, policy(missing), face_evidence=face)
    assert not gate['eligible']
    assert set(gate['keypoint_rules']['face_landmark_fallback']['skipped'].values()) == {'eye_visibility_uncertain'}


def test_eye_pose_limit_is_recomputed_and_does_not_block_profile_nose():
    value, face = evidence(('nose', 'left_eye'))
    # Original right eye supports ownership, but a yawed face cannot supply an unseen eye.
    face['geometry'][8] = 62
    face['quality']['pose_deviation'] = 0  # Stale metadata must not bypass geometric checking.
    gate = anchor_gate(value, policy(('nose', 'left_eye')), face_evidence=face)
    assert not gate['eligible']
    assert gate['keypoint_rules']['all'][0]['present'] == ['nose']
    assert gate['keypoint_rules']['face_landmark_fallback']['skipped'] == {'left_eye': 'eye_visibility_uncertain'}


def test_narrow_eye_baseline_is_not_rescued_by_missing_profile_metadata():
    value, face = evidence(('left_eye', 'right_eye'))
    face['geometry'][4:8] = [49, 18, 51, 18]
    face['quality'].pop('face_profile')
    gate = anchor_gate(value, policy(('left_eye', 'right_eye')), face_evidence=face)
    assert not gate['eligible']
    assert set(gate['keypoint_rules']['face_landmark_fallback']['skipped'].values()) == {'eye_visibility_uncertain'}


@pytest.mark.parametrize('case,reason', [
    ('outside', 'outside_crop'), ('coincident', 'coincident_head_points'),
    ('occluded', 'explicit_pose_occlusion'), ('conflicted', 'point_ownership_conflict'),
    ('refinement_conflict', 'point_ownership_conflict'),
])
def test_each_borrowed_point_respects_crop_and_recorded_exclusions(case, reason):
    value, face = evidence()
    if case == 'outside':
        face['geometry'][8] = 100
    elif case == 'coincident':
        face['geometry'][8:10] = [40, 18]
    elif case in ('occluded', 'conflicted'):
        value['person']['keypoint_states'] = [case]+['measured']*16
    else:
        value['person']['pose_refinement'] = dict(joint_ownership_withheld=[0])
    gate = anchor_gate(value, policy(), face_evidence=face)
    assert not gate['eligible']
    assert gate['keypoint_rules']['face_landmark_fallback']['skipped'] == {'nose': reason}


def test_existing_points_disabled_fallback_and_corrected_contact_stay_original():
    value, face = evidence(())
    config = policy()
    gate = anchor_gate(value, config, face_evidence=face)
    assert gate['eligible']
    assert gate['keypoint_rules']['all'][0]['point_sources'] == {'nose': 'pose_detector'}
    value, face = evidence()
    row = dict(quality=json.dumps(value), anchor_face=json.dumps(face))
    assert body_gate(row, config)['eligible']
    assert not body_gate(dict(row, _contact_correction={'owner': 'other'}), config)['eligible']
    config['anchor_face_landmark_fallback']['enabled'] = False
    assert not body_gate(row, config)['eligible']
    config['anchor_face_landmark_fallback'].update(enabled=True, points=['left_eye'])
    assert not body_gate(row, config)['eligible']


@pytest.mark.parametrize('patch', [
    {'enabled': 1}, {'points': []}, {'points': ['nose', 'nose']}, {'points': ['left_ear']},
    {'points': 'nose'}, {'min_face_score': -1}, {'max_eye_pose_deviation': 2},
])
def test_invalid_public_configuration_is_rejected(tmp_path, patch):
    path = tmp_path/'config.yml'
    path.write_text(yaml.safe_dump(dict(version=2, matching=dict(body=dict(anchors=dict(face_landmark_fallback=patch))))))
    with pytest.raises(ValueError, match='matching.body.anchors.face_landmark_fallback'):
        load(path)


def test_fallback_changes_matching_policy_without_invalidating_stored_inputs(tmp_path):
    from stash_ai.recognition.clustering import matching_policy
    from stash_ai.recognition.person_signal import contract
    from stash_ai.recognition.body_observations import contract as body_contract
    from stash_ai.recognition.config_layout import v2_document
    config = load(root=tmp_path)
    data = plain(config.data)
    data['sapiensid']['enabled'] = True
    baseline = replace(config, data=deepcopy(data))
    data['sapiensid']['identities']['anchor_face_landmark_fallback']['enabled'] = True
    changed = replace(config, data=data)
    assert matching_policy(baseline) != matching_policy(changed)
    assert contract(baseline) == contract(changed) and body_contract(baseline) == body_contract(changed)
    assert v2_document(data)['matching']['body']['anchors']['face_landmark_fallback']['points'] == ['nose', 'left_eye', 'right_eye']


def retained_scene(tmp_path):
    from tests.test_recognition_body_identity import configuration, scene
    from stash_ai.recognition.database import FaceDatabase, encoded
    config = configuration(tmp_path)
    config.data['sapiensid']['identities'].update(policy())
    config.data['sapiensid']['identities']['enabled'] = True
    config.data['sapiensid']['identities']['min_appearance_distance'] = 0
    with FaceDatabase(config.database) as state:
        db = state.connection
        item, ids = scene(db, config)
        for cue, oid in enumerate(ids):
            value, face = evidence()
            value['quality']['pixel_digest'] = str(cue)
            fid = 'face-'+oid
            db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                (fid, item['scan'], cue, float(cue), '', '[]', encoded(face['geometry']),
                 encoded(face['quality']), None, 2, None, None))
            db.execute('UPDATE body_observations SET face=?,geometry=?,quality=? WHERE id=?',
                (fid, encoded(value['person']), encoded(value), oid))
        db.commit()
    return config, item, ids


def test_rebuild_reuses_retained_faces_and_preserves_inputs_and_contact_veto(tmp_path):
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.body_identity import scene_plan, rebuild
    config, item, ids = retained_scene(tmp_path)
    with FaceDatabase(config.database) as state:
        db = state.connection
        before = list(map(tuple, db.execute('SELECT * FROM body_observations ORDER BY id')))
        fallback = config.data['sapiensid']['identities']['anchor_face_landmark_fallback']
        fallback['enabled'] = False
        assert {d['reason'] for d in scene_plan(db, item['scan'], config)[2].values()} == {'body_anchor_keypoint_rules'}
        fallback['enabled'] = True
        rebuild(state, [item], config, {})
        assert db.execute("SELECT COUNT(*) FROM body_members WHERE role='anchor' AND status='accepted'").fetchone()[0] == 3
        assert list(map(tuple, db.execute('SELECT * FROM body_observations ORDER BY id'))) == before
        # A matched face in a crowded cue remains withheld without contact corroboration.
        db.execute("INSERT INTO people VALUES ('A','scope','ns',1,NULL,'face-A')")
        db.execute("INSERT INTO members SELECT face,'A' FROM body_observations")
        db.execute("UPDATE body_observations SET quality=json_set(quality,'$.quality.detected_people',2,'$.quality.overlap',0.8)")
        assert {d['reason'] for d in scene_plan(db, item['scan'], config)[2].values()} == {'unconfirmed_contact_face'}


def test_linked_face_in_another_cue_cannot_supply_evidence(tmp_path):
    from stash_ai.recognition.database import FaceDatabase
    config, item, ids = retained_scene(tmp_path)
    with FaceDatabase(config.database) as state:
        db = state.connection
        db.execute('UPDATE body_observations SET face=? WHERE id=?', ('face-'+ids[1], ids[0]))
        row = db.execute('SELECT b.quality,'+body_face_sql()+' FROM body_observations b WHERE b.id=?', (ids[0],)).fetchone()
        gate = body_gate(row, config.data['sapiensid']['identities'])
        assert not gate['eligible']
        assert gate['keypoint_rules']['face_landmark_fallback']['reason'] == 'linked_face_context_mismatch'


def test_hybrid_references_use_same_fallback_without_body_collection(tmp_path):
    from tests.test_recognition_hybrid_merge import prepared
    from stash_ai.recognition.database import FaceDatabase, encoded
    from stash_ai.recognition.hybrid_references import body_rows
    config, people = prepared(tmp_path)
    config.data['sapiensid']['identities'].update(policy())
    with FaceDatabase(config.database) as state:
        db = state.connection
        value, face = evidence()
        db.execute('UPDATE person_evidence SET metadata=?', (encoded(value),))
        db.execute('UPDATE observations SET geometry=?', (encoded(face['geometry']),))
        ns = db.execute('SELECT namespace FROM people LIMIT 1').fetchone()[0]
        assert body_rows(db, config, 'scope', ns, people[0])
        config.data['sapiensid']['identities']['anchor_face_landmark_fallback']['enabled'] = False
        assert not body_rows(db, config, 'scope', ns, people[0])


def test_admission_report_exposes_sources_and_coordinates(tmp_path):
    from types import SimpleNamespace
    from stash_ai.recognition.admission_report import person
    config = load(root=tmp_path)
    data = plain(config.data)
    data['sapiensid']['identities'].update(policy())
    config = replace(config, data=data)
    value, face = evidence()
    policies = SimpleNamespace(resolve=lambda *args: (config, 'current_configuration'))
    report = person(value, config, policies, '', independent=True, face_evidence=face)
    rule = next(g for g in report['groups'] if g['stage'] == 'person_anchor_keypoints_all')['checks'][0]
    assert rule['status'] == 'pass' and rule['point_sources'] == {'nose': 'face_detector'}
    assert 'face-1' in rule['note']
    assert report['anchor_gate']['keypoint_rules']['face_landmark_fallback']['recovered'][0]['coordinates'] == [50, 26]


def test_selected_body_overlay_marks_borrowed_rule_point_without_rewriting_pose(tmp_path):
    from PIL import Image
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.inspection_overlays import read, point_info, render
    config, item, ids = retained_scene(tmp_path)
    with FaceDatabase(config.database) as state:
        geometry = read(state.connection, config, dict(scope='scope', namespace='ns'), 'body', ids[0])
    body = geometry['shapes'][0]
    assert body['geometry']['keypoints'][0] == [-1, -1]
    assert body['rule_landmarks'][0]['point'] == 'nose'
    assert 'face_detector fallback' in point_info(geometry, (100, 240), 50, 26)
    with Image.new('RGB', (100, 240)) as image:
        drawn, caption = render(image, geometry, layers=dict(keypoints=True))
        assert 'orange squares' in caption and drawn.tobytes() != image.tobytes()
        drawn.close()


def test_face_landmark_change_invalidates_cached_scene_plan(tmp_path):
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.scene_artifacts import body_key
    from stash_ai.recognition.derived_cache import scope
    from stash_ai.recognition.body_scene_jobs import planned
    config, item, ids = retained_scene(tmp_path)
    with FaceDatabase(config.database) as state, scope(state, config):
        db = state.connection
        before = body_key(db, item['scan'], config)
        first = list(planned(db, [item], config))[0][2]
        assert all(d['status'] == 'accepted' for d in first.values())
        db.execute("UPDATE observations SET geometry=json_set(geometry,'$[8]',100)")
        assert body_key(db, item['scan'], config) != before
        second = list(planned(db, [item], config))[0][2]
        assert all(d['reason'] == 'body_anchor_keypoint_rules' for d in second.values())


def test_persisted_contact_correction_blocks_old_face_in_inspection(tmp_path):
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.body_identity import rebuild, scene_plan
    from stash_ai.recognition.face_landmark_fallback import retained_body_face
    from stash_ai.recognition.inspection_overlays import read
    config, item, ids = retained_scene(tmp_path)
    with FaceDatabase(config.database) as state:
        db = state.connection
        rebuild(state, [item], config, {})
        db.execute("UPDATE body_members SET proof=json_set(proof,'$.contact_correction',json(?))",
                   ('{"original_owner":"A","owner":"B"}',))
        face = json.loads(retained_body_face(db, ids[0], config.data['sapiensid']['identities']))
        assert face['observation'] == 'face-'+ids[0]
        assert face['unavailable_reason'] == 'linked_face_contact_corrected'
        geometry = read(db, config, dict(scope='scope', namespace='ns'), 'body', ids[0])
        assert not geometry['shapes'][0]['rule_landmarks']
        # A new scene plan re-evaluates corrections instead of trusting stale memberships.
        assert all(d['status'] == 'accepted' for d in scene_plan(db, item['scan'], config)[2].values())


@pytest.mark.parametrize('missing,profile,accepted,rejected', [
    ('nose', True, 3, None), ('left_eye', False, 3, None), ('left_eye', True, 0, None),
    ('nose', True, 3, 'face_too_small'), ('nose', False, 3, 'face_incomplete'),
    ('left_eye', False, 3, 'face_pose'), ('nose', False, 0, 'face_occluded'),
])
def test_offline_replay_uses_identical_fallback_and_profile_guards(tmp_path, missing, profile, accepted, rejected):
    import struct
    from tests.test_recognition_body_integration import seeded
    from stash_ai.recognition.database import FaceDatabase, encoded
    from stash_ai.recognition.fragmentation_report import _open_readonly
    from stash_ai.recognition.merge_goals import resolve, evaluate
    from stash_ai.recognition.body_identity import scene_plan
    from stash_ai.recognition.pose_policy import compact_quality
    config, selected, ids, _ = seeded(tmp_path, vectors=[[1, 0]])
    config.data['sapiensid']['identities'].update(policy((missing,)))
    config.data['sapiensid']['identities']['enabled'] = True
    with FaceDatabase(config.database) as state:
        db = state.connection
        for cue, oid in enumerate(ids[0]):
            value, face = evidence((missing,))
            value['quality']['pixel_digest'] = str(cue)
            face['quality']['face_profile'] = profile
            face['quality']['query_only'] = True  # Face detections must not independently seed an identity.
            assert json.loads(compact_quality(encoded(face['quality'])))['face_profile'] is profile
            fid = 'face-'+oid
            db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                (fid, selected[0]['scan'], cue, float(cue), '', '[]', encoded(face['geometry']),
                 encoded(face['quality']), rejected, 3, struct.pack('<3f', 1, 0, 0), None))
            db.execute('UPDATE body_observations SET face=?,geometry=?,quality=? WHERE id=?',
                (fid, encoded(value['person']), encoded(value), oid))
        db.commit()
        decisions = scene_plan(db, selected[0]['scan'], config)[2]
        assert sum(d['status'] == 'accepted' for d in decisions.values()) == accepted
    before = config.database.read_bytes()
    with _open_readonly(config.database) as db:
        goals = resolve(db, config, [['900']])
        result = evaluate(config, goals, db)['trials']['baseline']
        assert result['goal_result']['groups'][0]['members'][0]['assigned'] == accepted
    assert config.database.read_bytes() == before


@pytest.mark.parametrize('rejected', ['face_occluded', 'face_blurred', 'face_landmarks_invalid',
                                   'face_detection_score', 'face_visibility_unmeasurable', 'unknown_reason'])
def test_existing_rejected_face_reports_its_id_and_actual_blocker(rejected):
    value, face = evidence()
    face['rejected'] = rejected
    gate = anchor_gate(value, policy(), face_evidence=face)
    assert not gate['eligible']
    proof = gate['keypoint_rules']['face_landmark_fallback']
    assert proof['reason'] == 'linked_face_rejected'
    assert proof['face_observation'] == 'face-1' and proof['face_rejection'] == rejected
    assert not proof['recovered']


@pytest.mark.parametrize('rejected', ['face_too_small', 'face_incomplete', 'face_pose'])
def test_recognition_only_rejection_keeps_verified_detector_landmarks(rejected):
    value, face = evidence()
    face['rejected'] = rejected
    original = deepcopy(face)
    gate = anchor_gate(value, policy(), face_evidence=face)
    assert gate['eligible'] and gate['keypoints'] == 16
    proof = gate['keypoint_rules']['face_landmark_fallback']
    assert proof['face_rejection'] == rejected and proof['recovered'][0]['point'] == 'nose'
    assert face == original
    face['quality']['alignment_valid'] = False
    assert not anchor_gate(value, policy(), face_evidence=face)['eligible']


@pytest.mark.parametrize('missing', [('nose',), ('left_eye',)])
def test_pose_rejected_profile_still_cannot_supply_missing_eyes(missing):
    value, face = evidence(missing)
    face.update(rejected='face_pose')
    face['quality']['face_profile'] = True
    gate = anchor_gate(value, policy(missing), face_evidence=face)
    assert gate['eligible'] == (missing == ('nose',))


@pytest.mark.parametrize('link,reason', [(None, 'face_link_missing'), ('not-retained', 'linked_face_not_retained')])
def test_missing_face_link_and_missing_observation_have_distinct_reasons(tmp_path, link, reason):
    from stash_ai.recognition.database import FaceDatabase
    config, item, ids = retained_scene(tmp_path)
    with FaceDatabase(config.database) as state:
        db = state.connection
        # Project an absent source without violating the database's foreign key.
        row = db.execute('SELECT b.quality,'+body_face_sql()+
            ' FROM (SELECT id,scan,cue,quality,? AS face FROM body_observations WHERE id=?) b', (link, ids[0])).fetchone()
        gate = body_gate(row, config.data['sapiensid']['identities'])
        assert not gate['eligible']
        proof = gate['keypoint_rules']['face_landmark_fallback']
        assert proof['reason'] == reason and proof['face_observation'] == link


def test_inspector_shows_rejected_linked_face_and_why_fallback_refused_it(tmp_path):
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.body_identity import rebuild
    from stash_ai.recognition.scene_inspection import UnassignedReader
    from stash_ai.recognition.models import namespace
    from stash_ai.recognition.scanning import scan_identity
    config, item, ids = retained_scene(tmp_path)
    with FaceDatabase(config.database) as state:
        db = state.connection
        db.execute('UPDATE scans SET namespace=?,policy_sha=?', (namespace(config), scan_identity(config, namespace(config))))
        db.execute("UPDATE observations SET rejected='face_occluded'")
        rebuild(state, [item], config, {})
        scan = dict(db.execute('SELECT * FROM scans WHERE id=?', (item['scan'],)).fetchone())
        db.commit()
    detail = UnassignedReader(config, scan).detail('body', ids[0], pixels=False)
    gate = detail['admission']['anchor_gate']
    assert gate['reason'] == 'body_anchor_keypoint_rules'
    assert gate['keypoint_rules']['face_landmark_fallback']['reason'] == 'linked_face_rejected'
    checks = next(g for g in detail['admission']['groups'] if g['stage'] == 'person_anchor_keypoints_all')['checks']
    assert 'face_occluded' in checks[0]['note'] and 'face-'+ids[0] in checks[0]['note']
    assert 'unavailable' not in checks[0]['note']
    assert any(shape['kind'] == 'face' and shape['geometry'] for shape in detail['geometry']['shapes'])
    assert any(ref['observation'] == 'face-'+ids[0] for decision in detail['trace']['decisions'] for ref in decision['references'])


def test_admission_change_invalidates_cached_withheld_and_reuses_retained_geometry(tmp_path, monkeypatch):
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.body_scene_jobs import planned
    from stash_ai.recognition.scene_artifacts import body_key
    from stash_ai.recognition.derived_cache import scope
    from stash_ai.recognition import face_landmark_fallback as fallback
    config, item, ids = retained_scene(tmp_path)
    with FaceDatabase(config.database) as state, scope(state, config):
        db = state.connection
        db.execute("UPDATE observations SET rejected='face_too_small'")
        before = list(map(tuple, db.execute('SELECT * FROM observations ORDER BY id')))
        with monkeypatch.context() as old:
            old.setattr(fallback, 'GEOMETRY_ONLY_REJECTIONS', frozenset())
            old.setattr(fallback, 'POLICY_VERSION', 1)
            key = body_key(db, item['scan'], config)
            first = list(planned(db, [item], config))[0][2]
            assert all(d['reason'] == 'body_anchor_keypoint_rules' for d in first.values())
        assert body_key(db, item['scan'], config) != key
        second = list(planned(db, [item], config))[0][2]
        assert all(d['status'] == 'accepted' for d in second.values())
        assert list(map(tuple, db.execute('SELECT * FROM observations ORDER BY id'))) == before
