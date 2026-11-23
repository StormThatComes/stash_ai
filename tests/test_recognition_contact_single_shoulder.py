"""A visible shoulder needs original head evidence and exclusive connected masks."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from stash_ai.recognition.contact_face_evidence import measure, retained
from tests.test_recognition_contact_upper_body import masked_profile, get_mask, set_mask


def single_shoulder(*, shoulder=5, angle=0, hips=False, two_eyes=False):
    face, own = masked_profile(angle=angle, hips=hips, two_eyes=two_eyes)
    rotation = np.array([[np.cos(angle),-np.sin(angle)], [np.sin(angle),np.cos(angle)]])
    mouth = [(np.array(p)-[160,160]) @ rotation.T+[160,160] for p in ((70,107),(90,107))]
    face.landmarks = [*face.landmarks, *(p.tolist() for p in mouth)]
    own['keypoints'][11-shoulder] = [-1.,-1.]  # The other shoulder, never synthesized.
    _, peer = masked_profile(220, angle)
    return face, own, peer


@pytest.mark.parametrize('shoulder',[5,6])
@pytest.mark.parametrize('angle',[0,np.pi/4,np.pi/2,np.pi,3*np.pi/2])
@pytest.mark.parametrize('hips,two_eyes',[(False,False),(True,False),(False,True)])
def test_one_shoulder_with_five_masked_landmarks_confirms_without_mutating_pose(shoulder,angle,hips,two_eyes):
    face,own,peer = single_shoulder(shoulder=shoulder,angle=angle,hips=hips,two_eyes=two_eyes)
    before = deepcopy([own,peer])
    proof = measure(face,own,[own,peer],face_id='face')
    assert proof['confirmed'], proof
    assert proof['reason'] == 'same_frame_single_shoulder'
    upper = proof['upper_body']
    assert upper['shoulder_mode'] == 'single'
    assert upper['shoulder_points'] == [shoulder]
    assert upper['face_landmark_mask_support'] == 1.
    assert upper['peers'][0]['reason'] == 'separated_upper_body_masks'
    assert json.loads(json.dumps(proof)) == proof
    assert [own,peer] == before


@pytest.mark.parametrize('change', ['one_original_eye_only','borrowed_ear','borrowed_shoulder','conflicted_shoulder',
    'occluded_shoulder','no_shoulder','missing_mouth','invalid_mouth','collapsed_face_axis',
    'landmark_outside_mask','mask_islands','broken_neck','stolen_shoulder','shoulder_above_head',
    'hip_above_head','masks_disabled','mask_missing','mask_conflict'])
def test_face_landmarks_alone_never_replace_independent_anatomy_or_mask_connection(change):
    face,own,peer = single_shoulder()
    if change=='one_original_eye_only': own['keypoints'][3] = [-1.,-1.]
    elif change=='borrowed_ear': own['pose_refinement']['face_prompt'] = [3,own['keypoints'][3]]
    elif change=='borrowed_shoulder': own['pose_refinement']['temporal_prompts'] = [[5,own['keypoints'][5]]]
    elif change=='conflicted_shoulder': own['pose_refinement']['joint_ownership_withheld'] = [5]
    elif change=='occluded_shoulder': own['keypoint_visibility'][5] = .1
    elif change=='no_shoulder': own['keypoints'][5] = [-1.,-1.]
    elif change=='missing_mouth': face.landmarks = face.landmarks[:3]
    elif change=='invalid_mouth': face.landmarks[3] = [float('nan'),107]
    elif change=='collapsed_face_axis': face.landmarks[3:5] = [[70,85],[90,85]]
    elif change=='landmark_outside_mask':
        mask = get_mask(own); mask[105:110,68:73] = False; set_mask(own,mask)
    elif change=='mask_islands':
        mask = np.zeros((320,320),bool)
        for point in [*face.landmarks, *(own['keypoints'][i] for i in (1,3,5))]:
            x,y = map(int,point); mask[y-2:y+3,x-2:x+3] = True
        set_mask(own,mask)
    elif change=='broken_neck':
        mask = get_mask(own); mask[110:115,:] = False; set_mask(own,mask)
    elif change=='stolen_shoulder': own['keypoints'][5] = peer['keypoints'][5]
    elif change=='shoulder_above_head':
        own['keypoints'][5] = [60,65]
        mask = get_mask(own); mask[55:125,45:115] = True; set_mask(own,mask)
    elif change=='hip_above_head': own['keypoints'][11] = [60,60]
    elif change=='masks_disabled': own['mask_usage'] = {'body_admission':False}
    elif change=='mask_missing': own['pose_refinement'].pop('overlap_mask')
    else: own['pose_refinement']['mask_conflict'] = True
    proof = measure(face,own,[own,peer],face_id='face')
    assert not proof['confirmed'], (change,proof)
    assert 'upper_body' in proof


@pytest.mark.parametrize('change',['same_head','same_face','foreign_mouth','peer_mask_missing','near_incomplete_peer'])
def test_single_shoulder_competes_against_all_faces_and_bodies(change):
    face,own,peer = single_shoulder()
    faces = [face]
    if change=='same_head': peer = deepcopy(own)
    elif change=='same_face': faces.append(deepcopy(face))
    elif change=='foreign_mouth':
        mask = get_mask(peer); mask[105:110,68:73] = True; set_mask(peer,mask)
    elif change=='peer_mask_missing': peer['pose_refinement'].pop('overlap_mask')
    else: peer['keypoints'] = [[-1.,-1.]]*17
    proof = measure(face,own,[own,peer],face_id='face',faces=faces)
    assert not proof['confirmed'], (change,proof)


def test_remote_incomplete_peer_does_not_block_single_shoulder_and_local_mask_proof():
    face,own,_ = single_shoulder()
    peer = dict(bounds=[140,160,320,320],keypoints=[[-1.,-1.]]*17)
    proof = measure(face,own,[own,peer],face_id='face')
    assert proof['confirmed']
    assert proof['upper_body']['peers'][0]['reason'] == 'separated_upper_body_box'


def test_single_shoulder_reason_and_diagnostics_survive_temporal_check():
    from stash_ai.recognition.contact_anchors import questionable_links
    from stash_ai.recognition.contact_diagnostics import describe
    from stash_ai.recognition.body_identity import defaults
    from tests.test_recognition_contact_confirmation import row
    from tests.test_contact_body_claims import body
    face,own,peer = single_shoulder()
    proof = measure(face,own,[own,peer],face_id='face')
    candidate = row(local=False)
    quality = json.loads(candidate['quality'])
    quality['quality']['contact_face_evidence'] = proof
    candidate['quality'] = json.dumps(quality)
    checks = {}
    scores = SimpleNamespace(face_similarity=np.dot)
    assert not questionable_links([candidate],defaults(),scores,evidence=checks)
    assert checks['contact']['reason'] == 'same_frame_single_shoulder'
    text = '\n'.join(describe(checks['contact']))
    assert 'Single-shoulder path' in text
    assert 'no shoulder is synthesized' in text
    assert 'Face landmarks inside own mask: 1.0000' in text
    assert questionable_links([candidate,body('rival',2,50,'B')],defaults(),scores,evidence=checks) == {'contact'}
    assert checks['contact']['reason'] == 'temporal_identity_conflict'


def test_single_shoulder_collection_version_refreshes_cached_proofs(tmp_path):
    from stash_ai.recognition.body_observations import _contract, contract, compatible_contracts
    from stash_ai.recognition.config import load
    config = load(root=tmp_path)
    old = _contract(config,'independent-person-v6-upper-body-contact',legacy_mask_overlap=False,
                    legacy_foreground_sharpness=False,legacy_drift_mask=False,legacy_mask_prompts=False)
    assert contract(config) != old and old in compatible_contracts(config)
    assert not retained({'face':'face'},dict(contact_face_evidence=dict(
        version='same-cue-head-torso-v2-upper-body',face='face',confirmed=True)))['confirmed']
