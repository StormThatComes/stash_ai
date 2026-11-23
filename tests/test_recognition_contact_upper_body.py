"""Profile/partial-body ownership needs independent anatomy and exclusive masks."""
import base64
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
import pytest

from stash_ai.recognition.contact_face_evidence import measure, retained
from tests.test_recognition_contact_confirmation import sample, row


def masked_profile(x=80, angle=0, *, hips=False, two_eyes=False):
    face, person = sample(x, angle)
    matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    person['keypoints'][3] = ((np.array([x+25,90])-[160,160]) @ matrix.T+[160,160]).tolist()
    person['keypoints'][0] = [-1.,-1.]
    if not two_eyes:
        person['keypoints'][2] = [-1.,-1.]
    if not hips:
        person['keypoints'][11:13] = [[-1.,-1.],[-1.,-1.]]
    # A broad silhouette independent of the production corridor sampler.
    canvas = Image.new('1', (320,320))
    draw = ImageDraw.Draw(canvas)
    draw.ellipse((x-22,65,x+32,110), fill=1)
    draw.rectangle((x-12,90,x+12,135), fill=1)
    draw.rectangle((x-28,112,x+28,195), fill=1)
    canvas = canvas.rotate(-np.degrees(angle), center=(160,160))
    set_mask(person, np.asarray(canvas, dtype=bool))
    return face, person


def set_mask(person, mask):
    person.setdefault('pose_refinement', {}).update(status='masked', mask_valid=True,
        bounds=list(person['bounds']),
        overlap_mask=base64.b64encode(np.packbits(mask).tobytes()).decode())


def get_mask(person):
    return np.unpackbits(np.frombuffer(base64.b64decode(person['pose_refinement']['overlap_mask']),
                        np.uint8))[:320*320].reshape(320,320).astype(bool)


@pytest.mark.parametrize('angle', [0,np.pi/4,np.pi/2,np.pi,3*np.pi/2])
@pytest.mark.parametrize('hips,two_eyes', [(False,False),(True,False),(False,True)])
def test_profiles_and_upper_bodies_confirm_without_neighbours(angle, hips, two_eyes):
    face, own = masked_profile(angle=angle, hips=hips, two_eyes=two_eyes)
    _, peer = masked_profile(220, angle)
    before = deepcopy([own,peer])
    proof = measure(face,own,[own,peer],face_id='face')
    assert proof['confirmed'], proof
    assert proof['reason'] == 'same_frame_upper_body'
    assert proof['upper_body']['peers'][0]['reason'] == 'separated_upper_body_masks'
    assert [own,peer] == before
    assert json.loads(json.dumps(proof)) == proof


def test_remote_box_intersection_does_not_require_a_complete_peer():
    face, own = masked_profile()
    peer = dict(bounds=[140,160,320,320], keypoints=[[-1.,-1.]]*17)
    proof = measure(face,own,[own,peer],face_id='face')
    assert proof['confirmed']
    assert proof['upper_body']['peers'][0]['local_box_separated']
    peer['bounds'] = [40,60,320,320]
    proof = measure(face,own,[own,peer],face_id='face')
    assert not proof['confirmed']
    assert proof['upper_body']['peers'][0]['reason'] == 'insufficient_peer_upper_body'


def test_right_eye_and_right_ear_use_the_same_confirmation_path():
    face, own = masked_profile(); _,peer = masked_profile(220)
    own['keypoints'][1] = own['keypoints'][3] = [-1.,-1.]
    own['keypoints'][2], own['keypoints'][4] = [70.,85.], [55.,90.]
    mask = get_mask(own); mask[65:110,48:105] = True; set_mask(own,mask)
    proof = measure(face,own,[own,peer],face_id='face')
    assert proof['confirmed']
    assert proof['upper_body']['ear_points'] == [4]


def test_crop_local_masks_and_peer_order_do_not_change_ownership():
    face, own = masked_profile(); _,peer = masked_profile(220)
    for person,bounds in ((own,[40,55,240,220]),(peer,[60,55,280,220])):
        mask = get_mask(person)
        person['bounds'] = bounds
        left,top,right,bottom = bounds
        set_mask(person,mask[top:bottom,left:right])
    for people in ([own,peer],[peer,own]):
        proof = measure(face,own,people,face_id='face')
        assert proof['confirmed']
        assert proof['upper_body']['peers'][0]['reason'] == 'separated_upper_body_masks'
    # A third, incomplete nearby detection cannot disappear from competition.
    third = dict(bounds=[50,60,120,150],keypoints=[[-1.,-1.]]*17)
    assert not measure(face,own,[own,peer,third],face_id='face')['confirmed']


def test_mask_contact_below_shoulders_does_not_block_exclusive_heads():
    face, own = masked_profile(hips=True)
    _, peer = masked_profile(220, hips=True)
    for person in (own,peer):
        mask = get_mask(person)
        mask[160:210,60:250] = True
        set_mask(person,mask)
    proof = measure(face,own,[own,peer],face_id='face')
    assert proof['confirmed']


@pytest.mark.parametrize('change', ['missing_ear','occluded_ear','borrowed_ear','temporal_ear',
    'borrowed_eye','borrowed_shoulder','coincident_ear','distant_ear','missing_shoulder',
    'collapsed_shoulders','stolen_head','contradictory_nose','conflicting_hips',
    'mask_missing','mask_conflict','mask_islands','mask_neck_gap','disabled_overlap','disabled_admission',
    'ear_conflicted','ear_state_occluded','ear_ownership_withheld','ear_near_peer'])
def test_partial_body_never_self_confirms_from_weak_or_conflicting_evidence(change):
    face, own = masked_profile()
    _, peer = masked_profile(220)
    if change=='missing_ear': own['keypoints'][3] = [-1.,-1.]
    elif change=='occluded_ear': own['keypoint_visibility'][3] = .1
    elif change=='borrowed_ear': own['pose_refinement']['face_prompt'] = [3,own['keypoints'][3]]
    elif change=='temporal_ear': own['pose_refinement']['temporal_prompts'] = [[3,own['keypoints'][3]]]
    elif change=='borrowed_eye': own['pose_refinement']['face_prompt'] = [1,own['keypoints'][1]]
    elif change=='borrowed_shoulder': own['pose_refinement']['temporal_prompts'] = [[5,own['keypoints'][5]]]
    elif change=='coincident_ear': own['keypoints'][3] = own['keypoints'][1]
    elif change=='distant_ear': own['keypoints'][3] = [160,90]
    elif change=='missing_shoulder': own['keypoints'][6] = [-1.,-1.]
    elif change=='collapsed_shoulders': own['keypoints'][6] = own['keypoints'][5]
    elif change=='stolen_head': own['keypoints'][5:7] = deepcopy(peer['keypoints'][5:7])
    elif change=='contradictory_nose': own['keypoints'][0] = [130,100]
    elif change=='conflicting_hips': own['keypoints'][11:13] = [[70,70],[90,70]]
    elif change in ('ear_conflicted','ear_state_occluded'):
        own['keypoint_states'] = ['measured']*17
        own['keypoint_states'][3] = 'conflicted' if change=='ear_conflicted' else 'occluded'
    elif change=='ear_ownership_withheld': own['pose_refinement']['joint_ownership_withheld'] = [3]
    elif change=='ear_near_peer': own['pose_refinement']['point_exclusions'] = {'near_peer':1 << 3}
    elif change=='mask_missing': own['pose_refinement'].pop('overlap_mask')
    elif change=='mask_conflict': own['pose_refinement']['mask_conflict'] = True
    elif change=='mask_islands':
        mask = np.zeros((320,320),bool)
        for i in (1,3,5,6):
            x,y = map(int,own['keypoints'][i]); mask[y-2:y+3,x-2:x+3] = True
        set_mask(own,mask)
    elif change=='mask_neck_gap':
        mask = get_mask(own); mask[106:114,:] = False; set_mask(own,mask)
    else:
        own['mask_usage'] = {'overlap' if change=='disabled_overlap' else 'body_admission':False}
    proof = measure(face,own,[own,peer],face_id='face')
    assert not proof['confirmed'], (change,proof)
    assert 'upper_body' in proof and len(proof['upper_body']['peers']) == 1


@pytest.mark.parametrize('change', ['same_head','same_face','foreign_head','shared_mask',
                                   'peer_mask_missing','peer_mask_disabled','peer_mask_invalid','peer_mask_islands'])
def test_competing_claims_and_unreliable_peer_masks_veto_confirmation(change):
    face, own = masked_profile()
    other_face, peer = masked_profile(220)
    faces = [face,other_face]
    if change=='same_head': peer = deepcopy(own)
    elif change=='same_face': faces.append(deepcopy(face))
    elif change=='foreign_head':
        mask = get_mask(peer); mask[80:100,80:110] = True; set_mask(peer,mask)
    elif change=='shared_mask':
        mask = get_mask(own) | get_mask(peer)
        set_mask(own,mask); set_mask(peer,mask)
    elif change=='peer_mask_missing': peer['pose_refinement'].pop('overlap_mask')
    elif change=='peer_mask_disabled': peer['mask_usage'] = {'body_admission':False}
    elif change=='peer_mask_invalid': peer['pose_refinement']['mask_valid'] = False
    else:
        mask = np.zeros((320,320),bool)
        for i in (1,3,5,6):
            x,y = map(int,peer['keypoints'][i]); mask[y-2:y+3,x-2:x+3] = True
        set_mask(peer,mask)
    proof = measure(face,own,[own,peer],face_id='face',faces=faces)
    assert not proof['confirmed'], (change,proof)


def test_original_report_error_is_accepted_and_temporal_conflict_still_vetoes():
    from stash_ai.recognition.contact_anchors import questionable_links
    from stash_ai.recognition.body_identity import defaults
    from tests.test_contact_body_claims import body
    face, own = masked_profile(); _,peer = masked_profile(220)
    own['keypoints'][1] = [92.036,85.]
    proof = measure(face,own,[own,peer],face_id='face')
    assert proof['head_error'] == pytest.approx(.0509)
    candidate = row(local=False)
    data = json.loads(candidate['quality'])
    data['quality'].update(contact_face_evidence=proof,overlap=0.,box_overlap=.2604)
    candidate['quality'] = json.dumps(data)
    checks = {}
    scores = SimpleNamespace(face_similarity=np.dot)
    assert not questionable_links([candidate],defaults(),scores,evidence=checks)
    assert checks['contact']['reason'] == 'same_frame_upper_body'
    rival = body('rival',2,50,'B')
    assert questionable_links([candidate,rival],defaults(),scores,evidence=checks) == {'contact'}
    assert checks['contact']['reason'] == 'temporal_identity_conflict'


def test_both_path_diagnostics_survive_an_early_head_failure():
    from stash_ai.recognition.contact_diagnostics import describe
    face, own = masked_profile(); _,peer = masked_profile(220)
    own['keypoints'][3] = [-1.,-1.]
    proof = measure(face,own,[own,peer],face_id='face')
    upper = proof['upper_body']
    assert not proof['confirmed']
    assert upper['checks']['head'] == 'insufficient_head_points'
    assert upper['checks']['shoulders'] == 'confirmed'
    assert upper['mask']['confirmed']
    text = '\n'.join(describe(dict(status='withheld',reason='no_clean_neighbour',
        crowding=dict(detected_people=2,overlap=0.,box_overlap=.2604,threshold=.25),same_frame=proof,
        temporal=dict(reason='no_clean_neighbour',threshold=.2385,margin=.002,neighbours=[]))))
    assert 'Full head/torso path: insufficient_head_points' in text
    assert 'shoulders: left_shoulder, right_shoulder' in text
    assert 'Own mask anatomy/corridor support' in text
    assert 'separated_upper_body_masks' in text


def test_new_evidence_contract_keeps_previous_version_only_as_competitor(tmp_path):
    from stash_ai.recognition.body_observations import _contract,contract,compatible_contracts
    from stash_ai.recognition.config import load
    config = load(root=tmp_path)
    previous = _contract(config,'independent-person-v5-contact-evidence',legacy_mask_overlap=False,
                         legacy_foreground_sharpness=False,legacy_drift_mask=False,legacy_mask_prompts=False)
    assert contract(config) != previous and previous in compatible_contracts(config)
    assert not retained({'face':'face'},dict(contact_face_evidence=dict(
        version='same-cue-head-torso-v1',face='face',confirmed=True)))['confirmed']


@pytest.mark.parametrize('single', [False, True])
def test_collection_retains_upper_body_peer_references_and_inspector_resolves_them(tmp_path, single):
    import uuid
    from stash_ai.recognition.body_collection import collect
    from stash_ai.recognition.config import load
    from stash_ai.recognition.assignment_explanation import references
    face, own = masked_profile(); _,peer = masked_profile(220)
    if single:
        from tests.test_recognition_contact_single_shoulder import single_shoulder
        face, own, peer = single_shoulder()
    captured = []
    pipeline = SimpleNamespace(collect_items=lambda db,image,items:captured.extend(items))
    collect(pipeline,None,SimpleNamespace(person_geometry=lambda *args:None),load(root=tmp_path),
            str(uuid.uuid4()),0,0.,None,[dict(id='face',rejected=None)],[face],
            captured=([own,peer],[0],[(dict(width=320,height=320,keypoints=4),None) for _ in range(2)]))
    proof = captured[0][0][1]['quality']['contact_face_evidence']
    peer_id = captured[1][0][0]
    assert proof['confirmed']
    assert proof['upper_body']['peers'][0]['observation'] == peer_id
    refs = references(dict(contact_confirmation=dict(same_frame=proof)), 'body')
    assert any(r['observation']==peer_id and r['label']=='Same-cue contact peer' for r in refs)


@pytest.mark.parametrize('single', [False, True])
def test_production_membership_and_replay_snapshot_preserve_the_upper_body_proof(tmp_path, single):
    import sqlite3
    from tests.test_recognition_face_landmark_fallback import retained_scene
    from stash_ai.recognition.database import FaceDatabase, SCHEMA
    from stash_ai.recognition.body_identity import scene_plan, rebuild
    from stash_ai.recognition.evaluation_scenarios import copy_history
    config,item,ids = retained_scene(tmp_path)
    face,own = masked_profile(); _,peer = masked_profile(220)
    if single:
        from tests.test_recognition_contact_single_shoulder import single_shoulder
        face, own, peer = single_shoulder()
    reason = 'same_frame_single_shoulder' if single else 'same_frame_upper_body'
    with FaceDatabase(config.database) as state:
        db = state.connection
        db.execute("INSERT INTO people VALUES ('A','scope','ns',1,NULL,'face-A')")
        db.execute("INSERT INTO members SELECT face,'A' FROM body_observations")
        for oid in ids:
            quality = json.loads(db.execute('SELECT quality FROM body_observations WHERE id=?',(oid,)).fetchone()[0])
            quality['quality'].update(detected_people=2,overlap=0.,box_overlap=.2604,
                contact_face_evidence=measure(face,own,[own,peer],face_id='face-'+oid))
            db.execute('UPDATE body_observations SET quality=? WHERE id=?',(json.dumps(quality),oid))
        expected = scene_plan(db,item['scan'],config)[2]
        assert all(d['status']=='accepted' and d['person']=='A' for d in expected.values())
        assert all(d['contact_confirmation']['reason']==reason for d in expected.values())
        before = list(map(tuple,db.execute('SELECT * FROM body_observations ORDER BY id')))
        rebuild(state,[item],config,{})
        stored = [json.loads(r[0]) for r in db.execute('SELECT proof FROM body_members')]
        assert all(d['contact_confirmation']['same_frame']['upper_body']['confirmed'] for d in stored)
        assert before == list(map(tuple,db.execute('SELECT * FROM body_observations ORDER BY id')))
        with sqlite3.connect(':memory:') as replay:
            replay.row_factory = sqlite3.Row
            replay.executescript(SCHEMA)
            copy_history(db,replay,[],'scope','ns')
            actual = scene_plan(replay,item['scan'],config)[2]
            assert actual == expected
