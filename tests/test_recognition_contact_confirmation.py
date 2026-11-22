"""Crowded ownership needs measured head/torso evidence, not just disjoint masks."""
import base64
from copy import deepcopy
import json
from types import SimpleNamespace
import uuid

import numpy as np
from PIL import Image, ImageDraw
import pytest

from stash_ai.recognition.contact_face_evidence import measure, retained
from stash_ai.recognition.contact_anchors import questionable_links
from stash_ai.recognition.body_identity import defaults


def sample(x=80, angle=0):
    points = [[-1., -1.] for _ in range(17)]
    for i, xy in {0:(x,100),1:(x+10,85),2:(x-10,85),5:(x-20,120),6:(x+20,120),
                  11:(x-10,180),12:(x+10,180)}.items():
        points[i] = list(xy)
    matrix = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    def rotate(p): return ((np.array(p)-[160,160]) @ matrix.T+[160,160]).tolist()
    points = [rotate(p) if p[0]>=0 else p for p in points]
    corners = np.array([rotate(p) for p in ((x-20,70),(x+20,70),(x-20,115),(x+20,115))])
    left,top = corners.min(axis=0); right,bottom = corners.max(axis=0)
    face = SimpleNamespace(box=(left,top,right-left,bottom-top),
                           landmarks=[points[2],points[1],points[0]],score=.99)
    person = dict(bounds=[0,0,320,320],keypoints=points,keypoint_visibility=[1.]*17)
    return face, person


def mask(person):
    points = person['keypoints']
    canvas = Image.new('1',(320,320))
    draw = ImageDraw.Draw(canvas)
    head = np.mean(points[:3],axis=0)
    shoulder = np.mean(points[5:7],axis=0)
    hip = np.mean(points[11:13],axis=0)
    for a,b in ((head,shoulder),(shoulder,hip),*( (points[i],head) for i in range(3)),
                *((points[i],shoulder) for i in (5,6)),*((points[i],hip) for i in (11,12))):
        draw.line([tuple(a),tuple(b)],fill=1,width=7)
    person['pose_refinement'] = dict(status='masked',mask_valid=True,bounds=person['bounds'],
        overlap_mask=base64.b64encode(np.packbits(np.asarray(canvas,dtype=bool)).tobytes()).decode())


@pytest.mark.parametrize('angle',[0,np.pi/4,np.pi/2,np.pi])
def test_large_overlapping_boxes_with_separate_measured_cores_are_confirmed(angle):
    face, own = sample(angle=angle)
    _, peer = sample(220,angle)
    before = deepcopy([own,peer])
    proof = measure(face,own,[own,peer],face_id='face')
    assert proof['confirmed'] and proof['peers'][0]['reason']=='separated_head_torso'
    assert before == [own,peer]


def test_rotated_intersecting_core_bounds_need_both_valid_masks():
    face, own = sample(angle=np.pi/4)
    _, peer = sample(140,np.pi/4)
    # Translate the parallel peer horizontally so core rectangles intersect,
    # while the two measured silhouettes remain separate.
    for p in peer['keypoints']:
        if p[0]>=0: p[0]+=15; p[1]-=25
    missing = measure(face,own,[own,peer],face_id='face')
    assert not missing['confirmed']
    mask(own); mask(peer)
    proof = measure(face,own,[own,peer],face_id='face')
    assert proof['confirmed'] and proof['peers'][0]['reason']=='separated_masks_and_anatomy'
    for application in ('overlap','body_admission'):
        peer['mask_usage'] = {application:False}
        assert not measure(face,own,[own,peer],face_id='face')['confirmed']
    peer.pop('mask_usage')
    peer['pose_refinement']['mask_valid'] = False
    assert not measure(face,own,[own,peer],face_id='face')['confirmed']


@pytest.mark.parametrize('missing', [0,1,2])
def test_two_original_head_points_suffice_including_a_missing_nose(missing):
    face, own = sample(); _, peer = sample(220)
    own['keypoints'][missing] = [-1.,-1.]
    proof = measure(face,own,[own,peer],face_id='face')
    assert proof['confirmed'] and missing not in proof['head_points']


def test_single_or_borrowed_head_point_cannot_self_confirm():
    face, own = sample(); _, peer = sample(220)
    own['keypoints'][2] = [-1.,-1.]
    own['pose_refinement'] = dict(face_prompt=[0,own['keypoints'][0]])
    proof = measure(face,own,[own,peer],face_id='face')
    assert not proof['confirmed'] and proof['head_points']==[1]
    own['pose_refinement'] = dict(temporal_prompts=[[0,own['keypoints'][0]]])
    assert not measure(face,own,[own,peer],face_id='face')['confirmed']


def test_distant_incomplete_peer_does_not_block_an_uncontested_head_and_torso():
    face,own=sample();_,peer=sample(220)
    peer['bounds']=[200,70,300,250]
    peer['keypoints']=[[-1.,-1.]]*17
    proof=measure(face,own,[own,peer],face_id='face')
    assert proof['confirmed'] and proof['peers'][0]['reason']=='separated_peer_box'
    peer['bounds']=[60,70,300,250]
    assert not measure(face,own,[own,peer],face_id='face')['confirmed']


def test_collapsed_head_points_and_shared_foreground_cannot_confirm():
    face,own=sample();_,peer=sample(140,np.pi/4)
    collapsed=deepcopy(face)
    collapsed.landmarks=[[80,100]]*3
    own['keypoints'][:3]=[[80,100]]*3
    assert not measure(collapsed,own,[own,peer],face_id='face')['confirmed']
    face,own=sample(angle=np.pi/4);_,peer=sample(140,np.pi/4)
    mask(own);mask(peer)
    packed=base64.b64encode(np.packbits(np.ones((320,320),dtype=bool)).tobytes()).decode()
    for person in (own,peer):person['pose_refinement']['overlap_mask']=packed
    proof=measure(face,own,[own,peer],face_id='face')
    assert not proof['confirmed'] and proof['peers'][0]['reason']=='mask_core_conflict'


@pytest.mark.parametrize('failure', ['stolen_head','competing_pose','missing_peer','conflicting_nose','competing_face'])
def test_disjoint_masks_do_not_override_contradictions_or_missing_peer_evidence(failure):
    face, own = sample(); other_face, peer = sample(220)
    faces = [face,other_face]
    if failure=='stolen_head':
        own['keypoints'][5:] = deepcopy(peer['keypoints'][5:])
    elif failure=='competing_pose':
        peer = deepcopy(own)
    elif failure=='missing_peer':
        peer['keypoints'][11:13] = [[-1.,-1.],[-1.,-1.]]
    elif failure=='conflicting_nose':
        own['keypoints'][0] = [105,100]
    else:
        faces = [face,deepcopy(face)]
    mask(own); mask(peer)
    assert not measure(face,own,[own,peer],face_id='face',faces=faces)['confirmed']


def row(oid='contact',cue=1,owner='A',*,local=True):
    from tests.test_contact_body_claims import body
    value = body(oid,cue,50,owner,detected_people=2,overlap=0,box_overlap=.8)
    value['face']='face'
    if local:
        face, own = sample(); _, peer = sample(220)
        quality=json.loads(value['quality'])
        quality['quality']['contact_face_evidence']=measure(face,own,[own,peer],face_id='face')
        value['quality']=json.dumps(quality)
    return value


def test_same_frame_confirmation_works_without_neighbours_but_preserves_temporal_conflict():
    from tests.test_contact_body_claims import body
    scores=SimpleNamespace(face_similarity=np.dot)
    candidate=row()
    checks={}
    assert not questionable_links([candidate],defaults(),scores,evidence=checks)
    assert checks['contact']['reason']=='same_frame_head_torso'
    assert checks['contact']['temporal']['reason']=='no_clean_neighbour'
    conflict=body('rival',2,50,'B')
    assert questionable_links([candidate,conflict],defaults(),scores,evidence=checks)=={'contact'}
    assert checks['contact']['reason']=='temporal_identity_conflict'
    assert checks['contact']['temporal']['neighbours'][1]['rival_observation']=='rival'


@pytest.mark.parametrize('change',[{'face':'other'},{'_contact_correction':{'owner':'B'}},{'face_rejected':'blurred'}])
def test_stale_or_corrected_link_cannot_reuse_same_frame_proof(change):
    candidate=row(); candidate.update(change)
    assert not retained(candidate,json.loads(candidate['quality'])['quality'])['confirmed']


def test_legacy_and_query_only_rows_keep_the_temporal_guard():
    scores=SimpleNamespace(face_similarity=np.dot)
    candidate=row(local=False)
    checks={}
    assert questionable_links([candidate],defaults(),scores,evidence=checks)=={'contact'}
    assert checks['contact']['same_frame']['reason']=='same_frame_evidence_unavailable'
    candidate=row()
    q=json.loads(candidate['quality']);q['quality']['query_only']=True;candidate['quality']=json.dumps(q)
    assert questionable_links([candidate],defaults(),scores)=={'contact'}


def test_diagnostics_distinguish_low_similarity_missing_torso_and_conflict():
    from tests.test_contact_body_claims import body
    from stash_ai.recognition.contact_diagnostics import describe
    candidate=row(local=False); neighbour=body('own',0,50,'A')
    neighbour['_vector']=np.array([0.,1.])
    checks={}
    questionable_links([candidate,neighbour],defaults(),SimpleNamespace(face_similarity=np.dot),evidence=checks)
    assert checks['contact']['reason']=='low_body_similarity'
    text='\n'.join(describe(checks['contact']))
    assert 'low_body_similarity' in text and '0.0000' in text and '0.8000' in text
    geometry=json.loads(candidate['geometry']);geometry['keypoints'][5:]=[[-1,-1]]*12
    candidate['geometry']=json.dumps(geometry)
    questionable_links([candidate],defaults(),SimpleNamespace(face_similarity=np.dot),evidence=checks)
    assert checks['contact']['reason']=='insufficient_torso_points'


def test_collection_retains_small_evidence_with_complete_roster_and_no_pose_mutation(tmp_path):
    from stash_ai.recognition.body_collection import collect
    from stash_ai.recognition.config import load
    face, own=sample(); _,peer=sample(220)
    mask(own); mask(peer)
    people=[own,peer];before=deepcopy(people);captured=[]
    pipeline=SimpleNamespace(collect_items=lambda db,image,items:captured.extend(items))
    model=SimpleNamespace(person_geometry=lambda *args:None)
    measurements=[(dict(width=320,height=320,keypoints=7),None) for _ in people]
    collect(pipeline,None,model,load(root=tmp_path),str(uuid.uuid4()),0,0.,None,
            [dict(id='face',rejected=None)],[face],captured=(people,[0],measurements))
    metadata=captured[0][0][1]
    proof=metadata['quality']['contact_face_evidence']
    assert proof['confirmed'] and proof['peers'][0]['observation']==captured[1][0][0]
    assert 'overlap_mask' not in json.dumps({k:v for k,v in metadata.items() if not k.startswith('_')})
    assert len(json.dumps(proof))<2000 and people==before


def test_scene_rebuild_persists_contact_proof_and_inspector_explains_it(tmp_path):
    from tests.test_recognition_face_landmark_fallback import retained_scene
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.body_identity import scene_plan,rebuild
    from stash_ai.recognition.assignment_explanation import body as explain
    from stash_ai.recognition.identity_window import describe
    config,item,ids=retained_scene(tmp_path)
    face,own=sample();_,peer=sample(220)
    with FaceDatabase(config.database) as state:
        db=state.connection
        db.execute("INSERT INTO people VALUES ('A','scope','ns',1,NULL,'face-A')")
        db.execute("INSERT INTO members SELECT face,'A' FROM body_observations")
        for oid in ids:
            quality=json.loads(db.execute('SELECT quality FROM body_observations WHERE id=?',(oid,)).fetchone()[0])
            quality['quality'].update(detected_people=2,overlap=0,box_overlap=.8,
                contact_face_evidence=measure(face,own,[own,peer],face_id='face-'+oid))
            db.execute('UPDATE body_observations SET quality=? WHERE id=?',(json.dumps(quality),oid))
        decisions=scene_plan(db,item['scan'],config)[2]
        assert all(d['status']=='accepted' and d['person']=='A' for d in decisions.values())
        assert all(d['contact_confirmation']['reason']=='same_frame_head_torso' for d in decisions.values())
        before=list(map(tuple,db.execute('SELECT * FROM body_observations ORDER BY id')))
        rebuild(state,[item],config,{})
        assert before==list(map(tuple,db.execute('SELECT * FROM body_observations ORDER BY id')))
        member=db.execute('SELECT * FROM body_members WHERE observation=?',(ids[0],)).fetchone()
        proof=json.loads(member['proof'])
        trace=explain(proof,linked_face='face-'+ids[0],role=member['role'],status=member['status'],
                      policy=member['policy'],current=True,merges={},target='A')
        assert 'Contact face check: CONFIRMED' in describe(trace)


def test_collection_contract_refreshes_old_evidence_but_keeps_old_rival_banks(tmp_path):
    from stash_ai.recognition.config import load
    from stash_ai.recognition.body_observations import contract,compatible_contracts,_contract
    config=load(root=tmp_path)
    old=_contract(config,'independent-person-v4-independent-torso',legacy_mask_overlap=False,
                  legacy_foreground_sharpness=False,legacy_drift_mask=False,legacy_mask_prompts=False)
    assert contract(config)!=old and old in compatible_contracts(config)


def test_withheld_details_include_contact_failure_and_reference_links(tmp_path):
    from tests.test_recognition_identity_inspector import scene_fixture
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.scene_inspection import SceneReader,UnassignedReader
    from stash_ai.recognition.identity_window import describe
    config,_=scene_fixture(tmp_path)
    checks={}
    questionable_links([row(local=False)],defaults(),SimpleNamespace(face_similarity=np.dot),evidence=checks)
    checks['contact']['temporal']['neighbours'][0].update(rival_observation='accepted-body',rival_score=.9)
    with FaceDatabase(config.database) as state:
        state.connection.execute('UPDATE body_members SET proof=? WHERE observation=?',
            (json.dumps(dict(reason='unconfirmed_contact_face',contact_confirmation=checks['contact'])),'withheld-body'))
        state.connection.commit()
    detail=UnassignedReader(config,SceneReader(config,'2').catalogue()['scan']).detail('body','withheld-body',pixels=False)
    text=describe(detail['trace'])
    assert 'no_clean_neighbour' in text and 'same_frame_evidence_unavailable' in text and '0.9000' in text
    assert any(r['observation']=='accepted-body' and r['label']=='Contact rival body'
               for r in detail['trace']['decisions'][0]['references'])


def test_endpoint_confirmation_and_diagnostics_remain_independent_of_local_proof():
    from tests.test_contact_body_claims import body
    rows=[body('first',0,50,'A'),row(cue=2,local=False),body('last',4,50,'A')]
    checks={}
    assert not questionable_links(rows,defaults(),SimpleNamespace(face_similarity=np.dot),
                                  graph=[(rows,[.99,.99])],evidence=checks)
    assert checks['contact']['reason']=='independent_clean_endpoints'
    assert checks['contact']['witness_ids']==['first','last']
