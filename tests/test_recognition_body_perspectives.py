import json
from types import SimpleNamespace
import numpy as np
import pytest

from stash_ai.recognition.body_perspectives import orientation, recover
from stash_ai.recognition.body_identity import Scores


def test_rear_requires_anatomical_torso_and_no_conflicting_head():
    pts = [[-1,-1] for _ in range(17)]
    pts[5:7] = [[30,20],[65,20]]
    pts[11:13] = [[36,80],[61,80]]
    assert orientation(dict(keypoints=pts)) == 'back'
    pts[1:3] = [[32,8],[60,8]]
    assert orientation(dict(keypoints=pts)) == 'unknown'
    pts[5],pts[6] = pts[6],pts[5]
    pts[11],pts[12] = pts[12],pts[11]
    assert orientation(dict(keypoints=pts)) == 'front'


@pytest.mark.parametrize('second_shot,expected', [(15, 'provisional'), (180, 'accepted')])
@pytest.mark.parametrize('self_only', [False, True])
def test_views_require_two_disjoint_confirmations_and_preserve_sources(second_shot,expected,self_only):
    from stash_ai.recognition.body_identity import defaults
    settings = dict(defaults(), min_appearance_distance=0)
    scores = Scores(SimpleNamespace(data={'sapiensid':{'execution':{
        'max_witness_comparisons':1000, 'reference_bytes':2**20}}}))
    rows, decisions, seeds = [], {}, {'A':[]}
    for cue, brightness, face in [(0,15,True),(1,15,False),(2,15,True),
                                 (5,second_shot,True),(6,second_shot,False),(7,second_shot,True),
                                 (10,80,False)]:
        oid = str(cue)
        q = dict(pixel_digest=oid, keypoints=10, sharpness=100, orientation='front' if face else 'back',
                 frame_signature=(bytes([brightness,15,15])*16).hex())
        row = dict(id=oid,scan='s',cue=cue,seconds=float(cue),quality=json.dumps({'quality':q}),
                   geometry='{"bounds":[0,0,80,100]}',face_person='A' if face else None,
                   _vector=np.array([1.,0.],dtype='float32'),_face_vector=None)
        if self_only:
            angle = np.deg2rad(17 if face else 34 if cue==6 else 0)
            row['_vector'] = np.asarray([np.cos(angle),np.sin(angle)],dtype='float32')
        rows.append(row)
        if face:
            seeds['A'].append(row)
        decisions[oid] = dict(person='A' if face else 'B', status='accepted' if face else 'provisional',
                              reason='measured_face_association' if face else 'body_scene_core')
    recover(rows,decisions,seeds,settings,scores,SimpleNamespace(allows=lambda *a:True))
    if self_only:
        # Two shots agree at .829. Including oneself would inflate both to .915.
        assert decisions['1'].get('reason') != 'face_rooted_body_view'
        assert decisions['6'].get('reason') != 'face_rooted_body_view'
        return
    assert decisions['10']['status'] == expected
    if expected=='accepted':
        assert decisions['10']['person']=='A' and decisions['10']['role']=='auxiliary'
        assert decisions['10']['source_tracks']==[('0','2','1'),('5','7','6')]


def test_representative_limit_covers_supported_rear_appearance():
    from stash_ai.recognition.body_global_banks import diverse_references
    def witness(i,direction):
        return dict(id=str(i), scan=str(i//3), cue=i,
                    quality=json.dumps({'quality':{'orientation':direction,'appearance_hash':f'{i:02x}'}}))
    rows = [witness(i,'front') for i in range(60)] + [witness(i,'back') for i in (60,61,62)]
    selected = diverse_references(rows,4)
    assert len(selected)==4
    assert {json.loads(rows[int(r['id'])]['quality'])['quality']['orientation'] for r in selected}=={'front','back'}


def test_predicted_hidden_eyes_do_not_veto_rear_torso():
    points = [[-1,-1] for _ in range(17)]
    for i, value in {1:(32,8),2:(60,8),5:(30,20),6:(65,20),11:(36,80),12:(61,80)}.items():
        points[i] = value
    visibility = [1.]*17
    visibility[1] = visibility[2] = .1
    assert orientation(dict(keypoints=points, keypoint_visibility=visibility)) == 'back'
    visibility[6] = .1
    assert orientation(dict(keypoints=points, keypoint_visibility=visibility)) == 'unknown'


def test_sharp_outlier_cannot_remove_two_agreeing_shots():
    from stash_ai.recognition.rooted_witnesses import select
    from stash_ai.recognition.body_identity import defaults
    tracks = []
    for i, direction in enumerate([-24,24,24]):
        angle = np.deg2rad(direction)
        row = dict(id=str(i),scan='s',cue=i,seconds=i,quality=json.dumps({'quality':{'pixel_digest':str(i)}}),
                   _vector=np.array([np.cos(angle),np.sin(angle)]))
        tracks.append((f's{i}',f'e{i}',row))
    scores = SimpleNamespace(face_similarity=np.dot)
    kept = select(tracks,defaults(),scores,lambda t:-t[2]['cue'],lambda r:r['cue'])
    assert [r['cue'] for _,_,r in kept] == [1,2]
