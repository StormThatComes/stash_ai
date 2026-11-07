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
def test_views_require_two_disjoint_confirmations_and_preserve_sources(second_shot,expected):
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
                 frame_signature=(bytes([brightness])*48).hex())
        row = dict(id=oid,scan='s',cue=cue,seconds=float(cue),quality=json.dumps({'quality':q}),
                   geometry='{"bounds":[0,0,80,100]}',face_person='A' if face else None,
                   _vector=np.array([1.,0.],dtype='float32'),_face_vector=None)
        rows.append(row)
        if face:
            seeds['A'].append(row)
        decisions[oid] = dict(person='A' if face else 'B', status='accepted' if face else 'provisional',
                              reason='measured_face_association' if face else 'body_scene_core')
    recover(rows,decisions,seeds,settings,scores,SimpleNamespace(allows=lambda *a:True))
    assert decisions['10']['status'] == expected
    if expected=='accepted':
        assert decisions['10']['person']=='A' and decisions['10']['role']=='auxiliary'
        assert decisions['10']['source_tracks']==[('0','2','1'),('5','7','6')]
