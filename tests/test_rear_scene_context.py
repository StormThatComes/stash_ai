from stash_ai.recognition.body_perspectives import record_scene_roster,orientation_measurement
from stash_ai.recognition.body_identity import scene_bonus


def test_later_faces_nominate_earlier_query_only_from_other_shots():
    rows = {i:[dict(id=str(i),cue=i,seconds=i,scan='s',face_person='A' if i>=2 else None,quality='{}')]
            for i in range(5)}
    roots = {str(i):'A' for i in range(2,5)}
    shots = {i:int(i>=1) for i in rows}
    decisions = {str(i):{} for i in rows}
    record_scene_roster(rows,roots,shots,set(),decisions,3)
    assert decisions['0']['roster_witnesses']=={'A':['2','3','4']}
    assert 'scene_roster' not in decisions['1']


def test_body_bonus_inherits_face_bonus_until_explicitly_overridden():
    data={'clustering':{'same_scene_match_bonus':.03},'sapiensid':{'identities':{}}}
    assert scene_bonus(data)==.03
    data['sapiensid']['identities']['scene_match_bonus']=0.
    assert scene_bonus(data)==0.


def test_predicted_eyes_leave_rear_torso_query_only():
    points = [[-1,-1] for i in range(17)]
    points[5:7]=[[30,20],[65,20]]
    points[11:13]=[[36,80],[61,80]]
    points[1:3]=[[30,10],[60,10]]
    result=orientation_measurement({'keypoints':points})
    assert result['label']=='unknown' and result['possible']=='back'


def test_one_sided_rear_evidence_requires_an_adjacent_unique_link_and_no_measured_face():
    import json
    from types import SimpleNamespace
    import numpy as np
    from stash_ai.recognition.body_identity import defaults
    from stash_ai.recognition.body_perspectives import recover
    def result(distance, face):
        rows,decisions = [],{}
        for i in (0,distance):
            points = [[-1,-1] for _ in range(17)]
            points[5:7] = [[30,20],[65,20]]
            points[11:13] = [[36,80],[61,80]]
            if i:
                points[1:3] = [[30,10],[60,10]]
            row = dict(id=str(i),scan='s',cue=i,seconds=i,face_person=None,
                       geometry=json.dumps(dict(bounds=[0,0,90,110],keypoints=points)),
                       quality=json.dumps(dict(quality=dict(orientation='unknown' if i else 'back',
                           frame_signature='00'*48,view_shot=0))),
                       _face_vector=face if i else None,_vector=np.array([1.,0.]))
            rows.append(row)
            decisions[str(i)] = dict(person=None,status='withheld')
        recover(rows,decisions,{},defaults(),SimpleNamespace(face_similarity=np.dot),None)
        return decisions[str(distance)]
    assert result(1,None)['query_perspective']=='back'
    assert 'rooted_track' not in result(1,None)
    assert 'query_perspective' not in result(2,None)
    assert 'query_perspective' not in result(1,np.array([1.,0.]))
