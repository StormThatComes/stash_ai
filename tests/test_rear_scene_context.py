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
