import json
import math
from types import SimpleNamespace

import numpy as np
from stash_ai.recognition.body_identity import defaults
from stash_ai.recognition.scene_continuity import recover_turns
from stash_ai.recognition.body_guards import FaceGuard

def scene(case):
    rows=[];ds={}; truths={}
    def row(cue,side,who,assigned,view,score):
        oid=f'{cue}-{side}';x=side*130
        points=[[x+27,25] for _ in range(17)]
        for j,dx,y in ((5,10,44),(6,42,44),(11,15,93),(12,38,93)):
            points[j]=[x+(dx if view=='back' else 52-dx),y]
        if view=='back':points[1:3]=[[-1,-1],[-1,-1]]
        g=dict(bounds=[x,8,x+54,125],keypoints=points)
        q=dict(pixel_digest=oid,view_shot=0,frame_signature='112233'*16,detected_people=2)
        vec=np.array(score,dtype=float)
        rows.append(dict(id=oid,cue=cue,seconds=cue,scan='s',face_person=assigned if view=='front' else None,
                         quality=json.dumps({'quality':q}),geometry=json.dumps(g),_vector=vec,_face_vector=None))
        ds[oid]=dict(person=assigned,status='accepted' if assigned else 'provisional',role='anchor' if assigned else 'auxiliary',
                     reason=('body_scene_core' if view=='back' else 'measured_face_association') if assigned else 'anonymous_body_track')
        truths[oid]=who
    for cue in (range(3) if case=='guest' else range(7)):
        middle=cue in (1,4)
        vec=[.76, math.sqrt(1-.76**2),0] if middle else [1,0,0]
        row(cue,0,'C' if middle and case=='guest' else 'A',('D' if case!='guest' else None) if middle else 'A','back' if middle else 'front',vec)
        row(cue,1,'B','B','front',[0,0,1])
    if case=='donor':
        row(30,0,'C','D','back',[-1,0,0])
    guard=FaceGuard.__new__(FaceGuard)
    recover_turns(rows,ds,{},defaults(),SimpleNamespace(face_similarity=np.dot),guard,[])
    return ds


def test_single_guest_and_disconnected_donor():
    assert scene('guest')['1-0']['person'] is None
    decisions = scene('donor')
    assert decisions['1-0']['person']==decisions['4-0']['person']=='A'
    assert decisions['30-0']['person']=='D'


def test_replacement_scan_does_not_protect_local_owner(tmp_path):
    from tests.test_recognition_body_identity import configuration,scene as seed
    from stash_ai.recognition.body_identity import _load_scene,rebuild
    from stash_ai.recognition.database import FaceDatabase
    c=configuration(tmp_path)
    c.data['sapiensid']['identities'].update(tracklet_matching=True,body_first_matching=True)
    with FaceDatabase(c.database) as state:
        db=state.connection
        old,_=seed(db,c,'11')
        owner=rebuild(state,[old],c,{old['scan']:[]})[old['scan']][0]['person']
        new,_=seed(db,c,'111')
        db.execute("UPDATE scans SET scene='11',policy_sha='replacement' WHERE id=?",(new['scan'],))
        record,*_=_load_scene(db,new['scan'],c)
        assert owner not in record['protected_turn_owners']
