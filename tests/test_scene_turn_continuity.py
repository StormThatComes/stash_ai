import json
from types import SimpleNamespace

import numpy as np

from stash_ai.recognition.track_assignment import joint_links
from stash_ai.recognition.scene_continuity import recover_turns
from stash_ai.recognition.body_identity import defaults
from tests.test_contact_body_claims import body


def test_two_certain_links_survive_ambiguous_third_person():
    edges = {('a','x'):.9781, ('a','y'):.8572, ('b','x'):.8829, ('b','y'):.9877,
             ('c','z'):.77, ('c','x'):.1}
    boxes = {k:dict(bounds=[i*25,0,i*25+20,90]) for i,k in enumerate('abcxyz')}
    assert {(a,b) for a,b,_ in joint_links(edges,boxes,.8,.1)}=={('a','x'),('b','y')}


def test_stable_partner_does_not_assign_one_sided_entrant():
    rows, decisions = [], {}
    for cue in range(3):
        for side in range(2):
            owner = ('A' if cue==0 else None) if side==0 else 'B'
            oid = f'{cue}-{side}'
            row = body(oid,cue,30+side*125,owner,detected_people=2)
            row['_vector'] = np.array([1.,0.]) if not side and cue==0 else (
                np.array([.76,.65]) if not side else np.array([0.,-1.]))
            rows.append(row)
            decisions[oid] = dict(person=owner,status='accepted' if owner else 'provisional',
                                  reason='measured_face_association' if owner else 'anonymous_body_track')
    recover_turns(rows,decisions,{},defaults(),SimpleNamespace(face_similarity=np.dot),
                  SimpleNamespace(allows=lambda *args:True),[])
    assert not decisions['1-0']['person'] and not decisions['2-0']['person']
