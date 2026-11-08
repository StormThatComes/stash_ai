from types import SimpleNamespace
import json
import numpy as np

from stash_ai.recognition.body_identity import defaults
from stash_ai.recognition.body_tracklets import paths, split_members
from stash_ai.recognition.contact_anchors import questionable_links, joint_body_claims
from tests.test_recognition_crowded_tracks import pose


def body(i,cue,x,owner=None,**quality):
    values = dict(detected_people=1,overlap=0.,frame_signature='11'*48,view_shot=0,pixel_digest=i)
    values.update(quality)
    return dict(id=i,cue=cue,seconds=cue,scan='s',face_person=owner,face_anchor=bool(owner),
                quality=json.dumps({'quality':values}),geometry=json.dumps(pose(x)),_vector=np.array([1.,0.]))


def test_contact_root_compares_both_sides():
    rows = [body('left',0,50,'A'),body('contact',1,50,'A',detected_people=2,overlap=.4),body('right',2,50,'B')]
    assert questionable_links(rows,defaults(),SimpleNamespace(face_similarity=np.dot))=={'contact'}


def test_partial_pair_preserves_only_the_separated_torso():
    rows = [body('A',0,30,detected_people=2),body('B',0,150,detected_people=2),
            body('a',1,30,expected_people=2)]
    chains = list(paths(rows,defaults(),SimpleNamespace(face_similarity=np.dot)))
    assert [[r['id'] for r in group] for group,_ in chains]==[['A','a']]


def test_two_interval_split_keeps_unknown_middle():
    rows = [body(str(i),i,30) for i in range(5)]
    proposed = {'0':'A','1':'A','3':'B','4':'B'}
    assert split_members(rows,proposed,lambda r,p:(r['id'],p) in {('1','B'),('3','A')},defaults())==set(proposed)
    assert not split_members(rows,proposed,lambda *args:False,defaults())


def test_joint_contact_explains_second_winner_with_the_other_body():
    rows=[body('x',0,30),body('y',0,140)]
    d={r['id']:dict(person=None,status='withheld') for r in rows}
    options={r['id']:[(v,v,p,[r],True,v) for p,v in zip('AB',scores)]
             for r,scores in zip(rows,[(.99,.86),(.94,.93)])}
    joint_body_claims(rows,d,options,defaults())
    assert [d[r['id']]['person'] for r in rows]==['A','B']
