import json
import numpy as np
from types import SimpleNamespace
from stash_ai.recognition.contact_identity_proof import correct,birth_ready


def row(i,owner,crowded):
    return dict(id=str(i),cue=i,scan='s',seconds=float(i),face_person=owner,face_anchor=True,
                _vector=np.array([1.,0.]),_face_vector=np.array([0.,1.]),
                face_quality=json.dumps(dict(evidence_pixel_sha256=str(i))),
                quality=json.dumps(dict(quality=dict(detected_people=2 if crowded else 1,box_overlap=.8 if crowded else 0))))


def test_two_clean_endpoints_correct_and_demote_contact_body_link_only():
    rows=[row(0,'a',False),row(1,'b',True),row(2,'a',False)]
    settings=dict(correct_contact_face_links=True,match_threshold=.8,cohesion_threshold=.8,ambiguity_margin=.1)
    result=correct(rows,[(rows,[.99,.99])],settings,SimpleNamespace(face_similarity=lambda a,b:float(a@b)))
    assert result[1]['face_person']=='a' and not result[1]['face_anchor']
    assert result[1]['_contact_correction']['original_owner']=='b'
    assert rows[1]['face_person']=='b'  # raw evidence remains intact


def test_contact_birth_needs_clean_support():
    settings=dict(defer_contact_births=True,contact_birth_support=2)
    group=[row(0,'a',True),row(1,'a',True)]
    assert not birth_ready(group,settings,lambda rows,s:rows)
    assert birth_ready(group+[row(2,'a',False),row(3,'a',False)],settings,lambda rows,s:rows)
