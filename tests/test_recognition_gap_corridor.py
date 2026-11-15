from types import SimpleNamespace
import numpy as np
from stash_ai.recognition.body_identity import defaults
from stash_ai.recognition.body_tracklets import paths, split_members
from tests.test_contact_body_claims import body


def test_solo_gap_requires_complete_evidence_and_clear_corridor():
    rows = [body('a',0,40),body('b',3,40)]
    def linked(box,motion):
        return bool(list(paths(rows,defaults(),SimpleNamespace(face_similarity=np.dot),[1],
                               extended=True,gap_boxes={1:[box]},motion=motion)))
    assert linked([250,10,280,70],'0111')
    assert not linked([250,10,280,70],'0101')
    assert not linked([30,10,70,90],'0111')
    assert not linked(None,'0111')


def test_two_supported_runs_keep_open_middle():
    rows = [body(str(i),i,30) for i in range(5)]
    claims = {str(i):'A' for i in (0,1,3,4)}
    assert split_members(rows,claims,lambda *_:False,defaults())==set(claims)
