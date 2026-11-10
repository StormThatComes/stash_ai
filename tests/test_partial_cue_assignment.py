import numpy as np
from stash_ai.recognition.body_identity import Scores
from stash_ai.recognition.joint_claims import partial
from stash_ai.recognition.sequence_validation import sequence_metrics


def test_known_face_survives_an_entrant_and_an_undecided_partner():
    def compatible(i,a,j,b):
        return None in (a[0],b[0]) or a[0]!=b[0]
    known = [('A',.96,.9)]
    tied = [('B',.82,.8),('C',.82,.8)]
    assert partial([known,[]],compatible,.8,.1)==[(0,known[0])]
    assert partial([known,tied],compatible,.8,.1)==[(0,known[0])]
    assert partial([known,known],compatible,.8,.1)==[]


def test_sparse_turn_requires_the_matching_physical_predecessor():
    labels = dict(root='A',back='A',end='A',partner='B')
    loc = dict(root=('1',0),back=('1',2),end=('1',3),partner=('1',0))
    samples = {k:dict(slices=['back'] if k=='back' else []) for k in labels}
    decisions = {k:dict(accepted_identity=p,evidence={}) for k,p in labels.items()}
    def measure(peer):
        decisions['back']['evidence'] = dict(gap_predecessor=peer)
        return sequence_metrics(labels,samples,loc,decisions)
    assert measure('root')['sparse_transitions']=={'certified':1}
    assert measure('root')['turns']==1 and measure('partner')['turns']==0


def test_cosine_budget_counts_unordered_pairs_once(tmp_path):
    from tests.test_recognition_body_identity import configuration
    scores = Scores(configuration(tmp_path))
    a,b = np.eye(2,dtype=np.float32)
    assert scores.face_similarity(a,b)==scores.face_similarity(b,a)
    assert scores.count==1


def test_body_path_compensates_certified_pan_across_empty_cue():
    import json
    from types import SimpleNamespace
    from stash_ai.recognition.body_identity import defaults
    from stash_ai.recognition.body_tracklets import paths
    from tests.test_contact_body_claims import body
    left = np.random.default_rng(19).integers(0,256,(4,4,3),dtype=np.uint8)
    def observation(cue,x,frame,shot):
        row = body(str(cue),cue,x,frame_signature=frame.tobytes().hex(),frame_size=[400,300],view_shot=shot)
        row['geometry'] = json.dumps({'bounds':[x,10,x+45,120]})
        return row
    start = observation(0,20,left,0)
    finish = observation(2,120,np.roll(left,1,axis=1),0)
    score = SimpleNamespace(face_similarity=np.dot)
    assert len(list(paths([start,finish],defaults(),score)))==1
    cut = observation(2,120,np.roll(left,1,axis=1),2)
    assert not list(paths([start,cut],defaults(),score))
    assert not list(paths([start,finish],defaults(),score,occupied=[1]))
