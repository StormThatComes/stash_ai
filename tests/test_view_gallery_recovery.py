import json
from itertools import permutations
from types import SimpleNamespace
import numpy as np
from stash_ai.recognition.view_prototypes import _Cluster,supported_cohorts
from stash_ai.recognition.rooted_witnesses import modes
from stash_ai.recognition.body_identity import defaults


def test_matching_votes_drive_representatives_and_centroid_in_any_order():
    rows = [(1,'a',[1.,0.]),(1,'b',[.8,.6]),(2,'a',[1.,0.])]
    for order in permutations(range(3)):
        slot = _Cluster('unknown','0')
        for i in order:
            sec,p,v = rows[i]
            slot.add(dict(scan='s',seconds=sec,id=str(i),quality=json.dumps(dict(evidence_pixel_sha256=p*64))),
                     np.asarray(v))
        [selected] = supported_cohorts([slot],[],2,4,complete=True)
        assert len(selected.cues)==selected.centre_count==2
        assert set(selected.observations)=={'1','2'}
        np.testing.assert_allclose(selected.total,[1.8,.6])


def test_legacy_multiscan_view_survives_measured_view():
    slots=[]
    for name,measure in [('old',False),('new',True)]:
        slot = _Cluster('unknown',name)
        for i in range(3):
            row=dict(id=name+str(i),scan=name+str(i),seconds=i,quality=json.dumps(dict(view_shot=i) if measure else {}))
            slot.add(row,np.array([float(measure),float(not measure)]))
        slots.append(slot)
    assert set(id(s) for s in supported_cohorts(slots,[],2,4,complete=True,primary=True))==set(id(s) for s in slots)


def test_four_disjoint_supported_body_looks_are_available():
    tracks=[]
    for i in range(8):
        row=dict(id=str(i),scan='s',cue=i,seconds=i,quality=json.dumps({'quality':{'pixel_digest':str(i)}}),
                 _vector=np.eye(4)[i//2])
        tracks.append(('a'+str(i),'b'+str(i),row))
    found=list(modes(tracks,dict(defaults(),min_appearance_distance=0),SimpleNamespace(face_similarity=np.dot),
                     lambda t:-t[2]['cue'],lambda r:r['cue']))
    assert len(found)==4
