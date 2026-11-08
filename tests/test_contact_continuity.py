from types import SimpleNamespace
import numpy as np
from stash_ai.recognition.crossing_tracks import crossings
from stash_ai.recognition.body_guards import FaceGuard
from stash_ai.recognition.body_scene_jobs import Witness


def test_contact_with_a_partial_endpoint_is_time_symmetric():
    roots = {'A0':'A','B0':'B','A2':'A'}
    for reverse in (False,True):
        layers = {i:[dict(id=p+str(2-i if reverse else i),seconds=i) for p in 'AB'] for i in range(3)}
        decisions = {r['id']:{} for rows in layers.values() for r in rows}
        claims = crossings(layers,roots,decisions,lambda a,b:.95 if a['id'][0]==b['id'][0] else .2,.1)
        assert {k:v[0] for k,v in claims.items()} == {'A1':'A','B1':'B'}


def test_contextual_veto_needs_independent_time_and_pixels():
    cfg=SimpleNamespace(data={'sapiensid':{'matching':{'face_floor':.5},'execution':{'reference_bytes':2**20}}})
    scores=SimpleNamespace(face_similarity=np.dot)
    query=dict(_face_vector=np.array([1.,0.]))
    def check(rows):
        return FaceGuard(None,cfg,scores=scores,frozen={'A':rows}).allows(query,(), 'A')
    def witness(t,p,strict=False):
        import json
        return Witness(2,np.asarray([0.,1.],dtype='<f4').tobytes(),strict,'s',t,
                       json.dumps({'evidence_pixel_sha256':p*64}))
    assert check([witness(1,'a'),witness(2,'a')])
    assert not check([witness(1,'a'),witness(2,'b')])
    assert not check([witness(1,'a',True)])


def test_only_two_measured_distinct_torsos_remove_a_rival():
    from tests.test_recognition_crowded_tracks import pose
    from stash_ai.recognition.rear_recovery import impossible_owners
    owners = impossible_owners({'id':'query'},pose(40),[('other','B',pose(110)),('near','C',pose(40))])
    assert owners=={'B'}
