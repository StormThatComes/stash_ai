from types import SimpleNamespace
import numpy as np

from stash_ai.recognition.fragment_endpoints import ExternalEndpoints, nominations
from stash_ai.recognition.reference_gap_scope import MissingReferences
from stash_ai.recognition.database import FaceDatabase, encoded
from tests.test_recognition_body_identity import configuration, scene
from tests.test_recognition_assignment_guards import face
from tests.test_recognition_crowded_tracks import pose


def test_open_same_profile_attaches_with_positive_body_corroboration():
    vector = np.array([1.,0.])
    search = ExternalEndpoints.__new__(ExternalEndpoints)
    search.minimum,search.body_match,search.body_floor = .85,.85,.75
    search.banks = {'A':[0]}
    witness = dict(scan='other',roots=['b','c'],perspective='side',id='x',_vector=vector)
    search.gallery = SimpleNamespace(banks=[('A','other',[witness])])
    search.face = lambda root,owner:(vector,root)
    hint = dict(open_face_root=dict(owner='A',observation='a'))
    for score in (.84,.85,.86):
        match = search.match('local','side',hint,set(),[],query=np.array([score,(1-score**2)**.5]),allow_open=True)
        assert bool(match)==(score>=.85)


def test_member_specific_appearance_is_order_independent():
    unit = np.array([1.,0.])
    roots = dict.fromkeys('abcd','A')
    rows = {k:dict(scan='s',seconds=i,_face_vector=unit) for i,k in enumerate(roots)}
    for k,v in [('x',unit),('y',-unit),('z',unit)]:
        rows[k] = dict(id=k,cue=0 if k!='z' else 4,_vector=v)
    entries = [('a','b',rows['x']),('a','b',rows['y']),('c','d',rows['z'])]
    def nominate(order):
        return nominations({('A','side'):order},rows,roots,{0:0,4:4},SimpleNamespace(face_similarity=np.dot),
                           dict(tracklet_matching=True,match_threshold=.85,face_margin=.1,face_threshold=.9))
    assert nominate(entries)==nominate(entries[::-1])
    assert set(nominate(entries))=={'y','z'}


def test_retired_rival_scopes_only_with_independent_physical_exclusions(tmp_path):
    config = configuration(tmp_path)
    with FaceDatabase(config.database) as state:
        db = state.connection
        for p in 'ABR':
            db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',(p,'scope','ns',0,None,p))
        def add(n, owners):
            item,ids = scene(db,config,str(n),times=[0]*len(owners),vectors=[[1,0]]*len(owners))
            for i,(owner,oid) in enumerate(zip(owners,ids)):
                fid = face(db,item['scan'],0,owner)
                db.execute('UPDATE observations SET geometry=? WHERE id=?',(encoded([i*100,0,20,20]),fid))
                db.execute('UPDATE body_observations SET geometry=?,face=? WHERE id=?',(encoded(pose(i*100+40)),fid,oid))
                db.execute("INSERT INTO body_members VALUES (?,?,'anchor','accepted','p','{}')",(oid,owner))
        add(1,'R')
        add(2,'RA')
        gap = MissingReferences(db,'scope','ns',['1'])
        assert gap.affected(['A','B'])==['1']
        add(3,'RB')
        gap = MissingReferences(db,'scope','ns',['1'])
        assert gap.affected(['A','B'])==[]
        assert gap.affected(['A','R'])==['1']
