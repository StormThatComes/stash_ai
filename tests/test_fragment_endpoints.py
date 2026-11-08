import json
import numpy as np
from types import SimpleNamespace

from stash_ai.recognition.body_origins import origins_for_scene, physical_key
from stash_ai.recognition.fragment_endpoints import nominations
from tests.test_recognition_crowded_tracks import pose
from tests.test_recognition_body_identity import configuration, scene
from stash_ai.recognition.database import FaceDatabase


def test_distinct_perspectives_compare_their_measured_faces():
    roots = dict.fromkeys('abcd','A')
    lookup = {k:dict(scan='s',seconds=i,_face_vector=np.array([1.,0.])) for i,k in enumerate(roots)}
    looks = {('A','front'):[('a','b',dict(id='front',cue=1))],
             ('A','back'):[('c','d',dict(id='back',cue=5))]}
    scores=SimpleNamespace(face_similarity=np.dot)
    settings=dict(tracklet_matching=True,face_threshold=.85,face_margin=.1)
    assert set(nominations(looks,lookup,roots,{1:0,5:4},scores,settings))=={'front','back'}
    lookup['rival']=dict(scan='s',seconds=9,_face_vector=np.array([1.,0.]))
    roots['rival']='B'
    assert not nominations(looks,lookup,roots,{1:0,5:4},scores,settings)


def test_jitter_matches_only_one_body_at_same_time(tmp_path):
    cfg = configuration(tmp_path)
    with FaceDatabase(cfg.database) as state:
        db=state.connection
        item,ids=scene(db,cfg)
        scan=dict(db.execute('SELECT * FROM scans WHERE id=?',(item['scan'],)).fetchone())
        db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',('A','scope','ns',1,None,'export-A'))
        db.execute('INSERT INTO body_origins VALUES (?,?)',(ids[0],'A'))
        db.execute('UPDATE body_observations SET geometry=? WHERE id=?',(json.dumps(pose(50)),ids[0]))
        old=dict(db.execute('SELECT * FROM body_observations WHERE id=?',(ids[0],)).fetchone())
        fresh=dict(old,id='q',scan='new',geometry=json.dumps(pose(51)))
        scan['id']='new'
        key=physical_key(scan,fresh)
        assert origins_for_scene(db,scan,[fresh])[key]=={'A'}
        assert key not in origins_for_scene(db,scan,[fresh,dict(fresh,id='other',geometry=json.dumps(pose(52)))])


def test_distant_external_perspective_requires_four_independent_faces():
    from stash_ai.recognition.fragment_endpoints import ExternalEndpoints
    index = ExternalEndpoints.__new__(ExternalEndpoints)
    index.minimum,index.body_floor = .85,.75
    index.banks = {'A':[0]}
    source = dict(scan='elsewhere',perspective='side',roots=['c','d'],id='side')
    index.gallery = SimpleNamespace(banks=[('A','elsewhere',[source])])
    index.face = lambda root,owner:(np.array([1.,0.]),('pixels',root))
    decision = dict(rooted_track=dict(owner='A',roots=['a','b']))
    assert index.match('query','back',decision,set(),[])[:2]==('A',1.)
    assert index.match('query','back',decision,{'A'},[]) is None
    assert index.match('query','back',decision,set(),[(.80,.78,'B','other',[],.80)]) is None
    source['roots']=['a','d']
    assert index.match('query','back',decision,set(),[]) is None
