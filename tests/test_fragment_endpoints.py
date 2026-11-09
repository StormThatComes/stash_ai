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


def test_seventeenth_bank_and_independent_face_veto_of_a_rival():
    from stash_ai.recognition.fragment_endpoints import ExternalEndpoints
    index = ExternalEndpoints.__new__(ExternalEndpoints)
    index.minimum,index.body_floor = .85,.75
    banks = [('A','elsewhere',[dict(id=str(i),scan='elsewhere',perspective='back' if i<16 else 'side',
                                   roots=['c','d'])]) for i in range(17)]
    index.gallery = SimpleNamespace(banks=banks)
    index.banks = {'A':list(range(17))}
    index.face = lambda root,owner:(np.array([1.,0.]),('pixels',root))
    decision = dict(rooted_track=dict(owner='A',roots=['a','b']))
    ranked = [(.90,.88,'B','other',[],.9)]
    assert index.match('s','back',decision,set(),ranked) is None
    found = index.match('s','back',decision,set(),ranked,reject_rival=lambda owner,faces:owner=='B' and len(faces)==2)
    assert found[:2]==('A',1.) and found[2]['id']=='16'


def test_joint_origins_keep_both_historical_ids():
    from stash_ai.recognition.body_origins import labels_for_groups
    from stash_ai.recognition.body_identity import defaults
    scan = dict(scope='s',namespace='n',scene='1',input_sha='v')
    rows = [dict(id=str(i),scan='s',cue=i,seconds=i,geometry='{}',quality='{}') for i in range(7)]
    origins = {physical_key(scan,r):{'B' if i in (3,4) else 'A'} for i,r in enumerate(rows)}
    assert labels_for_groups(scan,[rows[:5],rows[5:]],origins,defaults(),['A','B'])=={0:'B',1:'A'}


def test_different_face_ids_nominate_only_distinct_shots_without_copresence():
    from stash_ai.recognition.fragment_endpoints import split_owner_nominations
    roots = dict(a='A',b='A',c='B',d='B')
    lookup = {k:dict(scan='s',seconds=i,cue=i,_face_vector=np.array([1.,0.])) for i,k in enumerate(roots)}
    looks = {('A','front'):[('a','b',dict(id='x',cue=1))],('B','back'):[('c','d',dict(id='y',cue=3))]}
    args = (looks,lookup,roots,{1:0,3:1},SimpleNamespace(face_similarity=np.dot),
            dict(tracklet_matching=True,face_threshold=.85))
    assert list(split_owner_nominations(*args))[0][1]['roots']==list('abcd')
    lookup['d']['cue']=0
    assert not list(split_owner_nominations(*args))


def test_weak_profile_veto_needs_two_independent_foreign_faces():
    from stash_ai.recognition.body_guards import FaceGuard
    from stash_ai.recognition.body_scene_jobs import Witness
    cfg = SimpleNamespace(data=dict(clustering={'max_anchor_pose_deviation':.2},
                          sapiensid={'matching':{'face_floor':.4},'execution':{'reference_bytes':2**20}}))
    def face(seconds):
        return Witness(2,np.array([0.,1.],dtype='<f4').tobytes(),True,'s',seconds,'{}')
    witnesses = [face(1)]
    guard = FaceGuard(None,cfg,scores=SimpleNamespace(face_similarity=np.dot),frozen={'A':witnesses})
    query = dict(_face_vector=np.array([1.,0.]),face_quality='{"pose_deviation":0.8}')
    assert guard.allows(query,(), 'A')
    assert not guard.allows(dict(query,face_quality='{"pose_deviation":0.1}'),(), 'A')
    witnesses.append(face(2))
    assert not guard.allows(query,(), 'A')


def test_best_external_witness_is_invariant_to_gallery_order():
    from stash_ai.recognition.fragment_endpoints import ExternalEndpoints
    index = ExternalEndpoints.__new__(ExternalEndpoints)
    index.minimum,index.body_floor = .85,.75
    witnesses = [dict(id=name,scan='s2',perspective='side',roots=[name+'1',name+'2']) for name in ('low','high')]
    index.banks = {'A':[0,1]}
    def face(root,owner):
        x = .86 if root.startswith('low') else .98 if root.startswith('high') else 1.
        return np.array([x,(1-x*x)**.5]),root
    index.face = face
    for banks in (witnesses,witnesses[::-1]):
        index.gallery = SimpleNamespace(banks=[('A','s2',[w]) for w in banks])
        owner,score,w = index.match('s1','back',dict(rooted_track=dict(owner='A',roots=['a','b'])),set(),[])
        assert (owner,score,w['id'])==('A',.98,'high')


def test_local_attachment_keeps_split_faces_for_global_revalidation():
    from stash_ai.recognition.body_identity import Scores, defaults
    from stash_ai.recognition.body_perspectives import recover
    settings=dict(defaults(),tracklet_matching=True,face_threshold=.85,face_margin=.1)
    scores=Scores(SimpleNamespace(data={'sapiensid':{'execution':{
        'max_witness_comparisons':10000,'reference_bytes':2**20}}}))
    rows,seeds,decisions=[],{'A':[],'B':[]},{}
    for start,owner,kind,brightness in [(0,'A','back',15),(5,'A','back',180),(10,'B','front',80)]:
        for offset in range(3):
            cue=start+offset
            oid=str(cue)
            root=offset!=1
            q=dict(pixel_digest=oid,orientation=kind,view_shot=start,keypoints=12,sharpness=80,
                   frame_signature=(bytes([brightness,15,15])*16).hex())
            row=dict(id=oid,cue=cue,seconds=cue,scan='s',quality=json.dumps({'quality':q}),
                     geometry='{"bounds":[0,0,80,100]}',_vector=np.array([owner=='A',owner=='B'],dtype='f4'),
                     _face_vector=np.array([1.,0.]) if root else None,face_person=owner if root else None)
            rows.append(row)
            if root:
                seeds[owner].append(row)
            decisions[oid]=dict(person=row['face_person'],status='accepted' if root else 'withheld',
                                reason='measured_face_association' if root else 'body_anchor_quality')
    recover(rows,decisions,seeds,settings,scores,SimpleNamespace(allows=lambda *args:True))
    assert decisions['1']['reason']=='face_rooted_body_view'
    assert decisions['1']['endpoint_pairs']==[dict(people=['A','B'],roots=['0','2','10','12'],minimum=1.)]
