import json
import numpy as np

from stash_ai.recognition.body_identity import scene_plan, defaults
from stash_ai.recognition.body_perspectives import record_scene_roster
from stash_ai.recognition.database import FaceDatabase, encoded
from tests.test_recognition_body_identity import configuration, scene


def test_consistent_near_misses_remain_one_anonymous_fragment(tmp_path):
    cfg = configuration(tmp_path)
    with FaceDatabase(cfg.database) as state:
        db = state.connection
        item,ids = scene(db,cfg,vectors=[[1.,0.]]+[[.8,.6]]*3)
        db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',('A','scope','ns',1,None,'face-A'))
        quality = encoded(dict(pixels=100,sharpness=100,pose_deviation=0.,visible_face_fraction=1.))
        db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                   ('face-A',item['scan'],0,0.,'','[]','[]',quality,None,2,np.array([1.,0.],dtype='<f4').tobytes(),None))
        db.execute('INSERT INTO members VALUES (?,?)',('face-A','A'))
        db.execute('UPDATE body_observations SET face=? WHERE id=?',('face-A',ids[0]))
        decisions = scene_plan(db,item['scan'],cfg)[2]
        owners = {decisions[oid]['person'] for oid in ids[1:]}
        assert len(owners)==1 and not owners.intersection({None,'A'})
        assert all(decisions[oid]['status']=='provisional' for oid in ids[1:])


def test_roster_uses_body_hash_independence():
    rows = {i:[dict(id=str(i),scan='s',cue=i,seconds=i,
                    quality=json.dumps({'quality':dict(pixel_digest=str(i),appearance_hash=format(1<<i,'016x'))}))]
            for i in range(4)}
    decisions = {str(i):{} for i in rows}
    record_scene_roster(rows,dict.fromkeys(['0','1','2'],'A'),{i:i for i in rows},set(),decisions,3,settings=defaults())
    assert 'scene_roster' not in decisions['3']


def test_body_only_core_can_recover_borderline_cue_in_another_shot(tmp_path):
    cfg = configuration(tmp_path)
    with FaceDatabase(cfg.database) as state:
        db = state.connection
        item,ids = scene(db,cfg,vectors=[[1.,0.]]*3+[[.83,(1-.83**2)**.5]])
        for i,oid in enumerate(ids):
            db.execute("UPDATE body_observations SET quality=json_set(quality,'$.quality.view_shot',?) WHERE id=?",(i,oid))
        decisions = scene_plan(db,item['scan'],cfg)[2]
        assert decisions[ids[-1]]['reason']=='scene_body_attachment'
        assert decisions[ids[-1]]['person']==decisions[ids[0]]['person']
        assert decisions[ids[-1]]['role']=='auxiliary'


def test_scene_roster_does_not_access_unfrozen_external_face_banks(tmp_path):
    from stash_ai.recognition.body_identity import _load_scene, _plan
    cfg = configuration(tmp_path)
    with FaceDatabase(cfg.database) as state:
        db = state.connection
        item,ids = scene(db,cfg,vectors=[[1.,0.]]*3+[[.83,(1-.83**2)**.5]])
        for i,oid in enumerate(ids):
            db.execute("UPDATE body_observations SET quality=json_set(quality,'$.quality.view_shot',?) WHERE id=?",(i,oid))
        db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                   ('face',item['scan'],3,3.,'','[]','[]','{}',None,2,np.array([1.,0.],dtype='<f4').tobytes(),None))
        db.execute('UPDATE body_observations SET face=? WHERE id=?',('face',ids[-1]))
        data=_load_scene(db,item['scan'],cfg)
        serial=_plan(db,cfg,*data)[2]
        worker=_plan(None,cfg,*data,frozen_faces={})[2]
        assert serial==worker


def test_four_compatible_observations_survive_nearest_neighbour_pairs(tmp_path):
    from tests.test_contact_body_claims import body
    from tests.test_recognition_body_identity import configuration
    from stash_ai.recognition.body_identity import Scores, defaults
    from stash_ai.recognition.scene_fragments import preserve_fragments, plausible_pairs
    from types import SimpleNamespace
    import numpy as np
    rows = [body(str(i),i,40) for i in range(4)]
    for row,angle in zip(rows,(0,1,19,20)):
        row['_vector'] = np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))],dtype='f4')
    scene = dict(scene='1',scope='s',namespace='n',input_sha='v')
    cfg = configuration(tmp_path)
    decisions = {r['id']:dict(reason='ambiguous_or_insufficient_face_body_bank') for r in rows}
    preserve_fragments(scene,rows,decisions,defaults(),Scores(cfg),SimpleNamespace(allows=lambda *args:True))
    assert len({d['person'] for d in decisions.values()})==1
    assert all(d['status']=='provisional' for d in decisions.values())
    rng=np.random.default_rng(3)
    rows=[dict(id=i,_vector=v/np.linalg.norm(v)) for i,v in enumerate(rng.normal(size=(50,8)).astype('f4'))]
    hits={(a['id'],b['id']) for a,b in plausible_pairs(rows,.5) if a['_vector']@b['_vector']>=.5}
    assert hits=={(i,j) for i,a in enumerate(rows) for j,b in enumerate(rows) if i<j and a['_vector']@b['_vector']>=.5}


def test_scores_from_different_modalities_cannot_outvote_each_other():
    from stash_ai.recognition.body_perspectives import separated_proposals
    options = [('q','A','endpoint_faces',[],.85),('q','B','back',[],.99)]
    decisions={'q':{}}
    assert not separated_proposals(options,decisions,{'q':{'cue':1}},.1)
    assert decisions['q']['track_blocker']['reason']=='conflicting_modalities'


def test_sparse_singletons_do_not_trigger_an_all_pairs_rival_pass():
    import numpy as np
    from types import SimpleNamespace
    from stash_ai.recognition.scene_fragments import preserve_fragments
    from stash_ai.recognition.body_identity import defaults
    from tests.test_contact_body_claims import body
    rows = [body(str(i),i,30) for i in range(40)]
    for r,v in zip(rows,np.eye(40,dtype='f4')):
        r['_vector'] = v
    # Comparisons are impossible both as winners and as rivals.
    preserve_fragments({},rows,{},defaults(),SimpleNamespace(),SimpleNamespace())
