import json
import numpy as np
import pytest

from stash_ai.recognition.body_identity import rebuild
from stash_ai.recognition.database import FaceDatabase, encoded
from tests.test_recognition_body_global import configured, scene


@pytest.mark.parametrize('missing_root', [False, True])
def test_external_rear_only_scene_requires_current_independent_roots(tmp_path, missing_root):
    config = configured(tmp_path)
    with FaceDatabase(config.database) as state:
        db = state.connection
        front, back = [1.,0.], [.82,.5724]
        a, ids = scene(db, config, '1', [front,back,front,front,back,front],
                       times=[0,1,2,5,6,7])
        b, queries = scene(db, config, '2', [back]*3)
        db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',('A','scope','ns',1,None,'face-A'))
        for i,oid in enumerate(ids+queries):
            rooted = i in (0,2,3,5)
            color = (15 if i<3 else 180, 15,15)
            q = dict(pixel_digest=oid, keypoints=10, sharpness=100, orientation='front' if rooted else 'back',
                     frame_signature=(bytes(color)*16).hex())
            db.execute('UPDATE body_observations SET quality=?,geometry=? WHERE id=?',
                       (encoded(dict(quality=q)),encoded(dict(bounds=[0,0,80,100])),oid))
            if rooted:
                fid = 'face-'+oid
                quality = encoded(dict(pixels=100,sharpness=100,pose_deviation=0.,visible_face_fraction=1.))
                db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                    (fid,a['scan'],[0,1,2,5,6,7][i],float(i),'','[]','[]',quality,None,2,
                     np.array(front,dtype='<f4').tobytes(),None))
                db.execute('INSERT INTO members VALUES (?,?)',(fid,'A'))
                db.execute('UPDATE body_observations SET face=? WHERE id=?',(fid,oid))
        db.commit()
        rebuild(state,[a,b],config,{a['scan']:[],b['scan']:[]})
        def members():
            return [dict(r) for r in db.execute('SELECT * FROM body_members WHERE observation IN (?,?,?)',queries)]
        assert all(r['person']=='A' and r['role']=='auxiliary' and
                   json.loads(r['proof'])['reason']=='external_rear_view' for r in members())
        if missing_root:
            db.execute('DELETE FROM members WHERE observation=?',('face-'+ids[0],))
            db.commit()
            rebuild(state,[a],config,{a['scan']:[]})
            assert all(r['person']!='A' for r in members())


def test_long_vtt_turn_allows_a_single_empty_cue_and_keeps_open_root():
    from stash_ai.recognition.body_perspectives import recover
    from stash_ai.recognition.body_identity import defaults, Scores
    from types import SimpleNamespace
    settings = dict(defaults(), min_appearance_distance=0)
    scorer = Scores(SimpleNamespace(data={'sapiensid':{'execution':{
        'max_witness_comparisons':2000,'reference_bytes':2**20}}}))
    rows, decisions, seeds = [], {}, {'A':[]}
    for cue in [0,1,2,4,5,6,7,8,11,12,13,17,18,19]:
        root = cue in (0,8,11,13,17)
        rgb = [15 if cue<=8 else 180 if cue<=13 else 60,15,15]
        row = dict(id=str(cue),cue=cue,scan='s',seconds=cue,face_person='A' if root else None,
                   geometry=encoded({'bounds':[0,0,80,100]}),_vector=np.array([1.,0.]),_face_vector=None,
                   quality=encoded({'quality':dict(orientation='front' if root else 'back',
                       frame_signature=(bytes(rgb)*16).hex(),pixel_digest=str(cue))}))
        rows.append(row)
        if root: seeds['A'].append(row)
        decisions[str(cue)] = dict(person='A' if root else None,status='accepted' if root else 'withheld',
                                   reason='measured_face_association' if root else 'body_anchor_quality')
    recover(rows,decisions,seeds,settings,scorer,SimpleNamespace(allows=lambda *args:True))
    assert decisions['19']['reason']=='face_rooted_body_view'
    sources = decisions['19']['source_tracks']
    assert {(start,end) for start,end,_ in sources}=={('0','8'),('11','13')}
