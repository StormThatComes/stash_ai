import json
from types import SimpleNamespace

import numpy as np

from stash_ai.recognition.body_identity import Scores,defaults
from stash_ai.recognition.body_seed_modes import FrozenSeeds
from stash_ai.recognition.rooted_witnesses import select


def row(cue,angle,pixel):
    angle = np.deg2rad(angle)
    return dict(id=str(cue),scan='s',cue=cue,seconds=float(cue),_face_vector=None,
                quality=json.dumps(dict(quality=dict(pixel_digest=pixel))),
                _vector=np.array([np.cos(angle),np.sin(angle)],dtype='float32'))


def test_several_compatible_modes_of_one_owner_can_receive_new_witnesses():
    rows = [row(i,angle,str(i)) for i,angle in enumerate((-20,20,0,0,0,0))]
    scores = Scores(SimpleNamespace(data=dict(sapiensid=dict(execution=dict(
        reference_bytes=2**22,max_witness_comparisons=1000)))))
    seeds = FrozenSeeds({'A':rows},scores,defaults())
    assert len(seeds.modes['A'])==2
    assert seeds.rank(row(10,0,'q'),SimpleNamespace(allows=lambda *args:True))[0][4]


def test_repeated_pixels_do_not_use_witness_slots():
    tracks = [(f'a{i}',f'b{i}',row(i,0,'repeat' if i<8 else str(i))) for i in range(16)]
    kept = select(tracks,dict(defaults(),attach_min_support=3),SimpleNamespace(face_similarity=np.dot),
                  lambda t:-t[2]['cue'],lambda r:r['cue'])
    assert len(kept)==8
    assert len({t[2]['quality'] for t in kept})==8


def test_rival_witness_is_query_only_until_independent_conflict_is_removed(tmp_path):
    from tests.test_recognition_body_global import configured,scene
    from stash_ai.recognition.database import FaceDatabase,encoded
    from stash_ai.recognition.body_identity import scene_plan
    cfg = configured(tmp_path)
    with FaceDatabase(cfg.database) as state:
        db = state.connection
        front, back = [.82,.5724],[1.,0.]
        cues = [0,1,2,5,6,7,10]
        scan,ids = scene(db,cfg,'1',[front,back,front,front,back,front,back],times=cues)
        for owner in 'AB':
            db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',(owner,'scope','ns',1,None,owner))
        for i,oid in enumerate(ids):
            rooted = i not in (1,4)
            owner = 'B' if i==6 else 'A'
            color = 10 if i<3 else 140 if i<6 else 30
            q=dict(keypoints=10,sharpness=100,pixel_digest=oid,orientation='front' if rooted else 'back',
                   frame_signature=(bytes([color,15,15])*16).hex())
            db.execute('UPDATE body_observations SET quality=?,geometry=? WHERE id=?',
                       (encoded(dict(quality=q)),encoded(dict(bounds=[0,0,80,100])),oid))
            if rooted:
                fid='face-'+oid
                fq=encoded(dict(pixels=100,sharpness=100,pose_deviation=0.,visible_face_fraction=1.))
                db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                          (fid,scan['scan'],cues[i],float(cues[i]),'','[]','[]',fq,None,2,
                           np.array([1.,0.],dtype='<f4').tobytes(),None))
                db.execute('INSERT INTO members VALUES (?,?)',(fid,owner))
                db.execute('UPDATE body_observations SET face=? WHERE id=?',(fid,oid))
        _,_,guarded=scene_plan(db,scan['scan'],cfg)
        assert guarded[ids[1]]['view_exclusion']['rival']=='B'
        assert guarded[ids[1]]['role']=='auxiliary' and 'rooted_track' not in guarded[ids[1]]
        db.execute('DELETE FROM members WHERE person=?',('B',))
        _,_,clean=scene_plan(db,scan['scan'],cfg)
        assert 'view_exclusion' not in clean[ids[1]] and clean[ids[1]]['rooted_view']['owner']=='A'
