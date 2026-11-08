import json
import math
from types import SimpleNamespace
import numpy as np

from stash_ai.recognition.body_identity import defaults
from stash_ai.recognition.body_tracklets import plan, split_members


def test_turn_uses_adjacent_body_scores_and_keeps_anonymous_track_local():
    rows=[]
    for cue,angle in enumerate((0,24,48)):
        radians=math.radians(angle)
        q=dict(pixel_digest=str(cue),frame_signature='11'*48,view_shot=0,detected_people=1)
        rows.append(dict(id=str(cue),scan='s',cue=cue,seconds=cue,face_person=None,quality=json.dumps({'quality':q}),
                         geometry='{"bounds":[0,0,50,100]}',_vector=np.array([math.cos(radians),math.sin(radians)])))
    settings=defaults()
    guard=SimpleNamespace(allows=lambda *args:True)
    scores=SimpleNamespace(face_similarity=np.dot)
    bank=SimpleNamespace(rank=lambda *args:[])
    decisions={'0':dict(person='A')}
    scene=dict(scope='test',namespace='face',scene='1',input_sha='x')
    plan(scene,rows,{'A':[rows[0]]},decisions,settings,scores,guard,bank)
    assert decisions['2']['person']=='A' and decisions['2']['role']=='auxiliary'
    assert rows[0]['_vector']@rows[2]['_vector'] < settings['match_threshold']
    anonymous=plan(scene,rows,{}, {},settings,scores,guard,bank)
    assert len(anonymous)==3 and len(set(anonymous.values()))==1
    assert split_members(rows,{'0':'A','1':'A'},lambda r,p:False,settings)==set()
    assert split_members(rows,{'0':'A','1':'A'},lambda r,p:r['id']=='2',settings)=={'0','1'}


def test_real_scene_plan_replays_the_same_inputs_with_and_without_tracklets(tmp_path):
    from stash_ai.recognition.database import FaceDatabase,encoded
    from stash_ai.recognition.body_identity import scene_plan
    from tests.test_recognition_body_identity import configuration,scene
    cfg=configuration(tmp_path)
    angles=[math.radians(i) for i in (0,24,48)]
    with FaceDatabase(cfg.database) as state:
        db=state.connection
        item,ids=scene(db,cfg,vectors=[[math.cos(a),math.sin(a)] for a in angles])
        db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',('A','scope','ns',1,None,'face-A'))
        fid = 'face-'+ids[0]
        quality = encoded(dict(pixels=100,sharpness=100,pose_deviation=0.,visible_face_fraction=1.))
        db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                   (fid,item['scan'],0,0.,'','[]','[]',quality,None,2,np.array([1.,0.],dtype='<f4').tobytes(),None))
        db.execute('INSERT INTO members VALUES (?,?)',(fid,'A'))
        db.execute('UPDATE body_observations SET face=? WHERE id=?',(fid,ids[0]))
        for oid in ids:
            db.execute("UPDATE body_observations SET geometry=?,quality=json_set(quality,"
                       "'$.quality.frame_signature',?,'$.quality.view_shot',0,'$.quality.detected_people',1) WHERE id=?",
                       (encoded(dict(bounds=[0,0,50,100])), '11'*48,oid))
        without=scene_plan(db,item['scan'],cfg)[2]
        cfg.data['sapiensid']['identities']['tracklet_matching']=True
        with_tracks=scene_plan(db,item['scan'],cfg)[2]
        assert without[ids[-1]]['person']!='A'
        assert with_tracks[ids[-1]]['person']=='A'
        assert with_tracks[ids[-1]]['reason']=='local_body_track'


def test_face_change_withholds_only_the_unresolved_interval():
    from stash_ai.recognition.body_tracklets import compatible_segments
    chain = [dict(id=str(i),cue=i) for i in range(6)]
    decisions = {'1':{'person':'A'},'4':{'person':'B'}}
    parts = list(compatible_segments(chain,[.9]*5,decisions))
    assert [[r['id'] for r in rows] for rows,_ in parts]==[['0','1'],['4','5']]
    assert [decisions[k]['reason'] for k in ('2','3')]==['different_face_root']*2
    assert all(len(rows)==len(edges)+1 for rows,edges in parts)


def test_borderline_fragment_is_independent_of_iteration_order(tmp_path):
    from stash_ai.recognition.scene_fragments import preserve_fragments
    from stash_ai.recognition.body_identity import Scores
    from tests.test_recognition_body_identity import configuration
    from tests.test_contact_body_claims import body
    rows=[body(str(i),i,40) for i in range(5)]
    for row, angle in zip(rows,(0,5,10,20,50)):
        theta = np.deg2rad(angle)
        row['_vector'] = np.asarray([np.cos(theta),np.sin(theta)],dtype='float32')
    cfg = configuration(tmp_path)
    def partition(source):
        decisions = {r['id']:dict(reason='ambiguous_or_insufficient_face_body_bank') for r in rows}
        preserve_fragments(dict(scene='1',scope='test',namespace='ns',input_sha='v'),source,decisions,
                           defaults(),Scores(cfg),SimpleNamespace(allows=lambda *args:True))
        return [decisions[r['id']].get('person') for r in rows]
    assert partition(rows)==partition(rows[::-1])
    assert partition(rows)[0] and len(set(partition(rows)[:3]))==1
