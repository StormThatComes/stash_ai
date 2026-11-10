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


def test_weak_face_preserves_the_anonymous_neighbours_without_naming_them():
    from tests.test_contact_body_claims import body
    rows = [body(str(i),i,40) for i in range(3)]
    decisions = {'1':dict(person='A',role='auxiliary',reason='measured_face_association')}
    result = plan(dict(scope='s',scene='1',namespace='n',input_sha='v'),rows,{},decisions,
                  defaults(),SimpleNamespace(face_similarity=np.dot),None,None)
    assert set(result)=={'0','2'}
    assert result['0']==result['2']==decisions['1']['anonymous_fragment']
    assert decisions['1']['person']=='A'


def test_presence_is_independent_of_thumbnail_sampling_rate():
    from stash_ai.recognition.roster_presence import Presence
    def weight(step):
        shots={i*step:i for i in range(5)}
        roots=[dict(id=str(i),seconds=i*10,cue=i*step) for i in (0,2,4)]
        return Presence(roots,shots).weight(dict(cue=step,seconds=10))
    assert weight(1)==weight(5)


def test_visual_track_uses_strongest_root_and_ignores_vtt_spacing():
    from tests.test_contact_body_claims import body
    from stash_ai.recognition.body_tracklets import paths
    from stash_ai.recognition.sequence_validation import sequence_metrics
    from stash_ai.recognition.roster_presence import Presence
    rows = [body(str(i), i, 30) for i in range(4)]
    rows[0]['_vector'] = np.array([math.cos(.61), math.sin(.61)])
    scores = SimpleNamespace(face_similarity=np.dot)
    guard = SimpleNamespace(allows=lambda *args: True)
    bank = SimpleNamespace(rank=lambda *args: [(.76,.76,'B',(),True,.76)])
    def replay(times):
        for row,seconds in zip(rows,times):
            row['seconds'] = seconds
        decisions = {str(i):dict(person='A') for i in (0,3)}
        plan({},rows,{'A':[rows[0],rows[3]]},decisions,defaults(),scores,guard,bank)
        assert decisions['1']['root']=='3'
        labels = {r['id']:'actor' for r in rows}
        samples = {r['id']:dict(slices=['back'] if r['cue'] in (1,2) else []) for r in rows}
        locations = {r['id']:('s',r['cue']) for r in rows}
        predicted = {i:dict(accepted_identity=d['person'], evidence_kind='person' if i in ('1','2') else 'face',
                            evidence={'reason':'candidate_group'}) for i,d in decisions.items()}
        metrics = sequence_metrics(labels,samples,locations,predicted,timing={
            ('s',r['cue']):(r['seconds'],0) for r in rows})
        assert metrics['own_id_turns']==1
        weight = Presence([rows[0],rows[-1]],{i:i for i in range(4)}).weight(rows[1])
        return decisions, metrics, weight
    assert replay([0,1,2,3])==replay([0,15,40,110])
    rows[1]['quality'] = json.dumps({'quality':dict(view_shot=1,frame_signature='ee'*48)})
    assert all('0' not in [r['id'] for r in chain] for chain,_ in paths(rows,defaults(),scores))


def test_joint_tracks_keep_both_people_and_disjoint_large_frames():
    from stash_ai.recognition.body_tracklets import paths
    from tests.test_contact_body_claims import body
    for count in (2,5):
        rows = []
        for cue, angles in enumerate(((0,40),(12,31))):
            for side in range(count):
                row = body(f'{cue}-{side}',cue,40+side*20 if side<2 else 140*side)
                q = json.loads(row['quality'])
                q['quality']['detected_people'] = count
                row['quality'] = json.dumps(q)
                angle = math.radians(angles[side] if side<2 else 0)
                row['_vector'] = np.array([math.cos(angle), math.sin(angle)])
                rows.append(row)
        found = list(paths(rows,defaults(),SimpleNamespace(face_similarity=np.dot)))
        assert {tuple(r['id'] for r in chain) for chain,_ in found} == {
            (f'0-{i}', f'1-{i}') for i in range(count)}


def test_long_turn_needs_two_roots_and_conflict_preserves_supported_ends():
    from tests.test_contact_body_claims import body
    guard = SimpleNamespace(allows=lambda *args: True)
    bank = SimpleNamespace(rank=lambda *args: [])
    scores = SimpleNamespace(face_similarity=np.dot)
    rows = [body(str(i),i,40) for i in range(31)]
    for ends, expected in [([0],False),([0,30],True)]:
        decisions = {str(i):dict(person='A') for i in ends}
        plan({},rows,{'A':[rows[i] for i in ends]},decisions,defaults(),scores,guard,bank)
        assert ('15' in decisions) == expected
    rows = rows[:5]
    for row, angle in zip(rows,(0,15,45,75,90)):
        row['_vector'] = np.array([math.cos(math.radians(angle)),math.sin(math.radians(angle))])
    decisions = {'0':dict(person='A'),'4':dict(person='B')}
    plan({},rows,{'A':[rows[0]],'B':[rows[4]]},decisions,defaults(),scores,guard,bank)
    assert [decisions[str(i)]['person'] for i in range(5)] == ['A','A',None,'B','B']
