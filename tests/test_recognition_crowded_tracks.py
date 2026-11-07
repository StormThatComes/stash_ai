from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from stash_ai.recognition.crowd_context import attach_context, remember
from stash_ai.recognition.crowded_retry import retry_people
from stash_ai.recognition.crossing_tracks import crossings


def pose(x):
    points = [[-1.,-1.] for _ in range(17)]
    for i,dx,y in [(5,-6,25),(6,6,25),(11,-4,65),(12,4,65)]:
        points[i] = [x+dx,y]
    return dict(bounds=[x-14,5,x+14,95],keypoints=points)


@pytest.mark.parametrize('face_count',[0,1])
def test_faceless_or_single_face_retry_requires_two_independent_torsos(face_count):
    found = [pose(42),pose(115)]
    original = [dict(bounds=[0,0,160,100])]
    class Tensor:
        def __init__(self,value): self.value=np.asarray(value)
        def cpu(self): return self
        def detach(self): return self
        def numpy(self): return self.value
    model = SimpleNamespace(config=SimpleNamespace(data={'sapiensid':{'pose':{
        'max_people_per_frame':3,'score_threshold':.5}}}),pose_refiner=SimpleNamespace(predict=lambda *args:()),
        _people=lambda *args:found)
    def predict(image):
        boxes = SimpleNamespace(xyxy=Tensor([[0,0,image.width,image.height]]),conf=Tensor([.95]))
        return [SimpleNamespace(boxes=boxes)]
    model._predict_pose = predict
    faces = []
    if face_count:
        faces.append(SimpleNamespace(box=(20,0,35,27),landmarks=[(37,12),(47,12),(42,18)]))
        found[0]['keypoints'][:3] = [[42,18],[47,12],[37,12]]
    with Image.new('RGB',(160,100)) as image:
        for cue in range(2):
            image._recognition_cue_signature = (cue,0,'')
            attach_context(model,image,'scene',cue,cue)
            remember(model,image,found)
        image._recognition_cue_signature = (2,0,'')
        attach_context(model,image,'scene',2,2)
        assert retry_people(model,image,faces,original)==found
        image._recognition_cue_signature = (2,2,'')
        attach_context(model,image,'scene',2,2)
        assert retry_people(model,image,faces,original) is original


def test_crossing_scores_both_directions_and_abstains_on_tie_or_face_change():
    cues = {i:[dict(id=f'{i}{name}') for name in 'ab'] for i in range(3)}
    roots = {f'{i}{name}':name for i in (0,2) for name in 'ab'}
    decisions = {r['id']:{} for rows in cues.values() for r in rows}
    def transition(a,b):
        return .82 if a['id'][0]=='0' else .99 if a['id'][1]==b['id'][1] else .81
    claims = crossings(cues,roots,decisions,transition,.06)
    assert {oid:item[0] for oid,item in claims.items()}=={'1a':'a','1b':'b'}
    assert not crossings(cues,roots,decisions,lambda *a:.9,.06)
    roots['2b']='foreign'
    assert not crossings(cues,roots,decisions,transition,.06)


def test_occluded_wrong_eye_does_not_veto_visible_profile():
    from tests.test_recognition_partial_face_association import FACE,person
    from stash_ai.recognition.person_association import association_rejection
    candidate = person((0,0,100,140))
    candidate['keypoints'][1] = [86.,75.]
    candidate['keypoint_visibility'] = [1.]*17
    assert association_rejection(FACE,candidate)=='head_landmark_mismatch'
    candidate['keypoint_visibility'][1]=.07
    assert association_rejection(FACE,candidate) is None


def test_masks_benchmark_counts_both_ids_and_duplicate_pose(tmp_path):
    from benchmarks.benchmark_crowded_vtt import score, variant
    from stash_ai.recognition.database import FaceDatabase,encoded
    from tests.test_recognition_body_global import configured,scene
    cfg = configured(tmp_path)
    cfg.data['sapiensid']['pose']['backend']='pmpose'
    options=[variant(cfg,k,tmp_path/'state').data['sapiensid']['pose']['refinement']
             for k in ('unmasked','sam_overlap','masked_identity')]
    assert [(p['enabled'],p['masked_identity_crop']) for p in options]==[(False,False),(True,False),(True,True)]
    with FaceDatabase(cfg.database) as state:
        db=state.connection
        item,ids=scene(db,cfg,'1',[[1.,0.]]*7,times=[0,1,2,2,3,3,3])
        for owner in ('A','B'):
            db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',(owner,'scope','ns',1,None,owner))
        for oid,owner in zip(ids,'ABABABB'):
            db.execute('UPDATE body_observations SET geometry=? WHERE id=?',
                       (encoded(pose(35 if owner=='A' else 110)),oid))
            db.execute('INSERT INTO body_members VALUES (?,?,?,?,?,?)',(oid,owner,'anchor','accepted','test','{}'))
        a=dict(truth='A',point=[35,45]); b=dict(truth='B',point=[110,45])
        labels={'1':{'0':[dict(a,reference=True)],'1':[dict(b,reference=True)],'2':[a,b],'3':[a,b]}}
        result=score(db,[item],labels)
        assert result['both_correct']==1 and result['duplicate_poses']==1 and result['labelled_pairs']==2


def test_multi_cue_crossing_with_one_measured_middle_face():
    cues = {i:[dict(id=f'{i}{name}', face_person=name if i==2 and name=='a' else None)
               for name in 'ab'] for i in range(5)}
    roots = {f'{i}{name}':name for i in (0,4) for name in 'ab'}
    decisions = {r['id']:dict(person=r['face_person']) for rows in cues.values() for r in rows}
    def transition(a,b):
        return .99 if a['id'][1]==b['id'][1] else .70
    claims = crossings(cues,roots,decisions,transition,.05)
    assert {oid:p[0] for oid,p in claims.items()}=={key:key[1] for key in ('1a','1b','2b','3a','3b')}
    assert not crossings(cues,roots,decisions,lambda a,b:None,.05)


def test_future_context_needs_two_original_torsos_and_same_shot(tmp_path):
    from stash_ai.recognition.future_crowd import plans
    from stash_ai.recognition.database import FaceDatabase,encoded
    from stash_ai.recognition.body_observations import contract
    from tests.test_recognition_body_global import configured,scene
    cfg = configured(tmp_path)
    with FaceDatabase(cfg.database) as state:
        db = state.connection
        scan,ids = scene(db,cfg,'1',[[1.,0.]]*5,times=[0,1,1,2,2])
        for oid,x in zip(ids,[70,40,110,40,110]):
            person = pose(x)
            if oid == ids[0]:
                person['bounds'] = [0,0,160,100]
            db.execute('UPDATE body_observations SET geometry=?,quality=? WHERE id=?',
                       (encoded(person),encoded(dict(quality=dict(view_shot=0))),oid))
        def planned():
            return [cue for cue,*_ in plans(db,scan['scan'],contract(cfg))]
        assert planned()==[0]
        db.execute("UPDATE body_observations SET geometry=json_set(geometry,'$.crowded_box_retry',1) WHERE id=?",(ids[-1],))
        assert planned()==[]
