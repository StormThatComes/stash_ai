from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image

from benchmarks.benchmark_crowded_vtt import CheckedSource,contact_variants
from benchmarks.rtmo_contact import RTMO,needs_retry


def test_contact_variants_share_detector_and_keep_source_config(tmp_path):
    from tests.test_recognition_body_global import configured
    cfg = configured(tmp_path)
    variants = dict(contact_variants(cfg,tmp_path))
    assert [v.data['sapiensid']['pose']['backend'] for v in variants.values()]==[
        'vitpose_plus_large','pmpose','vitpose_plus_large','vitpose_plus_large','vitpose_plus_large']
    assert len({v.database for v in variants.values()})==5
    assert all(v.data['sapiensid']['pose']['detector']=='rfdetr_xl' for v in variants.values())
    assert variants['rfdetr-masked-pmpose'].data['sapiensid']['pose']['refinement']['masked_identity_crop']


def test_compare_uses_same_pixels_and_reports_changes():
    class Source:
        colour = 30
        def frames(self,start):
            yield start,SimpleNamespace(start=0,end=1),Image.new('RGB',(8,8),(self.colour,0,0))
    source,catalogue = Source(),{}
    for first in [True,False]:
        for _,_,image in CheckedSource(source,'1',catalogue,first).frames():
            image.close()
    source.colour = 31
    with pytest.raises(ValueError,match='VTT pixels'):
        next(CheckedSource(source,'1',catalogue,False).frames())


def test_rtmo_decodes_full_frame_only_for_conflicts():
    pytest.importorskip('cv2')
    model = object.__new__(RTMO)
    model.input='pixels'
    boxes = np.array([[[20.,40.,100.,160.,.9],[120.,40.,200.,160.,.8]]],dtype='float32')
    points = np.full((1,2,17,3),.8,dtype='float32')
    points[...,0:2] = [50.,80.]
    def run(_,inputs):
        assert inputs['pixels'].shape==(1,3,640,640)
        return boxes,points
    model.session=SimpleNamespace(run=run)
    with Image.new('RGB',(320,160)) as image:
        assert not needs_retry(image,[],[dict(bounds=[0,0,30,40])])
        assert needs_retry(image,[object(),object()],[dict(bounds=[0,0,30,40])])
        bounds,_,joints,scores = model.predict(image,dict(score_threshold=.5,nms_threshold=.7,max_people_per_frame=4))
    assert bounds==[[10,20,50,80],[60,20,100,80]]
    assert joints[0,0].tolist()==[25.,40.] and scores.shape==(2,17)


def test_one_invocation_scans_all_variants_without_publishing(tmp_path,monkeypatch):
    pytest.importorskip('torch')
    pytest.importorskip('torchvision')
    from dataclasses import replace
    from benchmarks.benchmark_crowded_vtt import run
    from tests.test_recognition_run import setup
    from tests.test_recognition_body_collection import Persons
    from stash_ai.recognition.person_collection import PersonCollector
    from stash_ai.recognition.config import plain
    cfg,client,services,_ = setup(tmp_path,monkeypatch)
    data = plain(cfg.data)
    data['models']['device']='cpu'
    data['sapiensid']['enabled']=True
    data['sapiensid']['collection']['enabled']=True
    data['sapiensid']['identities']['enabled']=True
    cfg = replace(cfg,data=data)
    monkeypatch.setattr('stash_ai.recognition.run.Services',lambda:services)
    models = []
    def collector(settings,paths,factory=None):
        models.append((settings.data['sapiensid']['pose']['backend'],factory is not None))
        return PersonCollector(settings,paths,lambda *args:Persons(1))
    monkeypatch.setattr('stash_ai.recognition.person_collection.PersonCollector',collector)
    labels = {'1':{'0':[dict(truth='A',point=[80,100],reference=True)]}}
    result = run(cfg,labels,tmp_path,compare_contact=True)
    assert models==[('vitpose_plus_large',False),('pmpose',False),('vitpose_plus_large',True),('vitpose_plus_large',True),('vitpose_plus_large',True)]
    assert len(result)==5 and all(v['verified_vtt_cues']==v['cues']>0 for v in result.values())
    assert all(v['scored_cues']==1 for v in result.values())
    assert not client.writes and not cfg.database.exists()


def test_contact_reference_uses_face_id_even_if_reference_body_was_misassigned(tmp_path):
    from tests.test_recognition_body_global import configured,scene
    from tests.test_recognition_crowded_tracks import pose
    from stash_ai.recognition.database import FaceDatabase,encoded
    from benchmarks.benchmark_crowded_vtt import score
    cfg = configured(tmp_path)
    with FaceDatabase(cfg.database) as state:
        db = state.connection
        item,ids = scene(db,cfg,vectors=[[1,0],[1,0]])
        for owner in ('face-owner','wrong-body'):
            db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',(owner,'scope','ns',1,None,owner))
        face_quality = encoded(dict(pixels=100,sharpness=100,pose_deviation=0.,visible_face_fraction=1.))
        db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                   ('anchor',item['scan'],0,0.,'','[]','[]',face_quality,None,2,np.array([1.,0.],dtype='<f4').tobytes(),None))
        db.execute('INSERT INTO members VALUES (?,?)',('anchor','face-owner'))
        db.execute('UPDATE body_observations SET face=? WHERE id=?',('anchor',ids[0]))
        for oid in ids:
            db.execute('UPDATE body_observations SET geometry=? WHERE id=?',(encoded(pose(40)),oid))
            db.execute('INSERT INTO body_members VALUES (?,?,?,?,?,?)',(oid,'wrong-body','auxiliary','accepted','p','{}'))
        label = dict(truth='A',point=[40,45])
        labels = {'1':{'0':[dict(label,reference=True)],'1':[label]}}
        assert score(db,[item],labels)['own_id_queries']==1
        result = score(db,[item],labels,reference_config=cfg)
        assert result['anchored_queries']==1 and result['own_id_queries']==0


def test_overlapping_rtmo_boxes_keep_distinct_torsos_and_protect_linked_faces():
    from benchmarks.rtmo_contact import torso_nms,preserves_faces
    from tests.test_recognition_crowded_tracks import pose
    a,b = pose(40),pose(100)
    box = [0,0,140,100]
    for p in (a,b):
        p['bounds'] = box
    def candidate(p,score):
        points = np.column_stack([p['keypoints'], np.ones(17)])
        return box,score,points
    assert torso_nms([candidate(a,.99),candidate(a,.98),candidate(b,.96)],1,
                     dict(nms_threshold=.5,max_people_per_frame=2))==[0,2]
    assert preserves_faces([a,b],[b,a],[0,1],[1,0])
    assert not preserves_faces([a,b],[b,a],[0,1],[0,1])


def test_separated_torsos_still_require_independent_head_evidence():
    from stash_ai.recognition.person_association import confirmed_in_contact
    from tests.test_recognition_partial_face_association import FACE,person
    candidate = person((0,0,100,140))
    peer = person((5,0,110,140))
    candidate['pose_refinement'] = {'separated':True}
    candidate['keypoints'][1:3] = [[-1,-1],[-1,-1]]
    assert not confirmed_in_contact(FACE,candidate,[candidate,peer])


def test_kpr_prompts_are_crop_relative_and_ignore_invisible_partner_points():
    from benchmarks.kpr_contact import prompts
    from tests.test_recognition_crowded_tracks import pose
    a,b = pose(40),pose(47)
    with Image.new('RGB',(140,100),(255,0,0)) as image:
        sample = prompts(image,[a,b],0)
    assert sample['keypoints_xyc'][5].tolist()==[8,20,1]
    assert sample['negative_kps'].shape==(1,17,3)
    assert sample['negative_kps'][0,0].tolist()==[0,0,0]
    assert sample['image'][0,0].tolist()==[0,0,255]


def test_completed_empty_cue_requires_two_independent_matching_neighbours(tmp_path):
    from tests.test_recognition_body_global import configured,scene
    from tests.test_recognition_crowded_tracks import pose
    from stash_ai.recognition.database import FaceDatabase,encoded
    from stash_ai.recognition.future_crowd import plans
    from stash_ai.recognition.body_observations import contract
    cfg = configured(tmp_path)
    with FaceDatabase(cfg.database) as state:
        db = state.connection
        item,ids = scene(db,cfg,'1',[[1,0]]*4,times=[0,0,2,2])
        for oid,x in zip(ids,[40,100,41,102]):
            db.execute('UPDATE body_observations SET geometry=?,quality=? WHERE id=?',
                       (encoded(pose(x)),encoded(dict(quality=dict(view_shot=0))),oid))
        def planned():
            return [c for c,rows,*_ in plans(db,item['scan'],contract(cfg)) if not rows]
        assert planned()==[]
        db.execute('INSERT INTO body_cues VALUES (?,?,?)',(item['scan'],contract(cfg),1))
        assert planned()==[1]
        db.execute("UPDATE body_observations SET quality=json_set(quality,'$.quality.view_shot',2) WHERE cue=2")
        assert planned()==[]


def test_paired_reid_scores_both_models_with_the_same_foreign_reference():
    from collections import Counter
    from benchmarks.kpr_contact import compare
    def record(label,ref,cue,features):
        return dict(truth=label,reference=ref,cue=('s',cue),pixels=cue,others={'B'},mixed_wrist=not ref,
                    body=np.asarray(features),kpr=np.asarray(features))
    rows = [record('A',True,0,[1,0]),record('B',True,1,[0,1]),record('A',False,2,[1,0])]
    # Deliberately choose the partner with the second encoder.
    probe = SimpleNamespace(distances=lambda q,refs:[.8,.1])
    result = compare(rows,Counter(),probe)['paired_top1']
    assert result['sapiensid']=={'own':1,'mixed_wrist_own':1}
    assert result['kpr']=={'partner':1,'mixed_wrist_partner':1}


def test_kpr_distance_uses_only_shared_visible_parts():
    from benchmarks.kpr_contact import Extractor
    query = np.zeros((2,3)),np.array([1.,0.])
    reference = np.array([[1.,0.,0.],[9.,9.,9.]]),np.array([.5,1.])
    invisible = query[0],np.array([0.,1.])
    assert Extractor.distances(query,[reference,invisible]).tolist()==[1.,np.inf]
