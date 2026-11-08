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
        'vitpose_plus_large','pmpose','vitpose_plus_large']
    assert len({v.database for v in variants.values()})==3
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
    assert models==[('vitpose_plus_large',False),('pmpose',False),('vitpose_plus_large',True)]
    assert len(result)==3 and all(v['verified_vtt_cues']==v['cues']>0 for v in result.values())
    assert all(v['scored_cues']==1 for v in result.values())
    assert not client.writes and not cfg.database.exists()
