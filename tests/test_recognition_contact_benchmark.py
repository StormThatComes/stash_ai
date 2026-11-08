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
