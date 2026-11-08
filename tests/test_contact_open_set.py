import numpy as np
from PIL import Image
from benchmarks.kpr_contact import open_set
from stash_ai.recognition.body_collection import _pixels


def test_foreground_edges_do_not_create_sharpness():
    mask=np.zeros((12,12),dtype=bool)
    mask[3:9,3:9]=True
    pixels=np.where(mask,35,127).astype('uint8')
    with Image.fromarray(pixels) as image:
        assert _pixels(image,(0,0,12,12))[0]>0
        assert _pixels(image,(0,0,12,12),mask)[0]==0


def test_absent_query_uses_a_threshold_learned_on_another_scene():
    items=[dict(scene=s,truth='A',owner='A',distance=.1,absent=d) for s,d in [('1',.7),('2',.6)]]
    result=open_set(items)
    assert result['known']==2 and result['recovered']==2
    assert result['absent']==2 and result['false_accepts']==1
    assert open_set(items[:1])['uncalibrated_queries']==1
