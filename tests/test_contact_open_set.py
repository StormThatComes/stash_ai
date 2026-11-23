import numpy as np
from PIL import Image
from benchmarks.kpr_contact import open_set
from stash_ai.recognition.body_collection import _pixels


def test_foreground_edges_do_not_create_sharpness():
    # Keep room for the complete reblur footprint inside the foreground. The
    # unmasked square has real edges; its constant masked interior has none.
    mask=np.zeros((96,96),dtype=bool)
    mask[24:72,24:72]=True
    pixels=np.where(mask,35,127).astype('uint8')
    with Image.fromarray(pixels) as image:
        assert _pixels(image,(0,0,96,96))[0]['sharpness']>0
        quality=_pixels(image,(0,0,96,96),mask)[0]
        assert quality['sharpness_pixels']>0
        assert quality['sharpness']==0


def test_absent_query_uses_a_threshold_learned_on_another_scene():
    items=[dict(scene=s,truth='A',owner='A',distance=.1,absent=d) for s,d in [('1',.7),('2',.6)]]
    result=open_set(items)
    assert result['known']==2 and result['recovered']==2
    assert result['absent']==2 and result['false_accepts']==1
    assert open_set(items[:1])['uncalibrated_queries']==1


def test_unenrolled_performer_is_scored_as_absent_only():
    items=[dict(scene='1',truth='A',owner='A',distance=.1,absent=.7,enrolled=True),
           dict(scene='2',truth='C',owner='A',distance=.6,absent=.6,enrolled=False)]
    result=open_set(items)
    assert result['known']==1 and result['absent']==2 and result['natural_absent']==1
