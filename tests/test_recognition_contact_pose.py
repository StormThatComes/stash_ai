import numpy as np
from tests.test_recognition_pose_refinement import pose_points, prediction, wrapper, BOX
from stash_ai.recognition.limb_ownership import fuse, maskless
from stash_ai.recognition.pose_deduplication import select
from stash_ai.recognition.pose_deduplication_config import defaults


def test_reserved_duplicate_cannot_replace_primary():
    raw=prediction([pose_points(),pose_points()])
    report=select([BOX,BOX],raw,defaults(),reserves={1:0})
    assert report['kept_indices']==[0] and report['suppressed_by']==[None,0]


def test_foreign_wrist_loses_to_supported_first_pass():
    raw=prediction([pose_points(),pose_points()+[40,0]])
    refined=prediction([pose_points()])
    raw[0][1,9]=[50,35]
    refined[0][0,9]=[50,35]
    xy,_,_,restored=fuse(0,0,raw,refined,np.ones((100,160),bool),BOX,wrapper().settings)
    assert 9 in restored
    np.testing.assert_allclose(xy[9],raw[0][0,9])


def test_maskless_ambiguous_wrist_uses_distinct_second_peak_or_abstains():
    raw=prediction([pose_points(),pose_points()+[50,0]])
    raw[0][0,9]=raw[0][1,9]
    raw[0][0,7]=[25,48]
    for alt in (True,False):
        for key,value in [('alt_x',22/160),('alt_y',55/100),('alt_strength',.9 if alt else 0)]:
            raw[2][key]=np.full((2,17),value)
        out=(raw[0].copy(),raw[1].copy(),{k:v.copy() for k,v in raw[2].items()})
        removed=maskless(raw,0,[0,0,100,100],(160,100),out,wrapper().settings)
        if alt:
            np.testing.assert_allclose(out[0][0,9],[22,55])
        else:
            assert 9 in removed and out[1][0,9]==0
