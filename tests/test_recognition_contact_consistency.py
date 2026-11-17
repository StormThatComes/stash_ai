import numpy as np
from stash_ai.recognition.contact_anchors import crowded_link
from stash_ai.recognition.limb_ownership import fuse


def test_contact_survives_disjoint_masks():
    assert crowded_link(dict(detected_people=2, overlap=0, box_overlap=.8))
    assert not crowded_link(dict(detected_people=1, overlap=0, box_overlap=.8))


def test_mask_does_not_authorize_a_long_rival_chain():
    xy = np.full((2,17,2), -1.)
    scores = np.zeros((2,17))
    xy[0,[5,7,9]] = [[10,10],[10,20],[70,20]]
    xy[1,[5,7,9]] = [[80,10],[80,20],[70,20]]
    scores[:,[5,7,9]] = .9
    probs = {k:scores.copy() for k in ('presence','visibility')}
    mask = np.ones((100,100), bool)
    result = fuse(0,0,(xy,scores,probs),(xy,scores,probs),mask,[0,0,100,100],
                  dict(min_keypoint_score=.5,min_presence=.5,min_visibility=.5))
    assert result[1][9] == 0
    assert result[1][7] > 0
