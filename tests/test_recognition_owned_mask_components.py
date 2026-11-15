import numpy as np
from stash_ai.recognition.mask_candidates import Candidate, shared_masks, check_mask
from stash_ai.recognition.pose_refinement_config import defaults


def candidate(mask):
    y,x = np.nonzero(mask)
    l,t,r,b = x.min(), y.min(), x.max()+1, y.max()+1
    return Candidate(np.packbits(mask[t:b,l:r]), (l,t,r,b), mask.sum(), [], np.array([]))


def test_ownership_conflict_uses_mask_extent_and_only_retries_the_leaking_person():
    a = np.zeros((20,40),bool)
    a[:,2:16] = True
    b = np.zeros_like(a)
    b[:,10:35] = True
    masks = {0:candidate(a), 1:candidate(b)}
    points = [np.array([[11,6],[13,12],[5,15]]),np.array([[22,6],[24,12]])]
    assert shared_masks(masks, points) == {1}
    assert shared_masks(masks, [points[0][2:],points[1]]) == {0,1}


def test_foreign_island_removed_but_separate_owned_hand_survives():
    mask = np.zeros((50,100),bool)
    mask[5:40,5:30] = True
    mask[10:15,31:36] = True
    mask[5:40,70:95] = True
    result, box, reason = check_mask(mask,.99,[[12,15]],[[80,10],[82,25]],
        [5,5,36,40],defaults(),(100,50),own=[[12,15],[32,12]])
    assert reason is None and result[12,32] and not result[10,80]
