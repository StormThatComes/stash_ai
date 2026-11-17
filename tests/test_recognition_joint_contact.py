import numpy as np
from stash_ai.recognition.mask_candidates import Candidate, resolve_options
from stash_ai.recognition.joint_limbs import settle
from stash_ai.recognition.mask_uncertainty import boundary_fraction


def candidate(lo,hi):
    return Candidate(np.packbits(np.ones((4,hi-lo),bool)),(lo,0,hi,4),4*(hi-lo),
                     [lo,0,hi,4],np.array([0]))


def test_three_person_mask_group_and_budget_abstention():
    chosen={i:candidate(0,12) for i in range(3)}
    options={i:[candidate(i*4,i*4+4)] for i in range(3)}
    evidence=[np.empty((0,2))]*3
    assert resolve_options(chosen,options,evidence,maximum=1)=={0,1,2}
    assert not resolve_options(chosen,options,evidence)


def test_unmeasured_parents_do_not_decide_shared_endpoint():
    xy=np.full((2,17,2),-1.); score=np.zeros((2,17))
    xy[:,9]=[40,40]; xy[0,7]=[10,10]; xy[1,7]=[70,10]; score[:,9]=.9
    prob=dict(presence=score.copy(),visibility=score.copy()); records=[{},{}]
    settle((xy,score,prob),[[0,0,100,100]]*2,records,dict(min_keypoint_score=.5))
    assert not score[:,9].any()
    assert all(r['joint_ownership_withheld']==[9] for r in records)


def test_thin_mask_has_more_boundary_uncertainty():
    solid=np.ones((20,20),bool)
    thin=np.zeros_like(solid); thin[:,9:11]=True
    assert boundary_fraction(thin)>boundary_fraction(solid)
