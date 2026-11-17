import pytest
from benchmarks.benchmark_face_recognizers import calibrate,rates,validate_manifest,pairs
from benchmarks.benchmark_dance_contact import select_profile


def test_tied_impostors_cannot_exceed_false_match_target():
    samples=[(.9,False,[]),(.9,False,[]),(.2,False,[]),(.95,True,[])]
    threshold=calibrate(samples,.34)
    result=rates(samples,threshold)
    assert result['false_match_rate']<=.34 and result['true_match_rate']==1


def test_split_identity_leakage_is_rejected():
    with pytest.raises(ValueError,match='identity leakage'):
        validate_manifest([dict(identity='a',scene='1',split='calibration'),dict(identity='a',scene='2',split='validation')])


def test_pair_budget_is_enforced_without_silent_truncation():
    rows=[dict(identity=str(i),scene=str(i),split='calibration') for i in range(3)]
    with pytest.raises(ValueError,match='budget'):
        pairs(rows,[[1,0]]*3,maximum=1)


def test_partner_joints_outrank_extra_recall():
    def result(correct,partner):
        totals=dict(people=2,clean_masks=2,matched=2,correct_joints=correct,partner_joint_hits=partner,
                    own_hits=20,partner_hits=0,background_hits=0,extra=0)
        return dict(totals=totals,frames={'photo':dict(counts=totals)})
    assert select_profile({'safe':result(10,0),'leaky':result(20,1)},{'safe':{},'leaky':{}})=='safe'
