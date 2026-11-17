from types import SimpleNamespace
import pytest
from stash_ai.recognition.contact_reserves import select


def test_detector_reserve_is_bounded_and_points_to_primary():
    torch=pytest.importorskip('torch')
    boxes=torch.tensor([[0.,0,100,100],[10,0,110,100],[150,0,200,100]])
    output=select(SimpleNamespace(boxes=SimpleNamespace(xyxy=boxes,conf=torch.tensor([.9,.8,.7]))),.7,2)
    assert output.reserve_parents.tolist()==[-1,-1,0]
    assert output.boxes.xyxy.tolist()==boxes[[0,2,1]].tolist()


def test_almost_identical_box_is_not_a_contact_reserve():
    torch=pytest.importorskip('torch')
    output=select(SimpleNamespace(boxes=SimpleNamespace(
        xyxy=torch.tensor([[0.,0,100,100],[1,0,101,100]]),conf=torch.tensor([.9,.8]))),.7,2)
    assert output.reserve_parents.tolist()==[-1]
