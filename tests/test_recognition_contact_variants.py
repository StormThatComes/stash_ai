import numpy as np
from PIL import Image
from types import SimpleNamespace

from stash_ai.recognition.mask_candidates import check_mask, select_masks, Candidate, resolve_options
from stash_ai.recognition.pose_mask_capture import capture_masks
from stash_ai.recognition.pose_refinement_config import defaults


def test_final_gates_rank_every_variant_and_report_each_attempt():
    xy = np.zeros((1,17,2))
    xy[0,[0,5]] = [[10,10],[40,40]]
    evidence = np.array([[10,10],[40,40],[11,10],[12,10],[13,10],[14,10]])
    first = np.zeros((64,64),bool)
    first[8:16,8:18] = True
    other = np.zeros_like(first)
    other[8:42,9:11] = True
    other[39:42,9:42] = True
    indices = np.array([0])
    records = [{'support_indices':[0,5]}]
    settings = dict(defaults(),adaptive_decoder_retries=True,min_prompt_points=1,max_prompt_points=1)
    owner = SimpleNamespace(settings=settings, segmenter=None,
        _prompts=lambda boxes, points, cores, i, chosen, leaks: (points[i,chosen],[],[]))
    with Image.new('RGB',(64,64)) as image, capture_masks(image) as capture:
        selected = list(select_masks(owner,image,[[0,0,64,64]],xy,[np.array([0,5])],
            [(0,indices,np.empty((0,2)))],records,[(np.zeros_like(first),.99)],[evidence],
            decoder=lambda *args: [(first,.98),(other,.94)]))
        assert capture.history(records[0])[1]['variants'][0]['reason'] == 'missing_independent_support'
    assert len(selected) == 1 and np.array_equal(selected[0][2],other)

    assert records[0]['mask_variant'] == 1


def test_pair_choice_resolves_both_without_redecode():
    def candidate(lo, hi):
        return Candidate(np.packbits(np.ones((4,hi-lo),bool)),(lo,0,hi,4),4*(hi-lo),
                         [lo,0,hi,4],np.array([0]))
    chosen = {0:candidate(1,8),1:candidate(2,9)}
    alternatives = {0:[candidate(0,4)],1:[candidate(6,10)]}
    assert not resolve_options(chosen,alternatives,[np.empty((0,2))]*2)
    assert chosen == {i:v[0] for i,v in alternatives.items()}


def test_tiny_supported_tip_and_replay_of_component_edit():
    raw = np.zeros((20,25),bool)
    raw[3:13,1:15] = True
    raw[8,16] = True
    args = (raw,.9,np.array([[6,8]]),np.empty((0,2)),[0,0,24,19],defaults(),(25,20))
    edits = []
    cleaned,_,_ = check_mask(*args,limbs=[(np.array([6,8]),np.array([16,8]))],edits=edits)
    assert cleaned[8,16] and edits
    np.testing.assert_equal(check_mask(*args,replay=edits)[0],cleaned)
