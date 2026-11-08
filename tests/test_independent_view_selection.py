import json
from types import SimpleNamespace

import numpy as np

from stash_ai.recognition.body_identity import Scores,defaults
from stash_ai.recognition.body_seed_modes import FrozenSeeds
from stash_ai.recognition.rooted_witnesses import select


def row(cue,angle,pixel):
    angle = np.deg2rad(angle)
    return dict(id=str(cue),scan='s',cue=cue,seconds=float(cue),_face_vector=None,
                quality=json.dumps(dict(quality=dict(pixel_digest=pixel))),
                _vector=np.array([np.cos(angle),np.sin(angle)],dtype='float32'))


def test_several_compatible_modes_of_one_owner_can_receive_new_witnesses():
    rows = [row(i,angle,str(i)) for i,angle in enumerate((-20,20,0,0,0,0))]
    scores = Scores(SimpleNamespace(data=dict(sapiensid=dict(execution=dict(
        reference_bytes=2**22,max_witness_comparisons=1000)))))
    seeds = FrozenSeeds({'A':rows},scores,defaults())
    assert len(seeds.modes['A'])==2
    assert seeds.rank(row(10,0,'q'),SimpleNamespace(allows=lambda *args:True))[0][4]


def test_repeated_pixels_do_not_use_witness_slots():
    tracks = [(f'a{i}',f'b{i}',row(i,0,'repeat' if i<8 else str(i))) for i in range(16)]
    kept = select(tracks,dict(defaults(),attach_min_support=3),SimpleNamespace(face_similarity=np.dot),
                  lambda t:-t[2]['cue'],lambda r:r['cue'])
    assert len(kept)==8
    assert len({t[2]['quality'] for t in kept})==8
