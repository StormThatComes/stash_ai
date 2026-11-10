import numpy as np
from types import SimpleNamespace
from stash_ai.recognition.turn_transitions import transitions
from tests.test_recognition_crowded_tracks import pose


def links(count, gap=1, occupied=(), cuts=(), same=False, overlap=0):
    cues,geometry,quality,signals,anchors,angles = {},{},{},{},{},{}
    for cue in (0,gap):
        cues[cue] = []
        for i in range(count):
            oid = f'{cue}-{i}'
            cues[cue].append(dict(id=oid,_vector=np.eye(count)[0 if same else i]))
            geometry[oid] = pose(50+i*100)
            quality[oid] = dict(detected_people=count,view_shot=0,overlap=overlap)
            signals[oid] = np.ones((16,3))
            angles[oid] = 'back'
            if i:
                anchors[oid] = str(i)
    settings = dict(cohesion_threshold=.80,ambiguity_margin=.1)
    return transitions(cues,geometry,quality,signals,anchors,angles,settings,SimpleNamespace(face_similarity=np.dot),
                       occupied,cuts)


def test_sparse_lanes_respect_cuts_and_occupied_intermediate_cues():
    assert links(2,gap=4)['0-0'][0]=='4-0'
    assert not links(2,gap=4,cuts=[2])
    assert not links(2,gap=4,occupied=[2])
    assert not links(2,gap=5)


def test_solo_and_separated_trio_can_supply_continuity():
    assert links(1)['0-0'][0]=='1-0'
    assert links(3)['0-0'][0]=='1-0'


def test_impossible_rooted_partner_is_excluded_only_in_a_separated_cue():
    assert links(2,same=True)['0-0'][0]=='1-0'
    assert not links(2,same=True,overlap=.4)
