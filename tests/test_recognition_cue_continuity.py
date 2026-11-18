import numpy as np
from stash_ai.recognition.temporal_prompts import adjacent
from stash_ai.recognition.cue_continuity import intrudes


def test_scan_signature_preserves_verified_camera_motion():
    from PIL import Image
    from stash_ai.recognition.perspective_shots import advance
    from stash_ai.recognition.temporal_prompts import capture
    with Image.new('RGB',(100,100)) as image:
        first=advance(image,0,None,evidence=True)
        image._recognition_cue_signature=advance(image,1,first,evidence=True)
        image._recognition_scan='s'
        context=capture(image,[],np.empty((0,17,2)),np.empty((0,17),bool))
        assert context[0][4] is True


def test_bidirectional_adjacent_cue_proof_stays_in_scene_and_shot():
    signature=np.zeros((16,3),np.uint8).tobytes().hex()
    a=(('scan',4,0,signature,False),[])
    b=(('scan',5,0,signature,False),[])
    assert adjacent(a,b) and adjacent(b,a)
    assert not adjacent(a,(('scan',6,0,signature,True),[]))
    assert not adjacent(a,(('other',5,0,signature,True),[]))


def test_sparse_peer_keeps_box_veto_but_clear_skeleton_does_not():
    box=[0,0,100,100]
    assert intrudes(np.array([80,80]),box,{})
    joints={5:[10,10],6:[20,10],11:[10,40],12:[20,40]}
    assert not intrudes(np.array([80,80]),box,joints)
    assert intrudes(np.array([15,20]),box,joints)
