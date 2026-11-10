from stash_ai.recognition.scene_continuity import scene_references
from stash_ai.recognition.turn_witnesses import TurnWitnesses
from tests.test_contact_body_claims import body


def test_later_clean_crop_replaces_blurred_crop_with_same_roots():
    bank = TurnWitnesses({}, {}, None)
    for cue,overlap,sharpness in ((1,.6,4),(2,0,90),(3,.5,100)):
        bank.add('A','back',body(str(cue),cue,50,overlap=overlap,sharpness=sharpness),('l','r'),0)
    assert [r['id'] for r,_,_ in bank.by_view['A','back']]==['2']


def test_roster_reserves_examples_for_late_shots():
    rows = [dict(cue=i,id=str(i)) for i in range(30)]
    metadata = {r['id']:dict(view_shot=r['cue']//10) for r in rows}
    bank = scene_references(rows,metadata)
    assert len(bank)==8
    assert {metadata[r['id']]['view_shot'] for r in bank}=={0,1,2}
    assert scene_references(rows[::-1],metadata)==bank
