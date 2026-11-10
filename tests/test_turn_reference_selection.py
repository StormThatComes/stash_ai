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


def test_verified_rear_segments_survive_one_unusable_member(monkeypatch):
    from collections import defaultdict
    from types import SimpleNamespace
    import numpy as np
    from stash_ai.recognition import rear_core_claims
    from stash_ai.recognition.body_identity import defaults,quality
    rows = []
    owners = defaultdict(list)
    seeds = defaultdict(list)
    angles,metadata,anchors,decisions,edges = {},{},{},{},{}
    for first in (0,10,20):
        for i in range(4):
            cue = first+i
            root = i in (0,3)
            owner = 'A' if root else 'donor'
            row = body(str(cue),cue,50,'A' if root else None,view_shot=first)
            rows.append(row)
            owners[owner].append(row)
            oid = row['id']
            metadata[oid] = quality(row)
            angles[oid] = 'front' if root else 'back'
            decisions[oid] = dict(person=owner,role='anchor',status='accepted',
                                  reason='measured_face_association' if root else 'body_scene_core')
            if root:
                seeds['A'].append(row)
                anchors[oid]='A'
            if i<3:
                edges[oid]=(str(cue+1),)
    bad = body('bad',30,50,view_shot=30,query_only=True)
    rows.append(bad)
    owners['donor'].append(bad)
    metadata['bad']=quality(bad)
    angles['bad']='back'
    decisions['bad']=dict(person='donor',role='anchor',status='accepted',reason='body_scene_core')
    monkeypatch.setattr(rear_core_claims,'transitions',lambda *args,**kwargs:edges)
    rear_core_claims.recover(rows,decisions,seeds,set(),{}, {},metadata,{},anchors,angles,owners,
                             (),(),defaults(),SimpleNamespace(face_similarity=np.dot),SimpleNamespace(allows=lambda *a:True),0.)
    assert all(decisions[str(i)]['person']=='A' for i in (1,2,11,12,21,22))
    assert decisions['bad']['person']=='donor' and decisions['bad']['status']=='provisional'
