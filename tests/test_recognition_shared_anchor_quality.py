import json
from dataclasses import replace
import pytest
from stash_ai.recognition.config import plain
from stash_ai.recognition.clustering import matching_policy
from stash_ai.recognition.database import FaceDatabase,encoded
from stash_ai.recognition.person_anchor import anchor_gate
from stash_ai.recognition.person_signal import contract
from stash_ai.recognition.hybrid_references import body_rows
from stash_ai.recognition.body_identity import anchor
from tests.test_recognition_hybrid_merge import prepared


@pytest.mark.parametrize('quality,eligible',[
    ({'keypoints':17,'sharpness':1000},True),
    ({'keypoints':3,'sharpness':1000},False),
    ({'keypoints':17,'sharpness':0},False),
    ({'keypoints':17,'sharpness':float('nan')},False),
    ({'keypoints':True,'sharpness':1000},False),({},False),
])
def test_both_routes_require_same_measured_quality(tmp_path,quality,eligible):
    c,people=prepared(tmp_path)
    settings=c.data['sapiensid']['identities'];settings.update(anchor_min_keypoints=17,anchor_min_sharpness=1000)
    metadata=dict(quality=quality)
    assert anchor_gate(metadata,settings)['eligible']==eligible
    assert anchor(dict(quality=json.dumps(metadata)),settings)==eligible
    # SQLite rejects non-JSON finite values itself; exercise this at admission above.
    if any(isinstance(v,float) and v!=v for v in quality.values()):return
    with FaceDatabase(c.database) as state:
        db=state.connection
        db.execute('UPDATE person_evidence SET metadata=?',(encoded(metadata),));db.commit()
        ns=db.execute('SELECT namespace FROM people LIMIT 1').fetchone()[0]
        assert bool(body_rows(db,c,'scope',ns,people[0]))==eligible
        assert db.execute('SELECT COUNT(*) FROM members').fetchone()[0]==6


@pytest.mark.parametrize('key,value',[('anchor_min_keypoints',17),('anchor_min_sharpness',1000)])
def test_hybrid_only_policy_tracks_quality_without_changing_embeddings(tmp_path,key,value):
    c,_=prepared(tmp_path);other=replace(c,data=plain(c.data))
    other.data['sapiensid']['identities'][key]=value
    assert not c.data['sapiensid']['identities']['enabled']
    assert matching_policy(c)!=matching_policy(other)
    assert contract(c)==contract(other)
