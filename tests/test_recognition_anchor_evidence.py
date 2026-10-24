"""Quality experiments preserve source authority and physical competitors."""
import json
import sqlite3
import numpy as np
import pytest

from stash_ai.recognition.config import validate, plain
from stash_ai.recognition.pose_policy import is_anchor, query_allows, anchor_sql, query_sql, compact_quality
from stash_ai.recognition.database import FaceDatabase
from tests.test_recognition_anchor_views import config_for, A
from tests.test_recognition_clustering import scan
from tests.test_recognition_rebuild import changed


@pytest.mark.parametrize('pixels,sharpness,valid,anchor,query',[
    (80,40,True,True,True),(48,40,True,False,True),(80,15,True,False,True),
    (20,40,True,False,False),(80,3,True,False,False),(80,40,False,False,False)])
def test_quality_floors_agree_in_python_sql_and_compact_replay(tmp_path,pixels,sharpness,valid,anchor,query):
    config=changed(config_for(tmp_path),clustering=dict(min_anchor_face_pixels=64,min_anchor_sharpness=30,
        min_auxiliary_face_pixels=32,min_auxiliary_sharpness=10))
    grouping=config.data['clustering']
    raw=json.dumps(dict(pixels=pixels,sharpness=sharpness,alignment_valid=valid,pose_deviation=0))
    for quality in (raw,compact_quality(raw)):
        assert is_anchor(quality,grouping)==anchor
        assert query_allows(quality,grouping)==query
        with sqlite3.connect(':memory:') as db:
            db.execute('CREATE TABLE observations(quality TEXT)');db.execute('INSERT INTO observations VALUES (?)',(quality,))
            result=db.execute('SELECT '+anchor_sql('o',grouping)+','+query_sql('o',grouping)+' FROM observations o').fetchone()
            assert tuple(map(bool,result))==(anchor,query)


def test_role_floors_require_retained_scan_measurements_and_validate(tmp_path):
    from stash_ai.recognition.scanning import scan_identity
    from stash_ai.recognition.clustering import matching_policy
    config=config_for(tmp_path)
    updated=changed(config,clustering={'min_anchor_face_pixels':80})
    assert not is_anchor('{}',updated.data['clustering'])
    assert scan_identity(config,'ns')==scan_identity(updated,'ns')
    assert matching_policy(config)!=matching_policy(updated)
    data=plain(config.data);data['clustering']['min_auxiliary_face_pixels']=1
    with pytest.raises(ValueError,match='min_auxiliary_face_pixels'):validate(data)


def test_quarantined_anchor_remains_a_raw_rival_and_never_changes_membership(tmp_path):
    from stash_ai.recognition.anchor_quarantine import views
    from stash_ai.recognition.clustering import Group
    from stash_ai.recognition.historical_references import History,rank
    from stash_ai.recognition.reference_usage import ReferenceUsage
    config=changed(config_for(tmp_path),recognition={'match_threshold':.9},
                   clustering={'anchor_outlier_quarantine':True})
    outlier=np.array([0.,1.,0.,0.])
    with FaceDatabase(config.database) as db:
        sid,_=scan(db,config,'1',[A]*3+[outlier],count=4)
        records=list(db.connection.execute('SELECT * FROM observations WHERE scan=? ORDER BY cue',(sid,)))
        from stash_ai.recognition.clustering import vector
        group=Group('owner',tuple(vector(r) for r in records),True,tuple(r['id'] for r in records))
        frozen=views(db.connection,(group,),config,ReferenceUsage(100000))
        assert len(frozen)==1 and frozen[0].authority=='quarantined_core'
        assert frozen[0].promotion['excluded_ids']==[records[-1]['id']]
        assert len(group.samples)==4 and db.connection.execute('SELECT COUNT(*) FROM members').fetchone()[0]==0
        query=dict(records[0]);query['id']='query'
        bank=History((group,),frozen,{},'test')
        assert rank([query],bank,config) is None
        assert rank([query],bank,config,strict_appearance=True,quarantine_only=True)[0]=='owner'
        from dataclasses import replace
        ordinary=History((group,),(replace(frozen[0],authority='strict'),),{},'test')
        assert rank([query],ordinary,config,strict_appearance=True,quarantine_only=True) is None
        # Other identities still see the isolated raw outlier as competition.
        from stash_ai.recognition.anchor_views import _competing_anchor
        assert _competing_anchor((group,),'other',outlier,1.,.01)
        db.connection.execute('UPDATE observations SET seconds=0 WHERE scan=?',(sid,))
        assert views(db.connection,(group,),config,ReferenceUsage(100000))==()


def test_adaptive_witnesses_grow_with_independent_support_and_keep_rare_endpoints():
    from stash_ai.recognition.view_prototypes import _Cluster
    from tests.test_recognition_subcomponents import vector
    slot=_Cluster('other','0')
    for i in range(100):
        slot.add(dict(id=str(i),scan='s',seconds=i,quality='{}'),vector(i%5),maximum=32)
    slot.add(dict(id='rare',scan='s',seconds=100,quality='{}'),vector(20),maximum=32)
    assert 8<len(slot.examples)<=32 and 'rare' in slot.observations
    before=len(slot.cues)
    slot.add(dict(id='copy',scan='s',seconds=100,quality='{}'),vector(20),maximum=32)
    assert len(slot.cues)==before


def face(x=0,shift=0,embedding=None):
    geometry=[x,0,100,100,25+shift,30,75+shift,30,50+shift,50,35+shift,75,65+shift,75,.99]
    return dict(geometry=json.dumps(geometry),embedding=np.array(A if embedding is None else embedding,dtype='<f4').tobytes(),dimensions=4)


def test_physical_duplicate_requires_location_landmarks_and_embedding():
    from stash_ai.recognition.physical_faces import duplicate
    assert duplicate(face(),face(1,.5),True)
    assert not duplicate(face(),face(1,.5),False)
    assert not duplicate(face(),face(80,80),True)
    assert not duplicate(face(),face(1,15),True)
    assert not duplicate(face(),face(1,.5,[0.,1.,0.,0.]),True)


def test_duplicate_policy_is_consistent_in_pending_and_assigned_copresence(tmp_path):
    from stash_ai.recognition.decision_resolution import conflicting_claims
    from stash_ai.recognition.global_views import _cooccurs
    config=changed(config_for(tmp_path),clustering={'deduplicate_detections':True})
    rows=[dict(face(i,i*.5),id=str(i),person='p',scan='s',cue=0) for i in range(2)]
    assert not list(conflicting_claims(rows,1.01,deduplicate=True))
    with FaceDatabase(config.database) as db:
        sid,_=scan(db,config,'1',[A]*2,count=2)
        ids=[r[0] for r in db.connection.execute('SELECT id FROM observations WHERE scan=? ORDER BY cue',(sid,))]
        for i,identifier in enumerate(ids):
            db.connection.execute('UPDATE observations SET cue=0,geometry=?,embedding=? WHERE id=?',
                                  (rows[i]['geometry'],rows[i]['embedding'],identifier))
            db.connection.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',(str(i),'scope','ns',i,None,'test'+str(i)))
            db.connection.execute('INSERT INTO members VALUES (?,?)',(identifier,str(i)))
        assert _cooccurs(db.connection,'0','1')
        assert not _cooccurs(db.connection,'0','1',config)
        db.connection.execute('UPDATE observations SET geometry=? WHERE id=?',(face(80,80)['geometry'],ids[1]))
        assert _cooccurs(db.connection,'0','1',config)
