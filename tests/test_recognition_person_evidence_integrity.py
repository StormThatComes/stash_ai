"""Physical body votes, complete vetoes and current historical-core admission."""
import json
import math
import uuid
import numpy as np
import pytest
from stash_ai.recognition.database import FaceDatabase, encoded
from tests.test_recognition_body_identity import configuration, scene
from tests.test_recognition_merge_report import seeded
from tests.test_recognition_hybrid_recovery import enable


def _hybrid_pixel_units(p):
    from stash_ai.recognition.hybrid_references import body_rows
    from stash_ai.recognition.hybrid_evidence import report
    config,people=seeded(p)
    with FaceDatabase(config.database) as state:
        db=state.connection;config=enable(config,db)
        db.execute("UPDATE person_evidence SET metadata=json_set(metadata,'$.quality.pixel_digest',?)",('a'*64,));db.commit()
        ns=db.execute('SELECT namespace FROM people LIMIT 1').fetchone()[0]
        rows=body_rows(db,config,'scope',ns,people[0])
        duplicate_body=dict(independent=rows.support,actual_body_pixels=1,outcomes=[d['outcome'] for d in report(db,config,'scope',ns)['decisions']])
        for index,row in enumerate(db.execute('SELECT observation FROM person_evidence').fetchall()):
            db.execute("UPDATE person_evidence SET metadata=json_set(metadata,'$.quality.pixel_digest',?) WHERE observation=?",(f'{index:064x}',row[0]))
        db.execute("UPDATE observations SET quality=json_set(quality,'$.evidence_pixel_sha256',?)",('a'*64,));db.commit()
        return dict(duplicate_body=duplicate_body,distinct_body_but_same_face=body_rows(db,config,'scope',ns,people[0]).support)


def _perceptual_veto(p):
    from stash_ai.recognition.body_identity import rebuild
    from stash_ai.recognition.body_global import report
    c=configuration(p);settings=c.data['sapiensid'];settings['identities']['min_appearance_distance']=4
    settings['global_matching']['enabled']=False
    with FaceDatabase(c.database) as state:
        db=state.connection;selected=[]
        for i,sign in enumerate([1,-1]):
            item,ids=scene(db,c,str(i+1),vectors=[[1,0]]*3+[[math.cos(math.pi/6),sign*.5]])
            for oid,hashed in zip(ids,['0','f','ff','1']):
                db.execute("UPDATE body_observations SET quality=json_set(quality,'$.quality.appearance_hash',?) WHERE id=?",(hashed,oid))
            selected.append(item)
        db.commit();rebuild(state,selected,c,{i['scan']:[] for i in selected})
        settings['global_matching']['enabled']=True
        from stash_ai.recognition.body_identity import policy
        db.execute('UPDATE body_members SET policy=?',(policy(c),));db.commit()
        result=report(db,c,'scope','ns')
        return dict(real_min=.5,reported=[r['minimum'] for r in result['pairs']],outcomes=[r['outcome'] for r in result['decisions']])


def _history_support(p):
    from stash_ai.recognition.hybrid_merge import apply
    from stash_ai.recognition.hybrid_queries import recover
    from stash_ai.recognition.hybrid_evidence import catalogue
    from stash_ai.recognition.clustering import matching_policy
    y=(.7-.07)/math.sqrt(.51)
    c,people=seeded(p,vectors=[[1,0,0,0],[.7,math.sqrt(.51),0,0],[.1,y,math.sqrt(.99-y*y),0]])
    with FaceDatabase(c.database) as state:
        db=state.connection;c=enable(c,db)
        cid=[r[0] for r in db.execute('SELECT observation FROM members WHERE person=?',(people[2],))]
        bid=[r[0] for r in db.execute('SELECT observation FROM members WHERE person=?',(people[1],))]
        db.execute('DELETE FROM members WHERE person=?',(people[2],))
        db.execute("UPDATE candidates SET status='withheld',person=NULL WHERE person=?",(people[2],));db.commit()
        selected=[dict(scan=r['id'],scene=r['scene']) for r in db.execute('SELECT * FROM scans')]
        merged=apply(state,selected,c)
        for oid in bid[1:]:db.execute("UPDATE person_evidence SET metadata=json_set(metadata,'$.quality.keypoints',1) WHERE observation=?",(oid,))
        db.commit();ns=db.execute('SELECT namespace FROM people LIMIT 1').fetchone()[0]
        refs=catalogue(db,c,'scope',ns)
        recovered=recover(state,selected,c)
        return dict(merged=[d['outcome'] for d in merged['decisions']],cores=[(len(i.independent),[len(b) for _,f,b in i.query_cores]) for i in refs],recovered=recovered,signals=[json.loads(r[0]).get('hybrid_query',{}).get('signal') for r in db.execute("SELECT payload FROM candidates WHERE json_extract(payload,'$.observation') IN (SELECT value FROM json_each(?))",(encoded(cid),))])


def _rejected_projection(p):
    from stash_ai.recognition.body_observations import save_records,contract
    from stash_ai.recognition.hybrid_references import body_rows
    from stash_ai.recognition.hybrid_evidence import report
    c,people=seeded(p)
    with FaceDatabase(c.database) as state:
        db=state.connection;c=enable(c,db);c.data['sapiensid']['collection']['enabled']=True
        for row in db.execute('SELECT * FROM observations').fetchall():
            bid=str(uuid.uuid4());metadata=dict(person=dict(bounds=[0,0,512,512]),quality=dict(keypoints=17,sharpness=100,overlap=.9,pixel_digest=bid),identity_rejection='person_overlap',_body=dict(scan=row['scan'],cue=row['cue'],seconds=row['seconds'],contract=contract(c),face=row['id'],face_accepted=True,rejected='person_overlap'))
            save_records(db,c,[(bid,metadata,np.ones(4096),None)])
        db.commit();ns=db.execute('SELECT namespace FROM people LIMIT 1').fetchone()[0]
        return dict(body_rejected=db.execute("SELECT COUNT(*) FROM body_observations WHERE rejected='person_overlap'").fetchone()[0],face_projection_ready=db.execute("SELECT COUNT(*) FROM person_evidence WHERE status='ready'").fetchone()[0],hybrid_support=body_rows(db,c,'scope',ns,people[0]).support,decisions=[d['outcome'] for d in report(db,c,'scope',ns)['decisions']])


def _keypoint_admission(p):
    from stash_ai.recognition.sapiensid_backend import SapiensIDFaces
    from stash_ai.recognition.body_collection import quality
    from stash_ai.recognition.person_anchor import anchor_gate
    from tests.helpers.recognition import textured_image
    from threading import get_ident
    c=configuration(p);model=SapiensIDFaces.__new__(SapiensIDFaces);model.owner=get_ident()
    points=[[140,20],[135,25],[145,25],[50,40],[70,40],[40,80],[80,80]]+[[-1,-1]]*10
    person=dict(bounds=[0,0,160,160],keypoints=points,score=1.)
    with textured_image() as image:
        measured,_=quality(image,person,[person],c.data['sapiensid']['collection'])
        _,used=model.person_geometry(image,person)
    return dict(measured=measured['keypoints'],used=sum(min(v)>=0 for v in used[:17]),minimum=c.data['sapiensid']['identities']['anchor_min_keypoints'],anchor=anchor_gate(dict(quality=measured),c.data['sapiensid']['identities'])['eligible'])


def _greedy_support(p):
    from stash_ai.recognition.body_identity import independent
    c=configuration(p)
    rows=[dict(id=str(i),scan='s',cue=cue,quality=encoded(dict(quality=dict(pixel_digest=pixels)))) for i,(cue,pixels) in enumerate([(0,'a'),(0,'b'),(1,'a'),(2,'c')])]
    return dict(support=len(independent(rows,c.data['sapiensid']['identities'])),possible_independent_ids=['1','2','3'])

def test_hybrid_counts_body_pixels_independently_of_faces(tmp_path):
    result=_hybrid_pixel_units(tmp_path)
    assert result['duplicate_body']['independent']==1
    assert 'merge' not in result['duplicate_body']['outcomes']
    assert result['distinct_body_but_same_face']==3


def test_perceptual_dedup_never_discards_a_global_veto(tmp_path):
    result=_perceptual_veto(tmp_path)
    assert result['reported']==[result['real_min']]
    assert 'merge' not in result['outcomes']


def test_original_query_core_must_keep_its_own_current_support(tmp_path):
    result=_history_support(tmp_path)
    assert result['merged']==['merge']
    assert result['recovered']==0
    assert any(0 in counts for _,counts in result['cores'])


def test_rejected_collection_cannot_reenter_as_ready_face_projection(tmp_path):
    result=_rejected_projection(tmp_path)
    assert result['body_rejected']==6
    assert result['face_projection_ready']==result['hybrid_support']==0
    assert 'merge' not in result['decisions']


def test_anchor_counts_pose_points_actually_used_by_encoder(tmp_path):
    result=_keypoint_admission(tmp_path)
    assert result['measured']==result['used']==4
    assert result['minimum']==6
    assert not result['anchor']


def test_body_support_uses_maximum_cue_pixel_matching(tmp_path):
    assert _greedy_support(tmp_path)['support']==3


@pytest.mark.parametrize('same_body,expected',[(True,0),(False,3)])
def test_hybrid_query_support_uses_person_pixels_and_cues(tmp_path,same_body,expected):
    from stash_ai.recognition.hybrid_queries import recover
    c,people=seeded(tmp_path)
    with FaceDatabase(c.database) as state:
        db=state.connection;c=enable(c,db)
        ids=[r[0] for r in db.execute('SELECT observation FROM members WHERE person=?',(people[1],))]
        db.execute('DELETE FROM members WHERE person=?',(people[1],))
        db.execute("UPDATE candidates SET status='withheld',person=NULL WHERE person=?",(people[1],))
        for index,oid in enumerate(ids):
            db.execute("UPDATE person_evidence SET metadata=json_set(metadata,'$.quality.pixel_digest',?) WHERE observation=?",
                       ('a'*64 if same_body else f'{index:064x}',oid))
            db.execute("UPDATE observations SET quality=json_set(quality,'$.evidence_pixel_sha256',?),seconds=0 WHERE id=?",('b'*64,oid))
        db.commit()
        selected=[dict(scan=r['id'],scene=r['scene']) for r in db.execute('SELECT * FROM scans')]
        assert recover(state,selected,c)==expected


def test_old_ready_projection_still_obeys_retained_overlap_measurement(tmp_path):
    from stash_ai.recognition.hybrid_references import body_rows
    c,people=seeded(tmp_path)
    with FaceDatabase(c.database) as state:
        db=state.connection;c=enable(c,db)
        db.execute("UPDATE person_evidence SET metadata=json_set(metadata,'$.quality.overlap',0.9)");db.commit()
        ns=db.execute('SELECT namespace FROM people LIMIT 1').fetchone()[0]
        assert body_rows(db,c,'scope',ns,people[0]).support==0


def test_perceptual_pixel_representatives_preserve_alternative_cue_edges(tmp_path):
    from stash_ai.recognition.body_identity import independent
    c=configuration(tmp_path);settings=c.data['sapiensid']['identities'];settings['min_appearance_distance']=4
    rows=[dict(id=str(i),scan='s',cue=cue,quality=encoded({'quality':{'pixel_digest':pixel,'appearance_hash':hashed}}))
          for i,(cue,pixel,hashed) in enumerate([(0,'a','0'),(0,'b','f'),(1,'a','0'),(2,'c','ff')])]
    assert [r['id'] for r in independent(rows,settings)]==['1','2','3']
    assert [r['id'] for r in independent(rows+rows,settings)]==['1','2','3']
