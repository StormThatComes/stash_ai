from tests.test_recognition_hybrid_recovery import ANCHOR_METADATA
"""Independent route support, body appearances and stable split labels."""
import json,math
import numpy as np
from stash_ai.recognition.database import FaceDatabase,encoded
from tests.test_recognition_body_identity import configuration,scene
from tests.test_recognition_merge_report import seeded
from tests.test_recognition_hybrid_recovery import enable

def mixed_votes(p,birth=False):
    from stash_ai.recognition.hybrid_queries import recover
    c,people=seeded(p,vectors=[[1,0,0,0],[1,0,0,0]])
    with FaceDatabase(c.database) as state:
        db=state.connection;c=enable(c,db)
        ids=[r[0] for r in db.execute('SELECT o.id FROM observations o JOIN members m ON m.observation=o.id WHERE m.person=? ORDER BY cue',(people[1],))]
        if not birth:
            db.execute('DELETE FROM members WHERE person=?',(people[1],));db.execute("UPDATE candidates SET person=NULL,status='withheld' WHERE person=?",(people[1],))
        db.execute('DELETE FROM person_evidence WHERE observation IN (?,?)',ids[:2])
        db.execute('UPDATE observations SET embedding=? WHERE id=?',(np.array([.7,math.sqrt(.51),0,0],dtype='<f4').tobytes(),ids[-1]));db.commit()
        selected=[dict(scan=r['id'],scene=r['scene']) for r in db.execute('SELECT * FROM scans')]
        n=recover(state,selected,c,births=(people[1],) if birth else ())
        return dict(accepted=n,remaining_birth=db.execute('SELECT COUNT(*) FROM members WHERE person=?',(people[1],)).fetchone()[0],signals=[r[0] for r in db.execute("SELECT json_extract(payload,'$.hybrid_query.signal') FROM candidates WHERE json_extract(payload,'$.reason')='hybrid_query'")])

def body_gray(p):
    from PIL import Image
    from stash_ai.recognition.body_collection import quality
    c=configuration(p);person=dict(bounds=[0,0,100,100],keypoints=[[1,1]]*17)
    with Image.new('RGB',(100,100),(255,0,0)) as a, Image.new('RGB',(100,100),(0,130,0)) as b:
        qa,_=quality(a,person,[person],c.data['sapiensid']['collection']);qb,_=quality(b,person,[person],c.data['sapiensid']['collection'])
        return dict(rgb_different=a.tobytes()!=b.tobytes(),hash_same=qa['pixel_digest']==qb['pixel_digest'],gray_values=[a.convert('L').getpixel((0,0)),b.convert('L').getpixel((0,0))])

def split_label(p):
    from stash_ai.recognition.body_identity import rebuild
    c=configuration(p)
    with FaceDatabase(c.database) as state:
        db=state.connection;a,ids=scene(db,c,vectors=[[1,0]]+[[math.cos(math.pi/9),math.sin(math.pi/9)]]*5);db.commit()
        rebuild(state,[a],c,{a['scan']:[]})
        old=db.execute('SELECT DISTINCT person FROM body_members').fetchone()[0]
        db.execute('UPDATE people SET performer=? WHERE id=?',('published',old));db.commit()
        c.data['sapiensid']['identities'].update(match_threshold=.99,cohesion_threshold=.98,ambiguity_margin=.01)
        rebuild(state,[a],c,{a['scan']:[]})
        return dict(old=old,new_groups=[dict(r) for r in db.execute('SELECT person,status,count(*) as samples FROM body_members GROUP BY person,status')])

def multimode_seed(p):
    from stash_ai.recognition.body_identity import scene_plan
    from tests.test_recognition_assignment_guards import face
    import uuid
    c=configuration(p)
    with FaceDatabase(c.database) as state:
        db=state.connection;a,ids=scene(db,c,vectors=[[1,0]]*2+[[0,1]]*2+[[1,0]])
        who=str(uuid.uuid4());db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',(who,'scope','ns',0,None,'owner'))
        for i,identifier in enumerate(ids[:4]):
            f=face(db,a['scan'],i,who);db.execute('UPDATE body_observations SET face=? WHERE id=?',(f,identifier))
        _,_,decisions=scene_plan(db,a['scan'],c)
        return dict(owner=who,last=decisions[ids[-1]])

def duplicate_hybrid(p):
    from stash_ai.recognition.hybrid_queries import recover
    from stash_ai.recognition.person_signal import save,contract
    from stash_ai.recognition.clustering import matching_policy
    import uuid
    c,people=seeded(p,vectors=[[1,0,0,0],[.99,math.sqrt(1-.99**2),0,0],[.7,0,math.sqrt(.51),0]],count=3,clustering={'max_samples_per_person':3})
    with FaceDatabase(c.database) as state:
        db=state.connection;c=enable(c,db);c.data['sapiensid']['matching']['ambiguity_margin']=.08
        db.execute('UPDATE candidates SET policy=?',(matching_policy(c),))
        for index,owner in enumerate(people):
            for i,row in enumerate(db.execute('SELECT o.* FROM observations o JOIN members m ON m.observation=o.id WHERE m.person=? ORDER BY cue',(owner,)).fetchall()):
                ang=([0,20,20][i] if index==0 else math.degrees(math.acos(.9)) if index==1 else 0)
                v=np.zeros(4096,dtype='<f4');v[:2]=[math.cos(math.radians(ang)),math.sin(math.radians(ang))]
                save(db,row['id'],contract(c),ANCHOR_METADATA,v)
                if index==0 and i==0:base=dict(row);db.execute("UPDATE observations SET quality=json_set(quality,'$.evidence_pixel_sha256',?) WHERE id=?",('a'*64,row['id']))
        db.execute('DELETE FROM members WHERE person=?',(people[2],));db.execute("UPDATE candidates SET person=NULL,status='withheld' WHERE person=?",(people[2],));db.commit()
        selected=[dict(scan=r['id'],scene=r['scene']) for r in db.execute('SELECT * FROM scans')];before=recover(state,selected,c)
        v=np.zeros(4096,dtype='<f4');v[0]=1
        base=dict(db.execute('SELECT * FROM observations WHERE id=?',(base['id'],)).fetchone())
        for i in range(100):
            row=dict(base,id=str(uuid.uuid4()),cue=i+3,seconds=float(i+3))
            db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',tuple(row.values()));db.execute('INSERT INTO members VALUES (?,?)',(row['id'],people[0]));db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?)',(str(uuid.uuid4()),row['scan'],matching_policy(c),people[0],'accepted',encoded(dict(observation=row['id'],reason='candidate_group'))));save(db,row['id'],contract(c),ANCHOR_METADATA,v)
        db.execute('UPDATE scans SET total=103,cursor=103 WHERE id=?',(base['scan'],));db.commit()
        after=recover(state,selected,c)
        return dict(before=before,after=after,independent_owner_crops=3,repeated_crops=100)

def test_duplicate_pixels_cannot_improve_hybrid_query_margin(tmp_path):
    result=duplicate_hybrid(tmp_path)
    assert result['before']==result['after']==0


def test_face_votes_cannot_supply_body_query_support(tmp_path):
    result=mixed_votes(tmp_path)
    assert result['accepted']==2
    assert result['signals']==['face','face']


def test_faceless_query_can_attach_to_one_supported_owner_appearance(tmp_path):
    result=multimode_seed(tmp_path)
    assert result['last']['person']==result['owner']
    assert result['last']['status']=='accepted'
    assert result['last']['role']=='auxiliary'


def test_largest_split_keeps_the_published_identity(tmp_path):
    result=split_label(tmp_path)
    large=next(r for r in result['new_groups'] if r['samples']==5)
    assert large['person']==result['old'] and large['status']=='accepted'


def test_exact_body_pixels_keep_colour_information(tmp_path):
    result=body_gray(tmp_path)
    assert result['gray_values'][0]==result['gray_values'][1]
    assert result['rgb_different'] and not result['hash_same']


def test_mixed_route_birth_cannot_be_reassigned_partially(tmp_path):
    result=mixed_votes(tmp_path,birth=True)
    assert result['accepted']==0 and result['remaining_birth']==3


def test_body_query_support_is_separate_for_each_frozen_appearance(tmp_path):
    from stash_ai.recognition.hybrid_queries import recover
    from stash_ai.recognition.person_signal import save,contract
    config,people=seeded(tmp_path,vectors=[[1,0,0,0],[.7,math.sqrt(.51),0,0]],count=6)
    with FaceDatabase(config.database) as state:
        db=state.connection;config=enable(config,db)
        for owner,cutoff in zip(people,(3,2)):
            for row in db.execute('SELECT o.id,o.cue FROM observations o JOIN members m ON m.observation=o.id WHERE m.person=?',(owner,)).fetchall():
                vector=np.zeros(4096,dtype='<f4');vector[int(row['cue']>=cutoff)]=1
                save(db,row['id'],contract(config),ANCHOR_METADATA,vector)
        db.execute('DELETE FROM members WHERE person=?',(people[1],))
        db.execute("UPDATE candidates SET person=NULL,status='withheld' WHERE person=?",(people[1],));db.commit()
        selected=[dict(scan=r['id'],scene=r['scene']) for r in db.execute('SELECT * FROM scans')]
        assert recover(state,selected,config)==4
        assert {r[0] for r in db.execute("SELECT json_extract(payload,'$.hybrid_query.independent_query_support') FROM candidates WHERE json_extract(payload,'$.reason')='hybrid_query'")}=={4}


def test_exact_body_pixels_include_dimensions(tmp_path):
    from PIL import Image
    from stash_ai.recognition.body_collection import quality
    config=configuration(tmp_path);digests=[]
    for size in ((64,128),(128,64)):
        person=dict(bounds=[0,0,*size],keypoints=[[1,1]]*17)
        with Image.new('RGB',size,'red') as image:
            measured,_=quality(image,person,[person],config.data['sapiensid']['collection'])
        digests.append(measured['pixel_digest'])
    assert digests[0]!=digests[1]


def test_legacy_gray_banks_remain_rivals_until_recollected(tmp_path):
    from stash_ai.recognition.body_identity import rebuild
    from stash_ai.recognition.body_observations import compatible_contracts
    from stash_ai.recognition.body_global_banks import banks
    config=configuration(tmp_path)
    with FaceDatabase(config.database) as state:
        db=state.connection;item,ids=scene(db,config);db.commit()
        rebuild(state,[item],config,{item['scan']:[]})
        legacy=compatible_contracts(config)[1]
        db.execute('UPDATE body_observations SET contract=?',(legacy,))
        db.execute("UPDATE body_members SET policy='legacy'");db.commit()
        bank=banks(db,config,'scope','ns')
        assert len(bank)==1 and next(iter(bank.values()))['stale_policy']
