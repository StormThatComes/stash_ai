"""Regression cases for full witnesses, inherited vetoes and failed rescans."""
import json, math
import pytest
import numpy as np
from stash_ai.recognition.database import FaceDatabase, encoded
from tests.test_recognition_body_identity import configuration, scene
from tests.test_recognition_merge_report import seeded
from tests.test_recognition_hybrid_recovery import enable

def global_cap(p):
    from stash_ai.recognition.body_identity import rebuild
    from stash_ai.recognition.body_global import report
    c=configuration(p); c.data['sapiensid']['global_matching'].update(enabled=True,max_references=3)
    with FaceDatabase(c.database) as state:
        db=state.connection
        a,ai=scene(db,c,'1',vectors=[[1,0]]*3+[[math.cos(math.pi/6),.5]])
        b,bi=scene(db,c,'2',vectors=[[1,0]]*3+[[math.cos(math.pi/6),-.5]])
        db.commit()
        rebuild(state,[a,b],c,{a['scan']:[],b['scan']:[]})
        return dict(people=db.execute("SELECT COUNT(DISTINCT person) FROM body_members WHERE status='accepted'").fetchone()[0],aliases=[dict(r) for r in db.execute('SELECT * FROM identity_merges')],contradiction=.5)

def unsupported_mode(p):
    from stash_ai.recognition.hybrid_evidence import catalogue, report
    c,people=seeded(p,count=4,clustering={'max_samples_per_person':3})
    with FaceDatabase(c.database) as state:
        db=state.connection;c=enable(c,db)
        good=np.zeros(4096,dtype='<f4');good[0]=1
        poor=good.copy();poor[:2]=[.84,math.sqrt(1-.84**2)]
        meta=dict(person=dict(bounds=[0,0,512,512],score=1,keypoints=[[1,1]]*17))
        db.execute('UPDATE person_evidence SET embedding=?,metadata=?',(good.tobytes(),encoded(meta)))
        last=db.execute('SELECT o.id FROM observations o JOIN members m ON m.observation=o.id WHERE m.person=? ORDER BY cue DESC LIMIT 1',(people[0],)).fetchone()[0]
        db.execute('UPDATE person_evidence SET embedding=?,metadata=? WHERE observation=?',(poor.tobytes(),encoded(dict(person=dict(bounds=[0,0,1,1],score=0,keypoints=[]))),last));db.commit()
        ns=db.execute('SELECT namespace FROM people LIMIT 1').fetchone()[0]
        bank=catalogue(db,c,'scope',ns);r=report(db,c,'scope',ns)
        return dict(outlier=last,selected=[list(x.person_observations) for x in bank],mode_counts=[len(x.body_modes) for x in bank],decisions=r['decisions'],pairs=r['pairs'])

def uncovered_modes(p):
    from stash_ai.recognition.hybrid_evidence import catalogue, report
    c,people=seeded(p,count=9,clustering={'max_samples_per_person':2})
    with FaceDatabase(c.database) as state:
        db=state.connection;c=enable(c,db)
        good=np.zeros(4096,dtype='<f4');good[0]=1
        poor=good.copy();poor[:2]=[.76,math.sqrt(1-.76**2)]
        meta=dict(person=dict(bounds=[0,0,512,512],score=1,keypoints=[[1,1]]*17))
        db.execute('UPDATE person_evidence SET embedding=?,metadata=?',(good.tobytes(),encoded(meta)))
        ids=[(r[0],r[1]) for r in db.execute('SELECT o.id,o.cue FROM observations o JOIN members m ON m.observation=o.id WHERE m.person=? ORDER BY cue',(people[0],))]
        for identifier,cue in ids:
            angle=[-20,20,60][cue//3];v=good.copy();v[:2]=[math.cos(math.radians(angle)),math.sin(math.radians(angle))]
            db.execute('UPDATE person_evidence SET embedding=?,metadata=? WHERE observation=?',(v.tobytes(),encoded(meta if cue<6 else {}),identifier))
        db.commit();ns=db.execute('SELECT namespace FROM people LIMIT 1').fetchone()[0]
        bank=catalogue(db,c,'scope',ns);r=report(db,c,'scope',ns)
        return dict(mode_counts=[len(x.body_modes) for x in bank],outcomes=[d['outcome'] for d in r['decisions']],person_scores=[x['person'] for x in r['pairs']],full_min=.5)

def retired_rival(p):
    from stash_ai.recognition.body_identity import rebuild
    from stash_ai.recognition.body_global import report
    import uuid
    c=configuration(p)
    with FaceDatabase(c.database) as state:
        db=state.connection;items=[]
        for i,angle in enumerate([0,10,15]):
            a,ids=scene(db,c,str(i+1),vectors=[[math.cos(math.radians(angle)),math.sin(math.radians(angle))]]*3);items.append(a)
            from tests.test_recognition_assignment_guards import face
            for j,identifier in enumerate(ids):
                f=face(db,a['scan'],j,value=-1 if i==2 else 1)
                db.execute('UPDATE body_observations SET face=? WHERE id=?',(f,identifier))
        db.commit();rebuild(state,items,c,{a['scan']:[] for a in items})
        c.data['sapiensid']['global_matching']['enabled']=True
        from stash_ai.recognition.body_identity import policy
        db.execute('UPDATE body_members SET policy=?',(policy(c),));db.commit()
        before=report(db,c,'scope','ns')
        row=list(db.execute('SELECT * FROM scans WHERE id=?',(items[-1]['scan'],)).fetchone());row[0]=str(uuid.uuid4());row[4]='changed-input';row[-1]='partial';row[-2]=0
        db.execute('INSERT INTO scans VALUES (?,?,?,?,?,?,?,?,?,?,?)',row);db.commit()
        after=report(db,c,'scope','ns')
        return dict(before=[(d['outcome'],d['blockers']) for d in before['decisions']],after=[(d['outcome'],d['blockers']) for d in after['decisions']])

def lost_guard(p,legacy=False):
    from stash_ai.recognition.hybrid_merge import apply
    from stash_ai.recognition.hybrid_evidence import catalogue, report
    from stash_ai.recognition.clustering import matching_policy
    from stash_ai.recognition.person_signal import save,contract
    import uuid
    c,people=seeded(p,count=4,clustering={'max_samples_per_person':3})
    with FaceDatabase(c.database) as state:
        db=state.connection;c=enable(c,db)
        good=np.zeros(4096,dtype='<f4');good[0]=1
        angled=good.copy();angled[:2]=[math.cos(math.pi/6),.5]
        meta=dict(person=dict(bounds=[0,0,512,512],score=1,keypoints=[[1,1]]*17))
        db.execute('UPDATE person_evidence SET embedding=?,metadata=?',(good.tobytes(),encoded(meta)))
        out=db.execute('SELECT o.id FROM observations o JOIN members m ON m.observation=o.id WHERE m.person=? AND cue=3',(people[1],)).fetchone()[0]
        db.execute('UPDATE person_evidence SET embedding=?,metadata=? WHERE observation=?',(angled.tobytes(),encoded({}),out));db.commit()
        selected=[dict(scan=r['id'],scene=r['scene']) for r in db.execute('SELECT * FROM scans')]
        first=apply(state,selected,c)
        if legacy:
            for alias in db.execute('SELECT source,evidence FROM identity_merges').fetchall():
                proof=json.loads(alias['evidence'])
                for item in proof['proofs']:item.pop('body_witness_observations')
                db.execute('UPDATE identity_merges SET evidence=? WHERE source=?',(encoded(proof),alias['source']))
            db.commit()
        oldscan=db.execute('SELECT * FROM scans LIMIT 1').fetchone();newscan=list(oldscan);newscan[0]=str(uuid.uuid4());newscan[2]='3'
        newperson=str(uuid.uuid4());db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',(newperson,'scope',oldscan['namespace'],3,None,'third'))
        db.execute('INSERT INTO scans VALUES (?,?,?,?,?,?,?,?,?,?,?)',newscan)
        third=good.copy();third[:2]=[math.cos(math.radians(25)),-math.sin(math.radians(25))]
        for i in range(4):
            row=list(db.execute('SELECT * FROM observations LIMIT 1').fetchone());row[0]=str(uuid.uuid4());row[1]=newscan[0];row[2]=i;row[3]=float(i);row[10]=np.array([.7,0,math.sqrt(.51),0],dtype='<f4').tobytes()
            db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',row);db.execute('INSERT INTO members VALUES (?,?)',(row[0],newperson));db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?)',(str(uuid.uuid4()),newscan[0],matching_policy(c),newperson,'accepted',encoded(dict(observation=row[0],reason='candidate_group'))));save(db,row[0],contract(c),meta,third)
        db.commit();after=report(db,c,'scope',oldscan['namespace']);bank=catalogue(db,c,'scope',oldscan['namespace'])
        return dict(first=[d['outcome'] for d in first['decisions']],original_outlier_guarded=any(out in x.guard_person_observations for x in bank),later=[d['outcome'] for d in after['decisions']],worst_body=math.cos(math.radians(55)))

def cross_guards(p):
    from stash_ai.recognition.hybrid_evidence import Identity,pair
    c=configuration(p);core=np.zeros((3,4096),dtype='<f4');core[:,0]=1
    left=core[:1].copy();left[0,:2]=[math.cos(math.pi/6),.5]
    right=core[:1].copy();right[0,:2]=[math.cos(math.pi/6),-.5]
    a=Identity('a',('a1','a2','a3'),np.array([[1,0]],dtype='<f4'),core,tuple(range(3)),guard_persons=left,body_modes=(core,))
    b=Identity('b',('b1','b2','b3'),np.array([[1,0]],dtype='<f4'),core,tuple(range(3)),guard_persons=right,body_modes=(core,))
    r=pair(a,b,c.data['sapiensid']['matching'])
    return dict(blockers=r['blockers'],reported_body=r['person'],guard_cross=float(left[0]@right[0]))

def test_capped_global_bank_cannot_drop_a_contradictory_witness(tmp_path):
    result=global_cap(tmp_path)
    assert result['people']==2
    assert not result['aliases']


def test_unsupported_mode_cannot_be_rebuilt_from_selected_rows(tmp_path):
    result=unsupported_mode(tmp_path)
    assert all(result['outlier'] not in ids for ids in result['selected'])
    assert not any(d['outcome']=='merge' for d in result['decisions'])
    assert 'unsupported_body_mode' in result['pairs'][0]['blockers']


def test_supported_modes_need_complete_coverage_even_with_small_cap(tmp_path):
    result=uncovered_modes(tmp_path)
    assert sorted(result['mode_counts'])==[1,3]
    assert 'merge' not in result['outcomes']


@pytest.mark.parametrize('legacy',[False,True])
def test_complete_witnesses_survive_merge_and_veto_a_later_merge(tmp_path,legacy):
    result=lost_guard(tmp_path,legacy)
    assert result['first']==['merge']
    assert result['original_outlier_guarded']
    assert 'merge' not in result['later']


def test_partial_replacement_cannot_hide_a_historical_rival(tmp_path):
    result=retired_rival(tmp_path)
    assert result['before']==[('blocked',['body_ambiguity'])]
    assert result['after']==[('blocked',['incomplete_reference_scans'])]


def test_both_inherited_cores_must_agree(tmp_path):
    result=cross_guards(tmp_path)
    assert result['guard_cross']<.85
    assert 'original_body_core_contradiction' in result['blockers']


def test_failed_replacement_blocks_rebuild_before_any_assignment_changes(tmp_path):
    import uuid
    from stash_ai.recognition.rebuild import rebuild_selected
    config,people=seeded(tmp_path)
    with FaceDatabase(config.database) as state:
        db=state.connection
        old=list(db.execute('SELECT * FROM scans LIMIT 1').fetchone())
        old[0]=str(uuid.uuid4());old[4]='replacement';old[-1]='partial';old[-2]=0
        db.execute('INSERT INTO scans VALUES (?,?,?,?,?,?,?,?,?,?,?)',old);db.commit()
        selected=[dict(scan=r['id'],scene=r['scene']) for r in db.execute("SELECT * FROM scans WHERE scene<>?",(old[2],))]
        before=list(db.execute('SELECT * FROM members'))
        with pytest.raises(ValueError,match='incomplete replacement scans'):
            rebuild_selected(state,selected,config)
        assert list(db.execute('SELECT * FROM members'))==before
