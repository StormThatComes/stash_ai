from dataclasses import replace
import json
import uuid
import numpy as np
import pytest
from stash_ai.recognition.database import FaceDatabase,encoded
from stash_ai.recognition.body_observations import contract
from stash_ai.recognition.body_identity import rebuild,scene_plan,independent
from tests.test_recognition_body_collection import settings


def configuration(tmp_path):
    c=settings(tmp_path);c.data['sapiensid']['identities']['enabled']=True
    c.data['sapiensid']['identities']['min_appearance_distance']=0
    return c


def scene(db,c,scene='1',vectors=None,times=None,faces=None):
    vectors=vectors or [[1,0]]*3;times=times or list(range(len(vectors)));faces=faces or [None]*len(vectors)
    scan=str(uuid.uuid5(uuid.NAMESPACE_URL,'scene-'+scene))
    db.execute('INSERT INTO scans VALUES (?,?,?,?,?,?,?,?,?,?,?)',
        (scan,'scope',scene,'ns','input','policy','{}','{}',len(set(times)),len(set(times)),'complete'))
    ids=[]
    for i,(v,t,face) in enumerate(zip(vectors,times,faces)):
        identity=str(uuid.uuid5(uuid.UUID(scan),str(i)));ids.append(identity)
        a=np.zeros(4096,dtype='<f4');a[:len(v)]=v;a/=np.linalg.norm(a)
        q={'quality':{'keypoints':10,'sharpness':100.,'pixel_digest':f'{scene}-{i}'}}
        db.execute('INSERT INTO body_observations VALUES (?,?,?,?,?,?,?,?,?,?,?)',
            (identity,scan,t,float(t),contract(c),face,'{}',encoded(q),None,a.tobytes(),None))
        db.execute('INSERT OR IGNORE INTO body_cues VALUES (?,?,?)',(scan,contract(c),t))
    return dict(scan=scan,scene=scene),ids


def test_production_rebuild_creates_faceless_identity_and_reuses_label(tmp_path):
    c=configuration(tmp_path)
    with FaceDatabase(c.database) as state:
        item,ids=scene(state.connection,c);state.connection.commit()
        result=rebuild(state,[item],c,{item['scan']:[]})
        assert result[item['scan']][0]['status']=='accepted'
        person=result[item['scan']][0]['person']
        again=rebuild(state,[item],c,{item['scan']:[]})
        assert again[item['scan']][0]['person']==person
        from stash_ai.recognition.publication_queue import snapshot
        assert snapshot(state.connection,'scope','ns','1')[1]==[person]
        assert state.connection.execute('SELECT COUNT(*) FROM members').fetchone()[0]==0


def test_copresent_people_do_not_join_and_provisional_not_published(tmp_path):
    c=configuration(tmp_path)
    with FaceDatabase(c.database) as state:
        item,ids=scene(state.connection,c,vectors=[[1,0]]*4,times=[0,0,1,1]);state.connection.commit()
        result=rebuild(state,[item],c,{item['scan']:[]})
        assert not any(r['status']=='accepted' for r in result[item['scan']])
        assert len({r[0] for r in state.connection.execute('SELECT person FROM body_members WHERE person IS NOT NULL')})>=2


def test_auxiliary_cannot_confirm_new_identity(tmp_path):
    c=configuration(tmp_path)
    with FaceDatabase(c.database) as state:
        item,ids=scene(state.connection,c)
        state.connection.execute('UPDATE body_observations SET quality=? WHERE id IN (?,?)',
            (encoded({'quality':{'keypoints':4,'sharpness':10}}),*ids[1:]));state.connection.commit()
        result=rebuild(state,[item],c,{item['scan']:[]})
        assert not any(r['status']=='accepted' for r in result[item['scan']])


def test_support_counts_cues_and_deduplicates_pixels_regardless_of_seconds(tmp_path):
    c=configuration(tmp_path);p=c.data['sapiensid']['identities']
    rows=[dict(id=str(i),scan=s,cue=cue,seconds=0,quality=encoded({'quality':{'pixel_digest':pixels}}))
          for i,(s,cue,pixels) in enumerate([('s',0,'a'),('s',0,'b'),('s',1,'c'),
                                            ('s',2,'a'),('t',0,'d')])]
    assert [r['id'] for r in independent(rows,p)]==['0','2','4']
    for i,row in enumerate(rows):row['seconds']=100-i*10
    assert [r['id'] for r in independent(reversed(rows),p)]==['0','2','4']


def test_perceptual_deduplication_remains_separate_from_cue_support(tmp_path):
    c=configuration(tmp_path);p=c.data['sapiensid']['identities']
    rows=[dict(id=str(i),scan='s',cue=i,seconds=0,quality=encoded({'quality':{
        'pixel_digest':pixels,'appearance_hash':hashed}}))
        for i,(pixels,hashed) in enumerate([('a','0'),('b','1'),('c','f'),('a','ffff')])]
    p['min_appearance_distance']=4
    assert [r['id'] for r in independent(rows,p)]==['0','2']
    p['min_appearance_distance']=0
    assert [r['id'] for r in independent(rows,p)]==['0','1','2']


@pytest.mark.parametrize('spacing',[0,.01,30])
def test_production_support_does_not_depend_on_cue_timing(tmp_path,spacing):
    c=configuration(tmp_path)
    with FaceDatabase(c.database) as state:
        item,ids=scene(state.connection,c)
        state.connection.execute('UPDATE body_observations SET seconds=cue*?',(spacing,))
        state.connection.commit()
        result=rebuild(state,[item],c,{item['scan']:[]})
        assert result[item['scan']][0]['status']=='accepted'
        assert state.connection.execute("SELECT COUNT(*) FROM body_members WHERE role='anchor'").fetchone()[0]==3


def test_incomplete_body_input_aborts_without_replacing_members(tmp_path):
    c=configuration(tmp_path)
    with FaceDatabase(c.database) as state:
        item,ids=scene(state.connection,c);state.connection.commit()
        rebuild(state,[item],c,{item['scan']:[]})
        before=list(state.connection.execute('SELECT * FROM body_members'))
        state.connection.execute('DELETE FROM body_cues WHERE cue=2');state.connection.commit()
        with pytest.raises(ValueError,match='incomplete'):rebuild(state,[item],c,{item['scan']:[]})
        assert list(state.connection.execute('SELECT * FROM body_members'))==before


def test_reference_budget_fails_before_assignment(tmp_path):
    c=configuration(tmp_path);c.data['sapiensid']['execution']['reference_bytes']=1
    with FaceDatabase(c.database) as state:
        item,ids=scene(state.connection,c);state.connection.commit()
        with pytest.raises(ValueError,match='reference_bytes'):rebuild(state,[item],c,{item['scan']:[]})
        assert state.connection.execute('SELECT COUNT(*) FROM people').fetchone()[0]==0


def test_actual_rebuild_includes_body_results(tmp_path):
    c=configuration(tmp_path)
    from stash_ai.recognition.rebuild import rebuild_selected
    from stash_ai.recognition.scanning import scan_identity
    with FaceDatabase(c.database) as state:
        item,ids=scene(state.connection,c)
        state.connection.execute('UPDATE scans SET policy_sha=?',(scan_identity(c,'ns'),));state.connection.commit()
        result=rebuild_selected(state,[item],c)
        assert any(r['status']=='accepted' and r.get('evidence')=='person' for r in result[item['scan']])
