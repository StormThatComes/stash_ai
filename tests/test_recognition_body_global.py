import math
import numpy as np
import pytest
from stash_ai.recognition.database import FaceDatabase,encoded
from stash_ai.recognition.body_identity import rebuild
from stash_ai.recognition.body_global import report
from tests.test_recognition_body_identity import configuration,scene


def configured(tmp_path):
    c=configuration(tmp_path);c.data['sapiensid']['global_matching']['enabled']=True
    return c


def test_global_body_merge_missing_faces_and_stable_rerun(tmp_path):
    c=configured(tmp_path)
    with FaceDatabase(c.database) as state:
        a,_=scene(state.connection,c,'1');b,_=scene(state.connection,c,'2');state.connection.commit()
        selected=[a,b]
        def run():return rebuild(state,selected,c,{a['scan']:[],b['scan']:[]})
        result=run();people={g['person'] for groups in result.values() for g in groups if g['status']=='accepted'}
        assert len(people)==1
        aliases=list(state.connection.execute('SELECT * FROM identity_merges'));assert len(aliases)==1
        result=run();assert {g['person'] for groups in result.values() for g in groups if g['status']=='accepted'}==people
        assert state.connection.execute('SELECT COUNT(*) FROM identity_merges').fetchone()[0]==1
        assert state.connection.execute("SELECT COUNT(*) FROM body_members WHERE role='auxiliary'").fetchone()[0]==3


def test_incomplete_transitive_component_is_blocked(tmp_path):
    c=configured(tmp_path);c.data['sapiensid']['global_matching'].update(match_threshold=.9,cohesion_threshold=.75,ambiguity_margin=0)
    with FaceDatabase(c.database) as state:
        selected=[]
        for i,angle in enumerate((0,.4,.8),1):
            item,_=scene(state.connection,c,str(i),vectors=[[math.cos(angle),math.sin(angle)]]*3);selected.append(item)
        state.connection.commit();rebuild(state,selected,c,{i['scan']:[] for i in selected})
        assert state.connection.execute('SELECT COUNT(*) FROM identity_merges').fetchone()[0]==0
        result=report(state.connection,c,'scope','ns')
        assert any('incomplete_body_component' in r['blockers'] for r in result['decisions'])


def test_partial_scope_reports_but_does_not_apply_merge(tmp_path):
    c=configured(tmp_path)
    with FaceDatabase(c.database) as state:
        a,_=scene(state.connection,c,'1');b,_=scene(state.connection,c,'2');state.connection.commit()
        rebuild(state,[a],c,{a['scan']:[]});rebuild(state,[b],c,{b['scan']:[]})
        assert state.connection.execute('SELECT COUNT(*) FROM identity_merges').fetchone()[0]==0
        assert any(d['outcome']=='merge' for d in report(state.connection,c,'scope','ns')['decisions'])


def test_global_disabled_retains_separate_scene_cores(tmp_path):
    c=configuration(tmp_path)
    with FaceDatabase(c.database) as state:
        a,_=scene(state.connection,c,'1');b,_=scene(state.connection,c,'2');state.connection.commit()
        result=rebuild(state,[a,b],c,{a['scan']:[],b['scan']:[]})
        assert len({g['person'] for groups in result.values() for g in groups})==2


def test_weak_rival_still_blocks_close_pair(tmp_path):
    c=configured(tmp_path);c.data['sapiensid']['identities']['min_support']=2
    with FaceDatabase(c.database) as state:
        a,_=scene(state.connection,c,'1');b,_=scene(state.connection,c,'2')
        rival,_=scene(state.connection,c,'3',vectors=[[.99,.1]]*2)
        state.connection.commit();selected=[a,b,rival]
        rebuild(state,selected,c,{i['scan']:[] for i in selected})
        assert state.connection.execute('SELECT COUNT(*) FROM identity_merges').fetchone()[0]==0
        assert any('body_ambiguity' in r['blockers'] for r in report(state.connection,c,'scope','ns')['decisions'])


def test_contradictory_faces_veto_matching_bodies(tmp_path):
    c=configuration(tmp_path)
    from stash_ai.recognition.body_identity import policy
    with FaceDatabase(c.database) as state:
        db=state.connection
        a,_=scene(db,c,'1');b,_=scene(db,c,'2');db.commit()
        result=rebuild(state,[a,b],c,{a['scan']:[],b['scan']:[]})
        for i,item in enumerate((a,b)):
            person=result[item['scan']][0]['person'];identity='face-'+str(i)
            feature=np.array([1.,0.] if i==0 else [-1.,0.],dtype='<f4')
            db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                (identity,item['scan'],0,0.,'','[]','[]','{}',None,2,feature.tobytes(),None))
            db.execute('INSERT INTO members VALUES (?,?)',(identity,person))
        c.data['sapiensid']['global_matching']['enabled']=True
        db.execute('UPDATE body_members SET policy=?',(policy(c),));db.commit()
        result=report(db,c,'scope','ns')
        assert 'face_contradiction' in result['pairs'][0]['blockers']
        assert result['pairs'][0]['face_minimum']==-1.


def test_source_attachments_do_not_add_support_on_later_merge(tmp_path):
    c=configured(tmp_path)
    with FaceDatabase(c.database) as state:
        a,_=scene(state.connection,c,'1');b,_=scene(state.connection,c,'2');state.connection.commit()
        rebuild(state,[a,b],c,{a['scan']:[],b['scan']:[]})
        from stash_ai.recognition.body_global import banks
        bank=banks(state.connection,c,'scope','ns')
        assert len(bank)==1
        assert next(iter(bank.values()))['support']==3
        assert len(next(iter(bank.values()))['cores'])>=2
