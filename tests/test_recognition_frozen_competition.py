import json,math,uuid
from dataclasses import replace
from unittest.mock import patch
import numpy as np
import pytest
from stash_ai.recognition.database import FaceDatabase,encoded
from tests.test_recognition_body_identity import configuration,scene


def contradicted_rival(p):
    from stash_ai.recognition.hybrid_evidence import Identity,plan,pair
    c=configuration(p)
    body=lambda x:np.asarray([[x]+[0]*4095],dtype='<f4')
    face=lambda x:np.asarray([[x,0]],dtype='<f4')
    aa=Identity('a',('a',),face(1),np.repeat(body(1),3,axis=0),(1,2,3),body_modes=(np.repeat(body(1),3,axis=0),))
    bb=replace(aa,person='b',observations=('b',))
    cc=Identity('c',('c',),face(-1),np.vstack((body(1),body(-1))),(1,2,3,4,5,6),body_modes=(np.repeat(body(1),3,axis=0),np.repeat(body(-1),3,axis=0)))
    with patch('stash_ai.recognition.global_views._cooccurs',return_value=False):
        result=plan(None,c,[aa,bb,cc])
    return dict(result=result,ac=pair(aa,cc,c.data['sapiensid']['matching'])['blockers'])

def provisional_aux(p):
    from stash_ai.recognition.body_identity import rebuild
    c=configuration(p);c.data['sapiensid']['global_matching']['enabled']=True
    with FaceDatabase(c.database) as state:
        db=state.connection;selected=[];weak=[]
        for i in range(2):
            item,ids=scene(db,c,str(i+1));selected.append(item);weak.append(ids[-1])
            db.execute("UPDATE body_observations SET quality=json_set(quality,'$.quality.keypoints',4,'$.quality.sharpness',10) WHERE id=?",(ids[-1],))
        db.commit();rebuild(state,selected,c,{i['scan']:[] for i in selected})
        before=[tuple(r) for r in db.execute('SELECT status,role,COUNT(*) FROM body_members GROUP BY status,role')]
        rebuild(state,selected,c,{i['scan']:[] for i in selected})
        return dict(before=before,after=[tuple(r) for r in db.execute('SELECT status,role,COUNT(*) FROM body_members GROUP BY status,role')],aliases=[json.loads(r[0])['route'] for r in db.execute('SELECT evidence FROM identity_merges')])

def local_contradicted_seed(p):
    from stash_ai.recognition.body_identity import rebuild
    from tests.test_recognition_assignment_guards import face
    c=configuration(p)
    with FaceDatabase(c.database) as state:
        db=state.connection;item,ids=scene(db,c,vectors=[[1,0]]*6)
        owner=str(uuid.uuid4());db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',(owner,'scope','ns',0,None,'token'))
        for i,bid in enumerate(ids):
            fid=face(db,item['scan'],i,owner if i<3 else None,value=1 if i<3 else -1)
            db.execute('UPDATE body_observations SET face=? WHERE id=?',(fid,bid))
        db.commit();rebuild(state,[item],c,{item['scan']:[]})
        return [dict(r) for r in db.execute('SELECT status,role,json_extract(proof,\'$.reason\') AS reason,COUNT(*) AS n FROM body_members GROUP BY status,role,reason')]

def future_local_rival(p,angle=25,repetitions=3):
    from stash_ai.recognition.body_identity import scene_plan
    c=configuration(p)
    def vec(d):return [math.cos(math.radians(d)),math.sin(math.radians(d))]
    with FaceDatabase(c.database) as state:
        db=state.connection;out={}
        for s,angles in [('early_query',[0]*3+[angle]*repetitions+[angle*2]*3),('late_query',[0]*3+[angle*2]*3+[angle]*repetitions)]:
            item,ids=scene(db,c,s,vectors=[vec(d) for d in angles]);db.commit()
            _,_,decisions=scene_plan(db,item['scan'],c)
            out[s]=[decisions[oid]['status'] for oid,value in zip(ids,angles) if value==angle]
        return dict(similarity_to_both=math.cos(math.radians(angle)),decisions=out)


@pytest.mark.parametrize('angle,repetitions',[(25,3),(29,40)])
def test_later_local_rivals_revalidate_earlier_queries(tmp_path,angle,repetitions):
    result=future_local_rival(tmp_path,angle,repetitions)
    assert all(states==['withheld']*repetitions for states in result['decisions'].values())


def test_confirmed_global_cores_recover_weak_queries_without_training(tmp_path):
    result=provisional_aux(tmp_path)
    assert result['before']==result['after']
    assert sum(n for status,role,n in result['before'] if status=='accepted')==6
    assert sum(n for status,role,n in result['before'] if role=='anchor')==2
    assert result['aliases']==['body_global']


def test_face_contradiction_excludes_unnominated_hybrid_rival(tmp_path):
    result=contradicted_rival(tmp_path)
    assert 'face_contradiction' in result['ac']
    assert any(d['outcome']=='merge' and set(d['people'])=={'a','b'} for d in result['result']['decisions'])


def test_contradicted_seed_allows_independent_body_identity(tmp_path):
    result=local_contradicted_seed(tmp_path)
    assert sum(r['n'] for r in result if r['status']=='accepted')==6
    assert any(r['reason']=='body_scene_core' and r['n']==3 for r in result)


def test_cross_scene_rivals_survive_cutoff_and_weak_edge(tmp_path):
    from stash_ai.recognition.cross_scene import _rank_reference
    from stash_ai.recognition.clustering import Group
    q=np.array([1.,0.],dtype='<f4')
    def v(x):return np.array([x,math.sqrt(1-x*x)],dtype='<f4')
    a=Group('a',(v(.91),),True,('a',))
    b=Group('b',(v(.74),),True,('b',))
    ranked=_rank_reference([q],[a,b],.9,.7)
    assert [r[1] for r in ranked]==['a','b']
    assert ranked[0][0]-ranked[1][0]<.2
    b=Group('b',(v(.99),v(.99),v(.99),v(.5)),True,('b1','b2','b3','b4'))
    ranked=_rank_reference([q],[a,b],.9,.7)
    assert [r[1] for r in ranked]==['a','b']
    assert ranked[0][3] and not ranked[1][3]
    assert ranked[0][0]-ranked[1][0]<.1



def test_weak_queries_cannot_confirm_under_supported_cores(tmp_path):
    from stash_ai.recognition.body_identity import rebuild
    c=configuration(tmp_path);c.data['sapiensid']['global_matching']['enabled']=True
    with FaceDatabase(c.database) as state:
        db=state.connection;selected=[]
        for number in ('1','2'):
            item,ids=scene(db,c,number);selected.append(item)
            for oid in ids[1:]:
                db.execute("UPDATE body_observations SET quality=json_set(quality,'$.quality.keypoints',4) WHERE id=?",(oid,))
        db.commit();rebuild(state,selected,c,{i['scan']:[] for i in selected})
        assert db.execute("SELECT COUNT(*) FROM body_members WHERE status='accepted'").fetchone()[0]==0
        assert db.execute("SELECT COUNT(*) FROM body_members WHERE status='withheld'").fetchone()[0]==4
