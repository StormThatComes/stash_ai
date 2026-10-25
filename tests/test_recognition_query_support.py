"""Query votes may cross frozen prototypes without chaining or hiding rivals."""
import math
import json
import numpy as np
import pytest

from stash_ai.recognition.clustering import Group
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.global_views import IdentityView
from stash_ai.recognition.historical_references import History,rank
from stash_ai.recognition.query_support import cohorts
from stash_ai.recognition.rebuild import rebuild_selected
from tests.test_recognition_deterministic_resolution import configured
from tests.test_recognition_frozen_partition import fixture
from tests.test_recognition_rebuild import changed


def appearance(theta, phi=0):
    t,p=map(math.radians,(theta,phi))
    return np.array([math.cos(t),math.sin(t)*math.cos(p),math.sin(t)*math.sin(p),0.],dtype='float32')


def row(identifier,theta,*,person='owner',prototype=None,seconds=None):
    return dict(id=str(identifier),scan='scan',seconds=identifier if seconds is None else seconds,
                dimensions=4,embedding=appearance(theta).astype('<f4').tobytes(),geometry='[]',
                person=person,prototype=prototype or str(identifier))


def supported(config,rows):
    return list(cohorts(rows,config,key=lambda r:r['person'],prototype=lambda r:r['prototype']))


def test_coherent_queries_combine_different_frozen_prototypes(tmp_path):
    groups=supported(configured(tmp_path),[row(1,0),row(2,12)])
    assert len(groups)==1 and groups[0].observations==('1','2')
    assert groups[0].independent_support==2 and len(groups[0].prototypes)==2


@pytest.mark.parametrize('rows',[
    [row(1,0),row(2,50)],
    [row(1,0),row(2,0,person='other')],
    [row(1,0,seconds=1),row(2,0,seconds=1)],
])
def test_different_looks_people_and_duplicate_times_do_not_pool(tmp_path,rows):
    assert supported(configured(tmp_path),rows)==[]


def test_overlapping_hypotheses_keep_separate_frozen_proofs(tmp_path):
    groups=supported(configured(tmp_path),[row(1,0),row(2,17),row(3,34)])
    assert {i for g in groups for i in g.observations}=={'1','2','3'}
    assert all(len(g.witnesses)==2 for g in groups)
    assert not any(set(g.witnesses)=={'1','2','3'} for g in groups)


def test_existing_single_prototype_votes_are_preserved(tmp_path):
    groups=supported(configured(tmp_path),[row(1,0,prototype='same'),row(2,50,prototype='same')])
    assert {i for g in groups for i in g.observations}=={'1','2'}


def test_query_cohort_memory_overflow_is_explicit(tmp_path):
    config=configured(tmp_path)
    config=changed(config,clustering={'max_reference_bytes':16})
    with pytest.raises(ValueError,match='Frozen query cohorts'):
        supported(config,[row(1,0),row(2,50)])


def test_partial_view_rival_cannot_disappear_from_whole_query_ranking(tmp_path):
    def view(person,angle):
        x=appearance(angle)
        return IdentityView(Group(person,(x,),True),'other',person,(person,),3,(x,),'verified_view')
    config=configured(tmp_path)
    history=History((),(view('owner',0),view('rival',20)),{},'partial-rival')
    rows=[row(1,20),row(2,-20)]
    assert rank([rows[0]],history,config) is None
    assert rank([rows[1]],history,config)[0]=='owner'
    decisions=[]
    assert rank(rows,history,config,decisions=decisions) is None
    assert decisions[0]['outcome']=='per_query_competing_identity'


@pytest.mark.parametrize('coherent',[False,True])
@pytest.mark.parametrize('current_vote',[False,True])
def test_historical_queries_pool_across_two_confirmed_views(tmp_path,coherent,current_vote):
    config=configured(tmp_path)
    disabled=changed(config,recognition={'cross_scene_provisional':False})
    source_phi,query_phi=(15,6) if coherent else (55,55)
    with FaceDatabase(config.database) as db:
        source=fixture(db,config,'1',[appearance(0)]*3+
            [appearance(45,source_phi)]*3+[appearance(45,-source_phi)]*3,range(3,9))
        rebuild_selected(db,[source],config)
        owner=db.connection.execute('SELECT person FROM members LIMIT 1').fetchone()[0]
        old=fixture(db,disabled,'2',[appearance(65,query_phi)]);rebuild_selected(db,[old],disabled)
        other=fixture(db,config if current_vote else disabled,'3',[appearance(65,-query_phi)])
        if not current_vote:rebuild_selected(db,[other],disabled)
        selected=[other] if current_vote else [fixture(db,config,'4',[np.array([0.,0.,0.,1.])]*3)]
        rebuild_selected(db,selected,config)
        for item in (old,other):
            rows=db.connection.execute('SELECT c.status,c.person,c.payload FROM candidates c WHERE scan=? '
                'ORDER BY c.rowid DESC LIMIT 1',(item['scan'],)).fetchall()
            assert rows[0]['status']==('accepted' if coherent else 'withheld')
            if coherent:
                assert rows[0]['person']==owner
                payload=json.loads(rows[0]['payload'])
                assert payload['reference_role']=='attachment'
                assert len(payload['provisional_pool']['medoids'])==2


def test_matrix_candidates_preserve_scalar_threshold_boundaries():
    from stash_ai.recognition.query_matrix import QueryMatrix
    samples=[appearance(angle).astype('float32') for angle in (0,10,20,45,80)]
    index=QueryMatrix([(str(i),sample) for i,sample in enumerate(samples)])
    query=samples[1]
    for sample in samples:
        exact=float(np.dot(query,sample))
        for threshold in (exact,np.nextafter(exact,float('inf')),np.nextafter(exact,float('-inf'))):
            expected=[str(i) for i,v in enumerate(samples) if float(np.dot(query,v))>=threshold]
            assert [v[0] for v in index.matches(query,threshold)]==expected
