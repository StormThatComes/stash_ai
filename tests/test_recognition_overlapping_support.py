"""Frozen query hypotheses retain boundary votes without transitive support."""
import itertools
import json

import numpy as np
import pytest

from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.query_support import cohorts
from stash_ai.recognition.rebuild import rebuild_selected
from tests.test_recognition_deterministic_resolution import configured
from tests.test_recognition_frozen_partition import fixture
from tests.test_recognition_query_support import appearance, row, supported
from tests.test_recognition_rebuild import changed
from tests.test_recognition_scene_context import candidates


@pytest.mark.parametrize('order', list(itertools.permutations((0,17,34))))
def test_a_central_witness_recovers_all_three_boundary_votes(tmp_path, order):
    config = changed(configured(tmp_path), clustering={'view_attach_min_support':3})
    groups = supported(config, [row(i + 1, theta) for i, theta in enumerate(order)])
    assert {i for g in groups for i in g.observations} == {'1','2','3'}
    assert all(g.independent_support >= 3 for g in groups)


def test_no_single_frozen_centre_can_borrow_a_chain_to_reach_support(tmp_path):
    config = changed(configured(tmp_path), clustering={'view_attach_min_support':4})
    assert not supported(config, [row(i+1, theta) for i, theta in enumerate((0,17,34,51))])


def test_complete_exact_prototypes_need_only_one_stream_pass(tmp_path):
    config = configured(tmp_path)
    rows = [row(1,0,prototype='same'),row(2,50,prototype='same')]
    calls = []
    def source():
        calls.append(1)
        return iter(rows)
    groups = list(cohorts(source,config,key=lambda r:r['person'],prototype=lambda r:r['prototype']))
    assert len(calls) == 1 and groups[0].observations == ('1','2')


def test_supported_prototype_vectors_are_released_before_the_next_prototype(tmp_path):
    config = changed(configured(tmp_path),clustering={'max_reference_bytes':16})
    groups = supported(config,[row(1,0,prototype='a'),row(2,1,prototype='a'),
                              row(3,50,prototype='b'),row(4,51,prototype='b')])
    assert {i for g in groups for i in g.observations} == {'1','2','3','4'}


def test_all_proofs_are_frozen_before_a_caller_can_mutate_its_source(tmp_path):
    config = configured(tmp_path)
    rows = [row(1,0,prototype='same'),row(2,0,prototype='same'),row(3,12)]
    groups = cohorts(lambda:iter(rows),config,key=lambda r:r['person'],prototype=lambda r:r['prototype'])
    first = next(groups)
    rows.clear()
    remaining = list(groups)
    assert set(first.observations) == {'1','2'}
    assert {i for g in remaining for i in g.observations} == {'3'}
    assert len(remaining[0].witnesses) >= 2


def test_a_consumed_stream_requires_an_explicit_fresh_row_factory(tmp_path):
    with pytest.raises(ValueError,match='row factory'):
        supported(configured(tmp_path),iter([row(1,0),row(2,12)]))


@pytest.mark.parametrize('minimum',[2,3])
@pytest.mark.parametrize('offset',[0,20])
@pytest.mark.parametrize('current_vote',[False,True])
def test_historical_and_selected_boundary_queries_keep_the_same_owner(tmp_path,minimum,offset,current_vote):
    config = changed(configured(tmp_path),clustering={'view_attach_min_support':minimum,
                                                    'corroborated_view_enrichment':False})
    disabled = changed(config,recognition={'cross_scene_provisional':False})
    with FaceDatabase(config.database) as db:
        source = fixture(db,config,'1',[appearance(0)]*3+
            [appearance(45,offset-35)]*3+[appearance(45,offset)]*3+
            [appearance(45,offset+35)]*3,range(3,12))
        rebuild_selected(db,[source],config)
        owner = candidates(db,source['scan'],config)[0][1]
        queries = []
        for i,phi in enumerate((offset-19,offset,offset+19),start=2):
            item = fixture(db,config if current_vote and i==4 else disabled,str(i),[appearance(65,phi)])
            queries.append(item)
            if not current_vote or i!=4:
                rebuild_selected(db,[item],disabled)
        selected = [queries[-1]] if current_vote else [fixture(db,config,'5',[np.array([0.,0.,0.,1.])]*3)]
        rebuild_selected(db,selected,config)
        for query in queries:
            record = db.connection.execute('SELECT status,person,payload FROM candidates WHERE scan=? ORDER BY rowid DESC LIMIT 1',
                                            (query['scan'],)).fetchone()
            assert record['status']=='accepted' and record['person']==owner
            assert json.loads(record['payload'])['reference_role']=='attachment'
