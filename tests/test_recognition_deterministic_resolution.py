"""Appearance order and a restored external bank cannot manufacture fragments."""
import itertools
import numpy as np
import pytest

from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.rebuild import rebuild_selected
from stash_ai.recognition.historical_references import context
from tests.test_recognition_anchor_views import config_for,A,X,C
from tests.test_recognition_frozen_partition import fixture
from tests.test_recognition_reference_registry import angle,pending
from tests.test_recognition_rebuild import changed
from tests.test_recognition_scene_context import candidates


def configured(tmp_path):
    return changed(config_for(tmp_path,view_prototypes_enabled=True),
                   recognition={'min_identity_support':2,'cross_scene_provisional':True})


@pytest.mark.parametrize('auxiliary',[False,True])
@pytest.mark.parametrize('order',list(itertools.permutations('AXC')))
def test_local_appearance_permutations_keep_one_owner_and_query_authority(tmp_path,auxiliary,order):
    config=configured(tmp_path);lookup={'A':A,'X':X,'C':C}
    vectors=[lookup[k] for k in order for _ in range(3)]
    aux=[j for i,k in enumerate(order) if auxiliary and k=='X' for j in range(i*3,i*3+3)]
    with FaceDatabase(config.database) as db:
        source=fixture(db,config,'1',vectors,aux);rebuild_selected(db,[source],config)
        query=fixture(db,config,'2',[C]*3);rebuild_selected(db,[query],config)
        rows=[r for item in (source,query) for r in candidates(db,item['scan'],config)]
        assert all(r[0]=='accepted' for r in rows)
        assert len({r[1] for r in rows})==1
        assert all(r[2]['reference_role']=='attachment' for r in candidates(db,query['scan'],config))
        before=[tuple(r) for r in db.connection.execute('SELECT * FROM members ORDER BY observation')]
        rebuild_selected(db,[source,query],config)
        assert before==[tuple(r) for r in db.connection.execute('SELECT * FROM members ORDER BY observation')]
        assert all(r[2]['reference_role']=='attachment' for r in candidates(db,query['scan'],config))


def test_close_existing_view_defers_birth_without_lowering_match_threshold(tmp_path):
    config=configured(tmp_path)
    with FaceDatabase(config.database) as db:
        source=fixture(db,config,'1',[A]*3+[X]*3,range(3,6));rebuild_selected(db,[source],config)
        query=fixture(db,config,'2',[angle(82)]*2);rebuild_selected(db,[query],config)
        assert all(r[0]=='withheld' and r[1] is None and r[2]['reason']=='historical_potential_identity'
                   for r in candidates(db,query['scan'],config))
        assert db.connection.execute('SELECT COUNT(*) FROM people').fetchone()[0]==1
        again=fixture(db,config,'3',[angle(82)]*2);rebuild_selected(db,[again],config)
        assert not db.connection.execute('SELECT 1 FROM members WHERE observation IN (SELECT id FROM observations WHERE scan IN (?,?))',
                                         (query['scan'],again['scan'])).fetchone()
        unrelated=fixture(db,config,'4',[np.array([0.,0.,1.,0.])]*2);rebuild_selected(db,[unrelated],config)
        assert all(r[0]=='accepted' for r in candidates(db,unrelated['scan'],config))


def test_restored_bank_recovers_old_pending_views_during_unrelated_selection(tmp_path):
    config=configured(tmp_path)
    with FaceDatabase(config.database) as db:
        source=fixture(db,config,'3',[A]*3+[X]*3);rebuild_selected(db,[source],config)
        owner=candidates(db,source['scan'],config)[0][1]
        replacement=pending(db,source)
        old=[]
        for scene in ('1','2'):
            item=fixture(db,config,scene,[C]);old.append(item);rebuild_selected(db,[item],config)
        assert all(r[0]=='withheld' for item in old for r in candidates(db,item['scan'],config))
        db.connection.execute('DELETE FROM scans WHERE id=?',(replacement,));db.connection.commit()
        other=fixture(db,config,'4',[np.array([0.,0.,1.,0.])]*3);rebuild_selected(db,[other],config)
        assert all(r[0]=='accepted' and r[1]==owner and r[2]['reference_role']=='attachment'
                   for item in old for r in candidates(db,item['scan'],config))
        before=[tuple(r) for r in db.connection.execute('SELECT * FROM members ORDER BY observation')]
        rebuild_selected(db,[other],config)
        assert before==[tuple(r) for r in db.connection.execute('SELECT * FROM members ORDER BY observation')]


def test_missing_view_query_votes_are_not_borrowed_from_reference_cues(tmp_path):
    config=configured(tmp_path)
    with FaceDatabase(config.database) as db:
        source=fixture(db,config,'2',[A]*3+[X]*3);rebuild_selected(db,[source],config)
        replacement=pending(db,source)
        old=fixture(db,config,'1',[C]);rebuild_selected(db,[old],config)
        db.connection.execute('DELETE FROM scans WHERE id=?',(replacement,));db.connection.commit()
        other=fixture(db,config,'3',[np.array([0.,0.,1.,0.])]*3);rebuild_selected(db,[other],config)
        assert all(r[0]=='withheld' for r in candidates(db,old['scan'],config))


def test_final_auxiliary_owner_revokes_selected_whole_view_claim(tmp_path):
    config=configured(tmp_path)
    x=np.array([1.,0.,0.,0.]);a=np.array([.7,np.sqrt(.51),0.,0.])
    c=np.array([.914,0.,np.sqrt(1-.914**2),0.])
    r=np.array([.81,-.3,-np.sqrt(1-.81**2-.3**2),0.])
    with FaceDatabase(config.database) as db:
        rival=fixture(db,config,'1',[r]*3);rebuild_selected(db,[rival],config)
        source=fixture(db,config,'2',[a]*3+[x]*3,range(3,6))
        query=fixture(db,config,'3',[c]*3);rebuild_selected(db,[source,query],config)
        assert all(row[0]=='withheld' and row[2]['reason']=='shared_view_source_invalidated'
                   for row in candidates(db,query['scan'],config))


def test_historical_bank_combines_old_and_current_view_queries(tmp_path):
    config=configured(tmp_path)
    disabled=changed(config,recognition={'cross_scene_provisional':False})
    with FaceDatabase(config.database) as db:
        old=fixture(db,disabled,'1',[C]);rebuild_selected(db,[old],disabled)
        source=fixture(db,disabled,'2',[A]*3+[X]*3);rebuild_selected(db,[source],disabled)
        owner=candidates(db,source['scan'],disabled)[0][1]
        current=fixture(db,config,'3',[C]);rebuild_selected(db,[current],config)
        assert all(r[0]=='accepted' and r[1]==owner and r[2]['reference_role']=='attachment'
                   for item in (old,current) for r in candidates(db,item['scan'],config))


def test_historical_pool_backfill_rolls_back_with_selected_rebuild(tmp_path,monkeypatch):
    from stash_ai.recognition import unresolved_pool
    config=configured(tmp_path)
    disabled=changed(config,recognition={'cross_scene_provisional':False})
    with FaceDatabase(config.database) as db:
        old=fixture(db,disabled,'1',[A]);rebuild_selected(db,[old],disabled)
        source=fixture(db,disabled,'2',[A]*3);rebuild_selected(db,[source],disabled)
        query=fixture(db,config,'3',[np.array([0.,0.,1.,0.])]*3)
        original=unresolved_pool.apply
        def fail(*args,**kwargs):
            assert original(*args,**kwargs)
            raise RuntimeError('after historical backfill')
        monkeypatch.setattr(unresolved_pool,'apply',fail)
        before=[tuple(r) for r in db.connection.execute('SELECT * FROM members ORDER BY observation')]
        with pytest.raises(RuntimeError,match='historical backfill'):rebuild_selected(db,[query],config)
        assert before==[tuple(r) for r in db.connection.execute('SELECT * FROM members ORDER BY observation')]
        monkeypatch.setattr(unresolved_pool,'apply',original)
        rebuild_selected(db,[query],config)
        assert all(r[0]=='accepted' and r[2]['reason']=='late_strict_identity' for r in candidates(db,old['scan'],config))


@pytest.mark.parametrize('current_query',[False,True])
def test_final_source_reassignment_cannot_demote_a_valid_historical_core(tmp_path,current_query):
    config=configured(tmp_path)
    x=np.array([1.,0.,0.,0.]);a=np.array([.7,np.sqrt(.51),0.,0.])
    c=np.array([.914,0.,np.sqrt(1-.914**2),0.]);r=np.array([.81,-.3,-np.sqrt(1-.81**2-.3**2),0.])
    with FaceDatabase(config.database) as db:
        old=fixture(db,config,'1',[c]*3);rebuild_selected(db,[old],config)
        owner=candidates(db,old['scan'],config)[0][1]
        rival=fixture(db,config,'2',[r]*3);rebuild_selected(db,[rival],config)
        source=fixture(db,config,'3',[a]*3+[x]*3,range(3,6));selected=[source]
        if current_query:selected.append(fixture(db,config,'4',[c]*3))
        rebuild_selected(db,selected,config)
        assert all(row[0]=='accepted' and row[1]==owner and row[2]['reference_role']=='strict'
                   for row in candidates(db,old['scan'],config))
        rows=candidates(db,source['scan'],config)
        assert all(row[0]=='withheld' for row in rows[:3])
        assert all(row[0]=='accepted' and row[1]==owner for row in rows[3:])
        assert db.connection.execute('SELECT COUNT(*) FROM people').fetchone()[0]==2


def test_persistently_unstable_sources_never_commit_rebuilt_memberships(tmp_path,monkeypatch):
    from stash_ai.recognition import reference_registry
    config=configured(tmp_path)
    with FaceDatabase(config.database) as db:
        item=fixture(db,config,'1',[A]*3)
        monkeypatch.setattr(reference_registry,'invalidated_sources',lambda *a:{'unstable-source'})
        with pytest.raises(reference_registry.InvalidatedSourceViews,match='no rebuilt memberships'):
            rebuild_selected(db,[item],config)
        assert not db.connection.execute('SELECT 1 FROM members').fetchone()
        assert not db.connection.execute('SELECT 1 FROM people').fetchone()
        assert not db.connection.execute('SELECT 1 FROM candidates').fetchone()


def test_stronger_own_view_does_not_remove_independent_strict_authority(tmp_path):
    config=configured(tmp_path)
    with FaceDatabase(config.database) as db:
        source=fixture(db,config,'1',[A]*3+[X]*3);rebuild_selected(db,[source],config)
        owner=candidates(db,source['scan'],config)[0][1]
        query=fixture(db,config,'2',[angle(20)]*3);rebuild_selected(db,[query],config)
        assert all(r[0]=='accepted' and r[1]==owner and r[2]['reference_role']=='strict'
                   for r in candidates(db,query['scan'],config))


def test_valid_strict_margin_is_not_replaced_by_own_view_margin(tmp_path):
    from stash_ai.recognition.clustering import Group
    from stash_ai.recognition.historical_references import History,rank
    from stash_ai.recognition.global_views import IdentityView
    config=configured(tmp_path);q=angle(20)
    rival=.8205*q+np.sqrt(1-.8205**2)*np.array([0.,0.,1.,0.])
    view=IdentityView(Group('owner',(A,),True),'other','a',('a','b','c'),3,(A,))
    history=History((Group('owner',(A,X),True),Group('rival',(rival,),True)),(view,),{},'independent-strict')
    row=dict(geometry='[]',dimensions=4,embedding=q.astype('<f4').tobytes())
    result=rank([row],history,config)
    assert result is not None and result[0]=='owner' and result[1]['mode']=='strict'
    assert result[1]['ranking_score']>result[1]['mean_score']
    assert result[1]['ambiguity_margin']==config.data['recognition']['ambiguity_margin']


def test_below_threshold_own_view_cannot_hide_a_valid_strict_match(tmp_path):
    from stash_ai.recognition.clustering import Group
    from stash_ai.recognition.historical_references import History,rank
    from stash_ai.recognition.global_views import IdentityView
    config=configured(tmp_path);q=X
    appearance=.81*q+np.sqrt(1-.81**2)*np.array([0.,0.,1.,0.])
    view=IdentityView(Group('owner',(appearance,),True),'other','a',('a','b','c'),3,(appearance,))
    history=History((Group('owner',(A,appearance),True),),(view,),{},'strict-with-near-view')
    row=dict(geometry='[]',dimensions=4,embedding=q.astype('<f4').tobytes())
    result=rank([row],history,config)
    assert result is not None and result[0]=='owner' and result[1]['mode']=='strict'
