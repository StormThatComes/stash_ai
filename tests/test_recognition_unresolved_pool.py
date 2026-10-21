"""Independent unresolved evidence persists across run and policy boundaries."""
import itertools
import uuid
import numpy as np
import pytest

from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.historical_references import context
from stash_ai.recognition.rebuild import rebuild_selected
from tests.test_recognition_anchor_views import config_for,A,X,C
from tests.test_recognition_frozen_partition import fixture
from tests.test_recognition_reference_registry import pending,angle
from tests.test_recognition_rebuild import changed,library
from tests.test_recognition_scene_context import candidates


def configured(tmp_path,minimum=3):
    return changed(config_for(tmp_path),recognition={'min_identity_support':minimum,'cross_scene_provisional':True})


@pytest.mark.parametrize('order',list(itertools.permutations(('1','2','3'))))
def test_separate_sparse_runs_confirm_and_backfill_one_identity(tmp_path,order):
    config=configured(tmp_path)
    with FaceDatabase(config.database) as db:
        selected=[]
        for scene in order:
            item=fixture(db,config,scene,[A]);selected.append(item)
            rebuild_selected(db,[item],config)
        rows=[r for item in selected for r in candidates(db,item['scan'],config)]
        assert all(r[0]=='accepted' for r in rows)
        assert len({r[1] for r in rows})==1
        assert all(r[2]['provisional_pool']['independent_support']==3 for r in rows)
        expected=[tuple(r) for r in db.connection.execute('SELECT * FROM members ORDER BY observation')]
        rebuild_selected(db,[selected[-1]],config)
        assert expected==[tuple(r) for r in db.connection.execute('SELECT * FROM members ORDER BY observation')]
        rebuild_selected(db,selected,config)
        assert all(r[0]=='accepted' for item in selected for r in candidates(db,item['scan'],config))
        assert len({r[1] for item in selected for r in candidates(db,item['scan'],config)})==1


def test_duplicate_cue_times_do_not_supply_extra_independent_support(tmp_path):
    config=configured(tmp_path)
    with FaceDatabase(config.database) as db:
        first=fixture(db,config,'1',[A]*2)
        db.connection.execute('UPDATE observations SET seconds=0,cue=0 WHERE scan=?',(first['scan'],));db.connection.commit()
        rebuild_selected(db,[first],config)
        second=fixture(db,config,'2',[A]);rebuild_selected(db,[second],config)
        assert not db.connection.execute('SELECT 1 FROM members').fetchone()
        third=fixture(db,config,'3',[A]);rebuild_selected(db,[third],config)
        assert db.connection.execute('SELECT COUNT(*) FROM members').fetchone()[0]==4
        assert db.connection.execute('SELECT COUNT(DISTINCT person) FROM members').fetchone()[0]==1


@pytest.mark.parametrize('revoke',['superseded','rejected','namespace'])
def test_unavailable_raw_pending_evidence_cannot_vote(tmp_path,revoke):
    config=configured(tmp_path,2)
    with FaceDatabase(config.database) as db:
        first=fixture(db,config,'1',[A]);rebuild_selected(db,[first],config)
        before=context(db.connection,[],config,'scope','ns')
        if revoke=='superseded':pending(db,first)
        elif revoke=='rejected':
            db.connection.execute("UPDATE observations SET rejected='sharpness' WHERE scan=?",(first['scan'],));db.connection.commit()
        else:
            db.connection.execute("UPDATE scans SET namespace='other' WHERE id=?",(first['scan'],));db.connection.commit()
        assert context(db.connection,[],config,'scope','ns')!=before
        second=fixture(db,config,'2',[A]);rebuild_selected(db,[second],config)
        assert all(r[0]=='withheld' for r in candidates(db,second['scan'],config))
        assert not db.connection.execute('SELECT 1 FROM members').fetchone()


def test_pending_anchors_revalidate_across_scan_quality_policy_change(tmp_path):
    config=configured(tmp_path,2)
    with FaceDatabase(config.database) as db:
        first=fixture(db,config,'1',[A]);rebuild_selected(db,[first],config)
        tuned=changed(config,detection={'min_face_pixels':64})
        second=fixture(db,tuned,'2',[A]);rebuild_selected(db,[second],tuned)
        assert db.connection.execute('SELECT COUNT(*) FROM members').fetchone()[0]==2
        assert db.connection.execute('SELECT COUNT(DISTINCT person) FROM members').fetchone()[0]==1


@pytest.mark.parametrize('vectors',[[A,C,X],[X,A,C],[C,A,X]])
def test_weak_bridge_cannot_establish_pooled_identity(tmp_path,vectors):
    config=configured(tmp_path)
    with FaceDatabase(config.database) as db:
        for i,v in enumerate(vectors,1):
            item=fixture(db,config,str(i),[v]);rebuild_selected(db,[item],config)
        assert not db.connection.execute('SELECT 1 FROM members').fetchone()


def test_auxiliary_and_disabled_pool_do_not_establish_identity(tmp_path):
    config=configured(tmp_path,2)
    with FaceDatabase(config.database) as db:
        first=fixture(db,config,'1',[A]);rebuild_selected(db,[first],config)
        second=fixture(db,config,'2',[A],(0,));rebuild_selected(db,[second],config)
        assert not db.connection.execute('SELECT 1 FROM members').fetchone()
        disabled=changed(config,recognition={'cross_scene_provisional':False})
        third=fixture(db,disabled,'3',[A]);rebuild_selected(db,[third],disabled)
        assert not db.connection.execute('SELECT 1 FROM members').fetchone()


def test_pooled_view_support_remains_attachment_only(tmp_path):
    config=configured(tmp_path)
    with FaceDatabase(config.database) as db:
        source=fixture(db,config,'1',[A]*3+[X]*3);rebuild_selected(db,[source],config)
        owner=candidates(db,source['scan'],config)[0][1]
        queries=[]
        for scene in ('2','3'):
            item=fixture(db,config,scene,[C]);queries.append(item);rebuild_selected(db,[item],config)
        rows=[r for item in queries for r in candidates(db,item['scan'],config)]
        assert all(r[0]=='accepted' and r[1]==owner and r[2]['reference_role']=='attachment' for r in rows)
        assert all(r[2]['reason']=='pooled_view_identity' for r in rows)
        beyond=fixture(db,config,'4',[angle(84)]*3);rebuild_selected(db,[beyond],config)
        assert all(r[1]!=owner for r in candidates(db,beyond['scan'],config))


def test_backfill_and_selected_assignments_roll_back_together(tmp_path,monkeypatch):
    from stash_ai.recognition import unresolved_pool
    config=configured(tmp_path,2)
    with FaceDatabase(config.database) as db:
        first=fixture(db,config,'1',[A]);rebuild_selected(db,[first],config)
        second=fixture(db,config,'2',[A])
        original=unresolved_pool.apply
        def fail(*args,**kwargs):
            assert original(*args,**kwargs)
            raise RuntimeError('after pool writes')
        monkeypatch.setattr(unresolved_pool,'apply',fail)
        with pytest.raises(RuntimeError,match='after pool writes'):rebuild_selected(db,[second],config)
        assert not db.connection.execute('SELECT 1 FROM members').fetchone()
        assert not db.connection.execute('SELECT 1 FROM people').fetchone()
        assert all(r[0]=='withheld' for r in candidates(db,first['scan'],config))
        monkeypatch.setattr(unresolved_pool,'apply',original)
        rebuild_selected(db,[second],config)
        assert db.connection.execute('SELECT COUNT(*) FROM members').fetchone()[0]==2


def test_pool_backfill_preserves_stash_publication_scope(tmp_path,monkeypatch):
    from tests.test_recognition_run import run
    config,client,services,_=library(tmp_path,monkeypatch,{'1':[A],'2':[A]})
    config=changed(config,recognition={'min_identity_support':2,'cross_scene_provisional':True},
        clustering={'max_anchor_pose_deviation':.2,'anchor_view_matching':True})
    assert run(config,services,'1')==0
    assert client.links['1']==['77']
    assert run(config,services,'2')==0
    assert client.links['1']==['77']
    assert len(client.links['2'])==1
    owner=client.links['2'][0]
    assert run(config,services,'1')==0
    assert client.links['1']==['77',owner]
    assert sum(w[0]=='create' for w in client.writes)==1


@pytest.mark.parametrize('auxiliary',[False,True])
def test_new_frozen_view_recovers_older_pending_queries_without_new_query_scene(tmp_path,auxiliary):
    config=changed(configured(tmp_path),clustering={'view_prototypes_enabled':True})
    with FaceDatabase(config.database) as db:
        old=[]
        for scene in ('1','2'):
            item=fixture(db,config,scene,[C]);old.append(item);rebuild_selected(db,[item],config)
        assert not db.connection.execute('SELECT 1 FROM members').fetchone()
        source=fixture(db,config,'3',[A]*3+[X]*3,range(3,6) if auxiliary else ())
        rebuild_selected(db,[source],config)
        owner=candidates(db,source['scan'],config)[0][1]
        assert all(r[0]=='accepted' and r[1]==owner and r[2]['reference_role']=='attachment'
            for item in old for r in candidates(db,item['scan'],config))
        assert db.connection.execute('SELECT COUNT(*) FROM members').fetchone()[0]==8


def test_frozen_view_combines_current_and_old_query_support(tmp_path):
    config=configured(tmp_path)
    with FaceDatabase(config.database) as db:
        old=fixture(db,config,'1',[C]);rebuild_selected(db,[old],config)
        source=fixture(db,config,'2',[A]*3+[X]*3)
        current=fixture(db,config,'3',[C]);rebuild_selected(db,[source,current],config)
        owner=candidates(db,source['scan'],config)[0][1]
        assert all(r[0]=='accepted' and r[1]==owner and r[2]['reason']=='pooled_view_identity'
            for item in (old,current) for r in candidates(db,item['scan'],config))


def test_reference_training_cues_do_not_count_as_view_query_votes(tmp_path):
    config=configured(tmp_path)
    with FaceDatabase(config.database) as db:
        old=fixture(db,config,'1',[C]);rebuild_selected(db,[old],config)
        source=fixture(db,config,'2',[A]*3+[X]*3);rebuild_selected(db,[source],config)
        assert all(r[0]=='withheld' for r in candidates(db,old['scan'],config))


def test_distinct_faces_in_one_cue_cannot_claim_different_views_of_one_owner(tmp_path):
    config=configured(tmp_path)
    with FaceDatabase(config.database) as db:
        old=[]
        for scene in ('1','2'):
            item=fixture(db,config,scene,[angle(-20),angle(65)])
            db.connection.execute('UPDATE observations SET cue=0,seconds=0 WHERE scan=?',(item['scan'],));db.connection.commit()
            old.append(item);rebuild_selected(db,[item],config)
        source=fixture(db,config,'3',[A]*3+[X]*3);rebuild_selected(db,[source],config)
        assert all(r[0]=='withheld' for item in old for r in candidates(db,item['scan'],config))


def test_final_auxiliary_reassignment_revokes_early_pool_source(tmp_path):
    config=changed(configured(tmp_path),clustering={'view_prototypes_enabled':True})
    x=np.array([1.,0.,0.,0.])
    a=np.array([.7,np.sqrt(1-.7**2),0.,0.])
    c=np.array([.914,0.,np.sqrt(1-.914**2),0.])
    rival=np.array([.81,-.3,-np.sqrt(1-.81**2-.3**2),0.])
    with FaceDatabase(config.database) as db:
        old=fixture(db,config,'1',[c]);rebuild_selected(db,[old],config)
        known=fixture(db,config,'2',[rival]*3);rebuild_selected(db,[known],config)
        source=fixture(db,config,'3',[a]*3+[x]*3,range(3,6))
        current=fixture(db,config,'4',[c]);rebuild_selected(db,[source,current],config)
        assert all(r[0]=='withheld' for r in candidates(db,old['scan'],config))
        result=candidates(db,current['scan'],config)[0]
        assert result[0]=='withheld' and result[2]['reason']=='provisional_pool_source_invalidated'


def test_unrelated_malformed_cache_row_does_not_break_confirmation_or_pool_reads(tmp_path):
    from stash_ai.recognition.historical_references import capture
    from stash_ai.recognition.clustering import matching_policy
    config=changed(configured(tmp_path),clustering={'view_prototypes_enabled':True})
    with FaceDatabase(config.database) as db:
        source=fixture(db,config,'1',[A]*3+[X]*3,range(3,6));rebuild_selected(db,[source],config)
        before=context(db.connection,[],config,'scope','ns')
        db.connection.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?)',
            (str(uuid.uuid4()),source['scan'],'damaged-policy',None,'withheld','not-json'));db.connection.commit()
        assert context(db.connection,[],config,'scope','ns')==before
        assert any(v.authority=='verified_view' for v in capture(db.connection,[],config,'scope','ns','check').views)
