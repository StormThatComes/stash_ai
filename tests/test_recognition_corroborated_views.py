"""Only independently rooted, multi-scene appearances extend the query bank."""
import json
import numpy as np
import pytest

from stash_ai.recognition.clustering import matching_policy
from stash_ai.recognition.config import load
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.historical_references import capture,context
from stash_ai.recognition.rebuild import rebuild_selected
from stash_ai.recognition.scanning import scan_identity
from tests.test_recognition_deterministic_resolution import configured
from tests.test_recognition_frozen_partition import fixture
from tests.test_recognition_query_support import appearance
from tests.test_recognition_rebuild import changed
from tests.test_recognition_scene_context import candidates


def bank(db,config,selected=()):
    return capture(db.connection,selected,config,'scope','ns',context(db.connection,selected,config,'scope','ns'))


def seed(db,config,*,parents=2,scenes=2,votes=3):
    vectors=[appearance(0)]*3+[appearance(45,15)]*3
    if parents==2:vectors += [appearance(45,-15)]*3
    source=fixture(db,config,'1',vectors,range(3,len(vectors)));rebuild_selected(db,[source],config)
    queries=[]
    for i in range(scenes):
        item=fixture(db,config,str(i+2),[appearance(65)]*votes);rebuild_selected(db,[item],config);queries.append(item)
    return source,queries


@pytest.mark.parametrize('enabled',[False,True])
def test_corroborated_appearance_recognizes_next_scene_without_new_identity(tmp_path,enabled):
    config=changed(configured(tmp_path),clustering={'corroborated_view_enrichment':enabled})
    with FaceDatabase(config.database) as db:
        source,queries=seed(db,config)
        owner=candidates(db,source['scan'],config)[0][1]
        history=bank(db,config)
        promoted=[v for v in history.views if v.authority=='corroborated_view']
        assert bool(promoted)==enabled
        if enabled:
            proof=promoted[0].promotion
            assert proof['generation']==1 and proof['query_support']==6 and proof['scene_count']==2
            assert len(proof['parents'])==2
            a,b=proof['parents']
            assert not set(a['support_ids']).intersection(b['support_ids'])
            assert set(proof['source_reference_ids']).issubset(promoted[0].source_ids)
        new=fixture(db,config,'4',[appearance(88)]*3);rebuild_selected(db,[new],config)
        assert db.connection.execute('SELECT COUNT(*) FROM people').fetchone()[0]==(1 if enabled else 2)
        if enabled:
            rows=candidates(db,new['scan'],config)
            assert all(s=='accepted' and p==owner and d['reference_role']=='attachment' for s,p,d in rows)
            evidence=rows[0][2]['historical_reference']['view_evidence'][0]
            assert evidence['authority']=='corroborated_view' and evidence['promotion']['generation']==1


@pytest.mark.parametrize('options',[dict(parents=1),dict(scenes=1,votes=6),dict(votes=1)])
def test_missing_independent_parents_scenes_or_query_votes_cannot_promote(tmp_path,options):
    config=configured(tmp_path)
    with FaceDatabase(config.database) as db:
        seed(db,config,**options)
        assert not any(v.authority=='corroborated_view' for v in bank(db,config).views)


def test_one_root_cannot_be_counted_twice_through_overlapping_prototypes(tmp_path):
    from stash_ai.recognition.corroborated_views import _independent
    from stash_ai.recognition.global_views import IdentityView
    from stash_ai.recognition.clustering import Group
    sample=appearance(45)
    def view(medoid,ids):return IdentityView(Group('p',(sample,),True),'other',medoid,ids,3,(sample,))
    assert _independent([view('a',('a','b','c')),view('b',('b','c','d'))]) is None


@pytest.mark.parametrize('invalidate',['parent','query','supersede','reject'])
def test_lost_durable_support_revokes_next_capture_and_invalidates_context(tmp_path,invalidate):
    config=configured(tmp_path)
    with FaceDatabase(config.database) as db:
        source,queries=seed(db,config)
        history=bank(db,config);view=next(v for v in history.views if v.authority=='corroborated_view')
        before=context(db.connection,[],config,'scope','ns')
        if invalidate=='parent':
            db.connection.execute('DELETE FROM members WHERE observation=?',(view.promotion['parents'][0]['support_ids'][0],))
        elif invalidate=='query':
            db.connection.execute('DELETE FROM members WHERE observation IN (SELECT id FROM observations WHERE scan=?)',
                                  (queries[0]['scan'],))
        elif invalidate=='reject':
            db.connection.execute("UPDATE observations SET rejected='face_blurred' WHERE scan=?",(queries[0]['scan'],))
        else:
            from tests.test_recognition_reference_registry import pending
            pending(db,queries[0])
        db.connection.commit()
        assert context(db.connection,[],config,'scope','ns')!=before
        assert not any(v.authority=='corroborated_view' for v in bank(db,config).views)


def test_selected_queries_cannot_train_a_view_in_their_own_replay(tmp_path):
    config=configured(tmp_path)
    with FaceDatabase(config.database) as db:
        _,queries=seed(db,config)
        assert not any(v.authority=='corroborated_view' for v in bank(db,config,queries).views)


def test_promoted_views_cannot_become_promotion_parents(tmp_path):
    config=configured(tmp_path)
    with FaceDatabase(config.database) as db:
        source,_=seed(db,config)
        for scene in ('4','5'):
            item=fixture(db,config,scene,[appearance(88)]*3);rebuild_selected(db,[item],config)
            assert all(s=='accepted' for s,_,_ in candidates(db,item['scan'],config))
        promoted=[v for v in bank(db,config).views if v.authority=='corroborated_view']
        assert promoted and all(np.dot(v.group.samples[0],appearance(65))>.999 for v in promoted)
        assert all(p['authority'] in ('strict','verified_view') for v in promoted for p in v.promotion['parents'])


def test_corroboration_only_changes_matching_and_capture_is_read_only(tmp_path):
    config=configured(tmp_path);disabled=changed(config,clustering={'corroborated_view_enrichment':False})
    assert matching_policy(config)!=matching_policy(disabled)
    assert scan_identity(config,'ns')==scan_identity(disabled,'ns')
    with FaceDatabase(config.database) as db:
        seed(db,config)
        before=list(db.connection.iterdump())
        assert any(v.authority=='corroborated_view' for v in bank(db,config).views)
        assert before==list(db.connection.iterdump())


def test_corroboration_option_is_boolean(tmp_path):
    path=tmp_path/'bad.yml';path.write_text('version: 1\nclustering: {corroborated_view_enrichment: yesplease}\n')
    with pytest.raises(ValueError,match='corroborated_view_enrichment must be boolean'):load(path)


def test_promotion_memory_overflow_is_explicit(tmp_path):
    from stash_ai.recognition.corroborated_views import enrich
    from stash_ai.recognition.reference_usage import ReferenceUsage
    config=configured(tmp_path)
    with FaceDatabase(config.database) as db:
        seed(db,config);history=bank(db,config)
        with pytest.raises(ValueError,match='Corroborated view evidence'):
            enrich(db.connection,config,'scope','ns',[],history,ReferenceUsage(1))
