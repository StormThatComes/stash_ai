"""Previous query authority survives full and mixed selection replays."""
import json
import pytest

from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.rebuild import rebuild_selected
from stash_ai.recognition.corroborated_views import defer_new_groups
from tests.test_recognition_corroborated_views import seed
from tests.test_recognition_deterministic_resolution import configured
from tests.test_recognition_frozen_partition import fixture
from tests.test_recognition_query_support import appearance
from tests.test_recognition_scene_context import candidates


@pytest.mark.parametrize('selection',['all','mixed','query_only'])
def test_known_corroborated_queries_keep_owner_and_attachment_authority(tmp_path,selection,monkeypatch):
    config=configured(tmp_path)
    with FaceDatabase(config.database) as db:
        source,queries=seed(db,config)
        last=fixture(db,config,'4',[appearance(88)]*3);rebuild_selected(db,[last],config)
        owner=candidates(db,source['scan'],config)[0][1]
        selected=([source,*queries,last] if selection=='all' else
                  [source,queries[0],last] if selection=='mixed' else [last])
        rebuild_selected(db,selected,config)
        assert all(s=='accepted' and p==owner and d['reference_role']=='attachment'
                   for s,p,d in candidates(db,last['scan'],config))
        assert db.connection.execute('SELECT COUNT(DISTINCT person) FROM members').fetchone()[0]==1
        before=[tuple(r) for r in db.connection.execute('SELECT * FROM members ORDER BY observation')]
        def forbidden(*a,**k):pytest.fail('completed stable corroboration should reuse its cache')
        monkeypatch.setattr('stash_ai.recognition.rebuild.cluster_scene',forbidden)
        rebuild_selected(db,selected,config)
        assert before==[tuple(r) for r in db.connection.execute('SELECT * FROM members ORDER BY observation')]


def test_new_full_run_generation_waits_until_next_replay_then_becomes_cacheable(tmp_path,monkeypatch):
    config=configured(tmp_path)
    with FaceDatabase(config.database) as db:
        source=fixture(db,config,'1',[appearance(0)]*3+[appearance(45,15)]*3+[appearance(45,-15)]*3,range(3,9))
        queries=[fixture(db,config,str(i),[appearance(65)]*3) for i in (2,3)]
        last=fixture(db,config,'4',[appearance(88)]*3);selected=[source,*queries,last]
        rebuild_selected(db,selected,config)
        assert all(s=='withheld' for s,_,_ in candidates(db,last['scan'],config))
        state=json.loads(db.connection.execute("SELECT value FROM metadata WHERE key LIKE 'identity-rebuild:%'").fetchone()[0])
        assert state['corroboration_pending'] and state['corroborated_plan']
        rebuild_selected(db,selected,config)
        owner=candidates(db,source['scan'],config)[0][1]
        assert all(s=='accepted' and p==owner and d['reference_role']=='attachment' for s,p,d in candidates(db,last['scan'],config))
        def forbidden(*a,**k):pytest.fail('the completed second generation should now be cached')
        monkeypatch.setattr('stash_ai.recognition.rebuild.cluster_scene',forbidden)
        rebuild_selected(db,selected,config)


@pytest.mark.parametrize('established',[False,True])
def test_prior_hint_only_defers_new_hypotheses_and_never_assigns_an_owner(tmp_path,established):
    config=configured(tmp_path)
    with FaceDatabase(config.database) as db:
        item=fixture(db,config,'1',[appearance(88)]*3);rebuild_selected(db,[item],config)
        ids={r['id'] for r in db.connection.execute('SELECT id FROM observations')}
        hint=dict(id='hint',geometry='[]',dimensions=4,embedding=appearance(65).astype('<f4').tobytes())
        used=defer_new_groups(db.connection,config,[hint],ids if established else set())
        assert bool(used)==(not established)
        rows=candidates(db,item['scan'],config)
        if established:assert all(s=='accepted' for s,_,_ in rows)
        else:assert all(s=='withheld' and p is None and d['reason']=='prior_appearance_pending' for s,p,d in rows)


def test_stale_prior_hint_cannot_authorize_assignment_after_parent_is_rejected(tmp_path):
    config=configured(tmp_path)
    with FaceDatabase(config.database) as db:
        source,queries=seed(db,config)
        owner=candidates(db,source['scan'],config)[0][1]
        db.connection.execute("UPDATE observations SET rejected='face_blurred' WHERE scan=? AND cue=3",(source['scan'],));db.connection.commit()
        last=fixture(db,config,'4',[appearance(88)]*3)
        rebuild_selected(db,[source,*queries,last],config)
        assert all(s!='accepted' or p!=owner for s,p,_ in candidates(db,last['scan'],config))


def test_prior_attachment_queries_cannot_supply_missing_birth_votes(tmp_path):
    config=configured(tmp_path)
    from tests.test_recognition_rebuild import changed
    config=changed(config,recognition={'min_identity_support':3})
    with FaceDatabase(config.database) as db:
        item=fixture(db,config,'1',[appearance(0)]*3);rebuild_selected(db,[item],config)
        db.connection.execute("UPDATE candidates SET payload=json_set(payload,'$.reason','historical_view_identity',"
            "'$.reference_role','attachment') WHERE json_extract(payload,'$.observation') IN (SELECT id FROM observations WHERE cue IN (1,2))")
        db.connection.commit();rebuild_selected(db,[item],config)
        assert not db.connection.execute('SELECT 1 FROM members').fetchone()


def test_selected_corroboration_rollback_keeps_durable_assignments(tmp_path,monkeypatch):
    config=configured(tmp_path)
    with FaceDatabase(config.database) as db:
        source,queries=seed(db,config)
        last=fixture(db,config,'4',[appearance(88)]*3);rebuild_selected(db,[last],config)
        before={t:[tuple(r) for r in db.connection.execute('SELECT * FROM '+t)] for t in ('members','people','candidates')}
        def failed(*a,**k):raise RuntimeError('before final corroboration commit')
        monkeypatch.setattr('stash_ai.recognition.rebuild._apply',failed)
        with pytest.raises(RuntimeError,match='before final corroboration commit'):
            rebuild_selected(db,[source,*queries,last],config)
        assert before=={t:[tuple(r) for r in db.connection.execute('SELECT * FROM '+t)] for t in before}


@pytest.mark.parametrize('value',['null','[]','not-json'])
def test_corrupt_derived_plan_cannot_block_fresh_revalidation(tmp_path,value):
    config=configured(tmp_path)
    with FaceDatabase(config.database) as db:
        source,queries=seed(db,config)
        last=fixture(db,config,'4',[appearance(88)]*3);rebuild_selected(db,[last],config)
        db.connection.execute("UPDATE metadata SET value=? WHERE key LIKE 'identity-rebuild:%'",(value,));db.connection.commit()
        rebuild_selected(db,[source,*queries,last],config)
        assert all(s=='accepted' for s,_,_ in candidates(db,last['scan'],config))
        assert db.connection.execute('SELECT COUNT(DISTINCT person) FROM members').fetchone()[0]==1


def test_unrelated_selection_preserves_plan_for_later_selected_sources(tmp_path):
    import numpy as np
    config=configured(tmp_path)
    with FaceDatabase(config.database) as db:
        source,queries=seed(db,config)
        other=fixture(db,config,'4',[np.array([0.,0.,0.,1.])]*3);rebuild_selected(db,[other],config)
        last=fixture(db,config,'5',[appearance(88)]*3);rebuild_selected(db,[source,last],config)
        owner=candidates(db,source['scan'],config)[0][1]
        assert all(s=='accepted' and p==owner and d['reference_role']=='attachment' for s,p,d in candidates(db,last['scan'],config))


def test_obsolete_external_hint_is_cleared_after_fresh_validation_fails(tmp_path):
    config=configured(tmp_path)
    with FaceDatabase(config.database) as db:
        source,_=seed(db,config)
        db.connection.execute("UPDATE observations SET rejected='face_blurred' WHERE scan=? AND cue=3",(source['scan'],));db.connection.commit()
        last=fixture(db,config,'4',[appearance(88)]*3);rebuild_selected(db,[source,last],config)
        state=json.loads(db.connection.execute("SELECT value FROM metadata WHERE key LIKE 'identity-rebuild:%'").fetchone()[0])
        assert state['corroborated_plan']==[]
