"""One frozen rooted generation completes before any durable publication."""
from contextlib import closing
import json
import sqlite3

import pytest

from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.frozen_ownership import FrozenOwnership
from stash_ai.recognition.rebuild import rebuild_selected
from tests.test_recognition_corroborated_views import bank
from tests.test_recognition_deterministic_resolution import configured
from tests.test_recognition_frozen_partition import fixture
from tests.test_recognition_query_support import appearance
from tests.test_recognition_rebuild import changed
from tests.test_recognition_scene_context import candidates


def cold(db, config, *, parents=2, scenes=2, votes=3):
    vectors = [appearance(0)]*3+[appearance(45,15)]*3
    if parents==2:
        vectors += [appearance(45,-15)]*3
    source = fixture(db,config,'1',vectors,range(3,len(vectors)))
    queries = [fixture(db,config,str(i+2),[appearance(65)]*votes) for i in range(scenes)]
    last = fixture(db,config,'9',[appearance(88)]*3)
    return source,queries,last


def assert_complete(db,config,source,last):
    owner = candidates(db,source['scan'],config)[0][1]
    assert all(s=='accepted' and p==owner and d['reference_role']=='attachment'
               for s,p,d in candidates(db,last['scan'],config))
    return owner


def test_cold_run_freezes_once_and_never_replays_scenes_or_reinfers_faces(tmp_path,monkeypatch):
    from stash_ai.recognition import rebuild,selected_corroboration,models
    config = configured(tmp_path)
    with FaceDatabase(config.database) as db:
        source,queries,last = cold(db,config);selected = [source,*queries,last]
        calls,freezes = [],[]
        cluster,enrich = rebuild.cluster_scene,selected_corroboration.enrich
        def counted(*args,**kwargs):
            calls.append(args[1]);return cluster(*args,**kwargs)
        def frozen(*args,**kwargs):
            assert all(v.authority in ('strict','verified_view') for v in args[5].views)
            rows = list(kwargs['query_rows']())
            assert all('crop' not in r for r in rows)
            freezes.append({r['id'] for r in rows})
            return enrich(*args,**kwargs)
        def forbidden(*a,**k):pytest.fail('finalization must reuse retained embeddings')
        monkeypatch.setattr(rebuild,'cluster_scene',counted)
        monkeypatch.setattr(selected_corroboration,'enrich',frozen)
        monkeypatch.setattr(models,'load_backend',forbidden)
        rebuild_selected(db,selected,config)
        owner = assert_complete(db,config,source,last)
        assert calls == [r['scan'] for r in selected] and len(freezes)==1
        last_ids = {r[0] for r in db.connection.execute('SELECT id FROM observations WHERE scan=?',(last['scan'],))}
        assert not last_ids.intersection(freezes[0])
        state = json.loads(db.connection.execute("SELECT value FROM metadata WHERE key LIKE 'identity-rebuild:%'").fetchone()[0])
        assert not state['corroboration_pending']
        assert not last_ids.intersection(state['corroborated_plan'])
        monkeypatch.setattr(rebuild,'cluster_scene',forbidden)
        monkeypatch.setattr(selected_corroboration,'enrich',forbidden)
        rebuild_selected(db,selected,config)
        assert_complete(db,config,source,last)
        assert db.connection.execute('SELECT COUNT(DISTINCT person) FROM members').fetchone()[0]==1


@pytest.mark.parametrize('options',[dict(parents=1),dict(scenes=1),dict(votes=1),dict(disabled=True)])
def test_one_run_finalization_does_not_relax_promotion_requirements(tmp_path,options):
    config = configured(tmp_path)
    if options.get('disabled'):
        config = changed(config,clustering={'corroborated_view_enrichment':False});options={}
    with FaceDatabase(config.database) as db:
        source,queries,last = cold(db,config,**options)
        rebuild_selected(db,[source,*queries,last],config)
        owner = candidates(db,source['scan'],config)[0][1]
        assert all(s!='accepted' or p!=owner for s,p,_ in candidates(db,last['scan'],config))
        assert not any(v.authority=='corroborated_view' for v in bank(db,config).views)


@pytest.mark.parametrize('include_source',[False,True])
def test_current_and_foreign_query_scenes_complete_in_the_same_run(tmp_path,monkeypatch,include_source):
    config = configured(tmp_path)
    with FaceDatabase(config.database) as db:
        source,queries,last = cold(db,config)
        rebuild_selected(db,[source],config)
        rebuild_selected(db,[queries[0]],config)
        selected = ([source] if include_source else [])+[queries[1],last]
        rebuild_selected(db,selected,config)
        owner=assert_complete(db,config,source,last)
        assert all(s=='accepted' and p==owner for item in (source,*queries)
                   for s,p,_ in candidates(db,item['scan'],config))
        monkeypatch.setattr('stash_ai.recognition.rebuild.cluster_scene',lambda *a,**k:pytest.fail('mixed selection should be cached'))
        rebuild_selected(db,selected,config)


def test_one_prior_scene_can_defer_a_birth_but_cannot_authorize_promotion(tmp_path,monkeypatch):
    config=configured(tmp_path)
    with FaceDatabase(config.database) as db:
        source,queries,last=cold(db,config,scenes=1)
        rebuild_selected(db,[source],config)
        rebuild_selected(db,queries,config)
        selected=[source,last]
        rebuild_selected(db,selected,config)
        assert all(s=='withheld' and p is None for s,p,_ in candidates(db,last['scan'],config))
        state=json.loads(db.connection.execute("SELECT value FROM metadata WHERE key LIKE 'identity-rebuild:%'").fetchone()[0])
        assert not state['corroborated_plan'] and not state['corroboration_pending']
        monkeypatch.setattr('stash_ai.recognition.rebuild.cluster_scene',lambda *a,**k:pytest.fail('valid negative evidence should remain cached'))
        rebuild_selected(db,selected,config)


@pytest.mark.parametrize('checkpoint',[False,True])
def test_failure_after_final_matching_commits_no_generation_and_can_resume(tmp_path,monkeypatch,checkpoint):
    from stash_ai.recognition import rebuild,selected_corroboration
    config = changed(configured(tmp_path),execution={'rebuild_checkpoint_scenes':1 if checkpoint else 0})
    with FaceDatabase(config.database) as db:
        source,queries,last = cold(db,config);selected = [source,*queries,last]
        tables = ('members','people','candidates','identity_origins','publications','metadata')
        before = {t:[tuple(r) for r in db.connection.execute('SELECT * FROM '+t+' ORDER BY 1')] for t in tables}
        original = selected_corroboration._resolve_queries
        def failed(*args,**kwargs):
            original(*args,**kwargs)
            if not kwargs.get('prior_only',False):
                raise RuntimeError('after private finalization')
        with monkeypatch.context() as patch:
            patch.setattr(selected_corroboration,'_resolve_queries',failed)
            with pytest.raises(RuntimeError,match='private finalization'):
                rebuild_selected(db,selected,config)
        assert before == {t:[tuple(r) for r in db.connection.execute('SELECT * FROM '+t+' ORDER BY 1')] for t in tables}
        if checkpoint:
            monkeypatch.setattr(rebuild,'cluster_scene',lambda *a,**k:pytest.fail('resume the complete strict checkpoint'))
        rebuild_selected(db,selected,config)
        assert_complete(db,config,source,last)


def test_finalized_queries_do_not_supply_the_next_generation(tmp_path):
    config = configured(tmp_path)
    with FaceDatabase(config.database) as db:
        source,queries,last = cold(db,config)
        rebuild_selected(db,[source,*queries,last],config)
        owner = assert_complete(db,config,source,last)
        next_scene = fixture(db,config,'10',[appearance(112)]*3)
        rebuild_selected(db,[next_scene],config)
        assert all(s!='accepted' or p!=owner for s,p,_ in candidates(db,next_scene['scan'],config))
        assert all(p['authority'] in ('strict','verified_view') for v in bank(db,config).views
                   if v.authority=='corroborated_view' for p in v.promotion['parents'])


@pytest.mark.parametrize('selected_owner',[None,'another','owner'])
def test_selected_evidence_shadows_an_old_durable_owner(selected_owner):
    with closing(sqlite3.connect(':memory:')) as selected,closing(sqlite3.connect(':memory:')) as durable:
        for connection in (selected,durable):
            connection.row_factory=sqlite3.Row
            connection.executescript('CREATE TABLE observations(id TEXT);CREATE TABLE members(observation TEXT,person TEXT);')
            connection.execute("INSERT INTO observations VALUES ('face')")
        durable.execute("INSERT INTO members VALUES ('face','owner')")
        if selected_owner:
            selected.execute('INSERT INTO members VALUES (?,?)',('face',selected_owner))
        assert FrozenOwnership(selected,durable,{}).allows('owner',('face',)) == (selected_owner=='owner')


def test_foreign_source_ownership_uses_the_current_canonical_alias():
    with closing(sqlite3.connect(':memory:')) as selected,closing(sqlite3.connect(':memory:')) as durable:
        for connection in (selected,durable):
            connection.row_factory=sqlite3.Row
            connection.executescript('CREATE TABLE observations(id TEXT);CREATE TABLE members(observation TEXT,person TEXT);')
        durable.execute("INSERT INTO observations VALUES ('face')")
        durable.execute("INSERT INTO members VALUES ('face','old')")
        check = FrozenOwnership(selected,durable,{'old':'current'})
        assert check.allows('current',('face',)) and not check.allows('old',('face',))


@pytest.mark.parametrize('cue',[0,3])
def test_reference_binding_excludes_rejected_members_before_the_selected_repair(tmp_path,cue):
    from stash_ai.recognition.clustering import _groups
    config = configured(tmp_path)
    with FaceDatabase(config.database) as db:
        source,_,_ = cold(db,config)
        rebuild_selected(db,[source],config)
        rejected = db.connection.execute('SELECT id FROM observations WHERE scan=? AND cue=?',(source['scan'],cue)).fetchone()[0]
        db.connection.execute("UPDATE observations SET rejected='face_blurred' WHERE id=?",(rejected,))
        db.connection.commit()
        groups = _groups(db.connection,'scope','ns',config.data['clustering'])
        assert len(groups)==1 and rejected not in groups[0].sample_ids
        assert len(groups[0].samples)==(2 if cue==0 else 3)
