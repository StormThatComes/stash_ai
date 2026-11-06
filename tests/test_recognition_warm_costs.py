import hashlib
from dataclasses import replace
import pytest


def test_hybrid_stamp_batches_scans_and_ignores_profiling(tmp_path):
    from tests.test_recognition_hybrid_merge import prepared
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.hybrid_merge import signature
    from stash_ai.recognition.config import plain
    config,_=prepared(tmp_path)
    with FaceDatabase(config.database) as state:
        db=state.connection;sample=list(db.execute('SELECT * FROM scans LIMIT 1').fetchone())
        for i in range(3,301):
            row=sample.copy();row[0]=f's{i}';row[2]=str(i)
            db.execute('INSERT INTO scans VALUES (?,?,?,?,?,?,?,?,?,?,?)',row)
        db.commit();selected=[dict(scan=sample[0],scene=sample[2])]
        before=signature(db,selected,config);db.commit()
        queries=[];db.set_trace_callback(queries.append)
        assert signature(db,selected,config)==before
        db.set_trace_callback(None)
        assert sum(s.startswith('SELECT') for s in queries)<70
        data=plain(config.data);data['execution']['profile_rebuild']=not data['execution']['profile_rebuild']
        assert signature(db,selected,replace(config,data=data))==before
        data['execution']['rebuild_memory_bytes']//=2
        assert signature(db,selected,replace(config,data=data))!=before


def test_committed_trigger_audit_reused_but_ddl_and_rollback_checked(tmp_path):
    from tests.test_recognition_clustering import setup
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.dependency_schema import validate_triggers
    config=setup(tmp_path)
    with FaceDatabase(config.database) as state:
        db=state.connection;validate_triggers(db,required=True)
        statements=[];db.set_trace_callback(statements.append)
        for _ in range(10):validate_triggers(db,required=True)
        db.set_trace_callback(None)
        assert not any('SELECT type,name,sql' in q for q in statements)
        for name in ('recognition_version_observations_update','recognition_version_members_update'):
            db.execute('BEGIN');db.execute('DROP TRIGGER '+name)
            with pytest.raises(ValueError,match='Incomplete'):validate_triggers(db,required=True)
            db.rollback()
        validate_triggers(db,required=True)


def test_publication_validator_uses_hint_inside_its_own_snapshot(tmp_path):
    from tests.test_recognition_merge_report import seeded
    from tests.test_recognition_scoped_certificates import capture
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.publication_queue import validate
    config,_=seeded(tmp_path)
    with FaceDatabase(config.database) as state:
        db=state.connection;ns=db.execute('SELECT namespace FROM people LIMIT 1').fetchone()[0]
        reader,entry=capture(db,config,ns)
        statements=[];db.set_trace_callback(statements.append)
        for _ in range(10):validate(db,'scope',ns,entry,config,snapshot_reader=reader)
        db.set_trace_callback(None)
        assert not any('SELECT m.person FROM members' in q for q in statements)


def test_cached_scenes_do_not_prefetch_again_on_append(tmp_path,monkeypatch):
    from tests.test_recognition_rebuild import library
    from tests.test_recognition_run import run
    from stash_ai.recognition.replay_memory import Lookahead
    vectors={str(i):[[1.,0,0,0]]*3 for i in range(1,5)}
    config,_,services,_=library(tmp_path,monkeypatch,vectors)
    from stash_ai.recognition.config import plain
    config=replace(config,data=plain(config.data))
    config.data['clustering']['anchor_view_matching']=True
    config.data['execution']['rebuild_prefetch_scenes']=True
    assert run(config,services,'1','2','3')==0
    calls=[];original=Lookahead._prepare
    def measured(self,scan):calls.append(scan);return original(self,scan)
    monkeypatch.setattr(Lookahead,'_prepare',measured)
    assert run(config,services,'1','2','3','4')==0
    assert len(calls)==1


def test_sprite_read_hashes_pixels_once(tmp_path,monkeypatch):
    from tests.test_recognition_clustering import setup
    from tests.helpers.recognition import MediaClient
    from stash_ai.recognition.media import SpriteInput
    media=MediaClient();pixels=media.data['http://stash/sprite/1.jpg']
    calls=[];original=hashlib.sha256
    def measured(value=b'',*a,**kw):
        if value==pixels:calls.append(1)
        return original(value,*a,**kw)
    monkeypatch.setattr(hashlib,'sha256',measured)
    with SpriteInput(media,media.scene,setup(tmp_path)) as source:
        assert source.hashes['http://stash/sprite/1.jpg']==original(pixels).hexdigest()
    assert len(calls)==1
