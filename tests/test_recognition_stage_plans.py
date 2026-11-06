import sqlite3
import pytest
from stash_ai.recognition.stage_checkpoints import Plans,scope,report
from stash_ai.recognition.database import FaceDatabase
from tests.test_recognition_body_identity import configuration,scene


def test_stage_plan_survives_main_rollback_and_raw_edits_invalidate(tmp_path):
    config=configuration(tmp_path);config.data['execution']['rebuild_checkpoint_scenes']=1
    calls=[]
    with FaceDatabase(config.database) as state:
        item,ids=scene(state.connection,config);state.connection.commit()
        def build():calls.append(1);return dict(decisions=[],proof='all original witnesses')
        for attempt in range(2):
            state.connection.execute('BEGIN')
            with scope(state,config):assert report(state.connection,[item],config,'body-global',build)['decisions']==[]
            state.connection.rollback()
        assert len(calls)==1
        state.connection.execute("UPDATE body_observations SET quality=json_set(quality,'$.new_evidence',1) WHERE id=?",(ids[0],))
        state.connection.commit()
        with scope(state,config):report(state.connection,[item],config,'body-global',build)
        assert len(calls)==2


def test_stage_corruption_is_miss_and_foreign_file_is_preserved(tmp_path):
    store=Plans(tmp_path/'faces',1048576)
    try:
        store.put('body-scene','input',{'one':'decision'})
        store.db.execute("UPDATE plans SET payload='{}'");store.db.commit()
        assert store.get('body-scene','input') is None
    finally:store.close()
    path=tmp_path/'foreign.stage-plans.sqlite3'
    with sqlite3.connect(path) as db:db.execute('CREATE TABLE valuable(value)')
    original=path.read_bytes()
    with pytest.raises(ValueError,match='foreign'):Plans(tmp_path/'foreign',1048576).open()
    assert path.read_bytes()==original
