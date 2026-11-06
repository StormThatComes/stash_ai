from contextlib import closing
import json
import sqlite3

from stash_ai.recognition import rebuild
from stash_ai.recognition.checkpoint_delta import DeltaCheckpoints
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.replay_input_cache import InputCache
from stash_ai.recognition.replay_memory import memory_replay
from tests.test_recognition_clustering import setup
from tests.test_recognition_replay_checkpoints import configured,selection,interrupt,count_replay,grouping,invalidate_warm


def test_incremental_prefix_writes_only_changed_rows_and_resumes_identically(tmp_path,monkeypatch):
    config=configured(setup(tmp_path),rebuild_checkpoint_mode='incremental')
    with FaceDatabase(config.database) as state:
        selected=selection(state,config,count=4)
        interrupt(monkeypatch,state,selected,config,after=3)
        directory=config.database.with_name(config.database.name+'.rebuild-delta')
        manifest=json.loads((directory/'current.json').read_text())
        with closing(sqlite3.connect(directory/manifest['generation'])) as delta:
            header=json.loads(delta.execute('SELECT value FROM metadata').fetchone()[0])
            assert header['cursor']==3 and header['parent'] is not None
            assert delta.execute('SELECT COUNT(*) FROM candidates').fetchone()[0]==3
        calls=count_replay(monkeypatch)
        result=rebuild.rebuild_selected(state,selected,config)
        assert calls==[selected[-1]['scan']]
        expected=grouping(state.connection)
        invalidate_warm(state)
        config.data['execution']['rebuild_checkpoint_scenes']=0
        assert rebuild.rebuild_selected(state,selected,config)==result
        assert grouping(state.connection)==expected


def test_raw_cache_survives_decision_churn_and_refreshes_bindings(tmp_path):
    config=configured(setup(tmp_path))
    with FaceDatabase(config.database) as state:
        selected=selection(state,config,count=1)
        with memory_replay(config) as (replay,_):
            rebuild._copy_selection(state,replay,selected,'scope','ns',bindings=False)
            cache=InputCache(state,selected,config,'scope','ns');cache.save(replay)
        state.connection.execute("INSERT INTO candidates VALUES ('test',?,'policy',NULL,'withheld','{}')",(selected[0]['scan'],))
        state.connection.commit()
        newer=InputCache(state,selected,config,'scope','ns')
        assert newer.key==cache.key
        with memory_replay(config) as (replay,_):
            assert newer.restore(replay)
            rebuild._copy_bindings(state,replay,selected,'scope','ns')
            assert replay.execute('SELECT COUNT(*) FROM observations').fetchone()[0]==3
        state.connection.execute("UPDATE observations SET quality=json_set(quality,'$.changed',1)")
        state.connection.commit()
        assert InputCache(state,selected,config,'scope','ns').key!=cache.key


def test_failed_delta_publication_keeps_previous_committed_prefix(tmp_path,monkeypatch):
    from stash_ai.recognition import replay_checkpoint as base
    config=configured(setup(tmp_path),rebuild_checkpoint_mode='incremental')
    with FaceDatabase(config.database) as state:
        selected=selection(state,config,count=4)
        original=base._exclusive_json;published=[]
        def fail_second(path,value):
            if path.name.startswith('.manifest-'):
                published.append(path)
                if len(published)==2:raise OSError('interrupted delta publication')
            return original(path,value)
        with monkeypatch.context() as patch:
            patch.setattr(base,'_exclusive_json',fail_second)
            interrupt(patch,state,selected,config,after=3)
        with memory_replay(config) as (replay,_):
            rebuild._copy_selection(state,replay,selected,'scope','ns')
            checkpoints=DeltaCheckpoints(state,selected,config,'scope','ns')
            assert checkpoints.restore(replay)==(1,2)


def test_corrupt_parent_is_rejected_before_any_grouping_is_imported(tmp_path,monkeypatch):
    config=configured(setup(tmp_path),rebuild_checkpoint_mode='incremental')
    with FaceDatabase(config.database) as state:
        selected=selection(state,config,count=3)
        interrupt(monkeypatch,state,selected,config,after=2)
        checkpoints=DeltaCheckpoints(state,selected,config,'scope','ns')
        first=checkpoints._walk()[0][0]
        with sqlite3.connect(first) as db:db.execute("UPDATE candidates SET payload='{}'")
        with memory_replay(config) as (replay,_):
            rebuild._copy_selection(state,replay,selected,'scope','ns')
            assert checkpoints.restore(replay)==(0,1)
            assert replay.execute('SELECT COUNT(*) FROM candidates').fetchone()[0]==0
