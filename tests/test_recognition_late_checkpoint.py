import pytest
from stash_ai.recognition import rebuild,historical_references
from stash_ai.recognition.database import FaceDatabase
from tests.test_recognition_replay_checkpoints import configured,selection,grouping,count_replay
from tests.test_recognition_clustering import setup
from stash_ai.recognition.deferred_retries import RetryQueue


def test_resume_selected_partition_after_historical_failure(tmp_path,monkeypatch):
    config=configured(setup(tmp_path))
    with FaceDatabase(config.database) as state:
        selected=selection(state,config,count=3)
        original=grouping(state.connection)
        with monkeypatch.context() as patch:
            def fail(*args,**kwargs):
                raise RuntimeError('historical stage interrupted')
            patch.setattr(historical_references,'capture',fail)
            with pytest.raises(RuntimeError,match='historical stage'):
                rebuild.rebuild_selected(state,selected,config)
        assert grouping(state.connection)==original
        calls=count_replay(monkeypatch)
        result=rebuild.rebuild_selected(state,selected,config)
        assert result and calls==[]
        assert state.last_rebuild_profile['counts']['checkpoint_resumed_scenes']==3


def test_negative_retry_is_reused_only_for_identical_generation(tmp_path):
    config=setup(tmp_path)
    with FaceDatabase(config.database) as state:
        row=dict(candidate='c',id='o',geometry='[]',dimensions=1,embedding=b'1234')
        first=RetryQueue(state.connection,(),'p')
        assert list(first.pending([row]))==[row]
        first.remember(row)
        assert list(RetryQueue(state.connection,(),'p').pending([row]))==[]
        assert list(RetryQueue(state.connection,(),'new-policy').pending([row]))==[row]
        assert list(first.pending([dict(row,embedding=b'5678')]))


def test_ready_checkpoint_skips_late_analysis_after_commit_interruption(tmp_path,monkeypatch):
    config=configured(setup(tmp_path))
    with FaceDatabase(config.database) as state:
        selected=selection(state,config,count=3)
        original=grouping(state.connection)
        with monkeypatch.context() as patch:
            patch.setattr(rebuild,'_apply',lambda *a,**k: (_ for _ in ()).throw(RuntimeError('commit interrupted')))
            with pytest.raises(RuntimeError,match='commit interrupted'):
                rebuild.rebuild_selected(state,selected,config)
        assert grouping(state.connection)==original
        with monkeypatch.context() as patch:
            patch.setattr(historical_references,'capture',lambda *a,**k: pytest.fail('late analysis repeated'))
            assert rebuild.rebuild_selected(state,selected,config)


def test_convergence_detects_cycles_instead_of_repeating_forever():
    import sqlite3
    from stash_ai.recognition.recovery_convergence import converge
    db=sqlite3.connect(':memory:')
    try:
        db.executescript("CREATE TABLE candidates(id,person,status); CREATE TABLE identity_merges(source,target);"
                         "INSERT INTO candidates VALUES ('c','a','withheld');")
        def cycle():
            db.execute("UPDATE candidates SET person=CASE person WHEN 'a' THEN 'b' ELSE 'a' END")
        with pytest.raises(ValueError,match='cycle'):
            converge(db,cycle)
    finally:
        db.close()
