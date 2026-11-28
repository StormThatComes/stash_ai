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


def test_dirty_recovery_ranges_track_replace_delete_and_rollback():
    import sqlite3
    from stash_ai.recognition.recovery_digest import snapshots
    db=sqlite3.connect(':memory:')
    try:
        db.executescript("CREATE TABLE candidates(id TEXT PRIMARY KEY,person,status);"
                         "CREATE TABLE identity_merges(source TEXT PRIMARY KEY,target);"
                         "INSERT INTO candidates VALUES ('aa1','a','held'),('aa2','b','held');")
        with snapshots(db) as snapshot:
            original=snapshot()
            db.execute("SAVEPOINT attempt")
            db.execute("INSERT OR REPLACE INTO candidates VALUES ('aa1','c','held')")
            db.execute("DELETE FROM candidates WHERE id='aa2'")
            db.execute("ROLLBACK TO attempt")
            assert snapshot()==original
            db.execute("INSERT OR REPLACE INTO candidates VALUES ('aa1','c','held')")
            assert snapshot()!=original
            db.execute("INSERT OR REPLACE INTO candidates VALUES ('aa1','a','held')")
            assert snapshot()==original
    finally:db.close()


@pytest.mark.parametrize('phase', ['selected', 'ready'])
def test_large_phase_catalogue_is_checksummed_in_pages(phase):
    import sqlite3
    from contextlib import closing
    from stash_ai.recognition import replay_checkpoint as checkpoint

    with closing(sqlite3.connect(':memory:')) as db:
        db.executescript(checkpoint.SCHEMA)
        large = [str(i).zfill(40) for i in range(105_000)]
        payload = {'deferred': large} if phase == 'selected' else {'mapping': dict.fromkeys(large,'owner')}
        value = {'phase': phase, 'phase_payload': payload, 'selected': [{'scan':'one'}]}
        checkpoint._write_header(db, value)
        assert db.execute('SELECT max(length(value)) FROM metadata').fetchone()[0] < checkpoint._MAX_META
        assert checkpoint._header(db) == value
        db.execute("UPDATE metadata SET value=x'00' WHERE key='part-000000000001'")
        with pytest.raises(ValueError, match='digest'):
            checkpoint._header(db)


def test_face_label_conflict_preserves_both_historical_ids():
    from stash_ai.recognition.historical_face_labels import bind_votes

    def vote(cluster, person, count):
        return dict(cluster=cluster, person=person, total=count, performer=None, retired=False)
    rows = [vote('a','oldest',8), vote('b','oldest',7), vote('a','second',6)]
    for ordered in (rows, [rows[1], rows[0], rows[2]]):
        mapping = {}
        bind_votes(ordered, mapping, set(), set())
        assert mapping == {'a':'second','b':'oldest'}
    mapping = {}
    bind_votes([vote('a','oldest',2), vote('a','second',10)], mapping, set(), set())
    assert mapping == {'a':'oldest'}


def test_parallel_write_during_preparation_invalidates_publication(tmp_path):
    import sqlite3
    from contextlib import closing
    from types import SimpleNamespace
    from stash_ai.recognition.rebuild_transaction import atomic, SourceChanged

    path = tmp_path/'recognition.sqlite'
    with closing(sqlite3.connect(path)) as raw, closing(sqlite3.connect(path,timeout=.1)) as other:
        raw.execute('CREATE TABLE assignments(person)')
        database = SimpleNamespace(connection=raw)
        with pytest.raises(SourceChanged), atomic(database):
            assert list(database.connection.execute('SELECT * FROM assignments')) == []
            with other:
                other.execute("INSERT INTO assignments VALUES ('newer')")
            database.connection.execute("INSERT INTO assignments VALUES ('stale')")
        assert list(raw.execute('SELECT * FROM assignments')) == [('newer',)]
        with pytest.raises(RuntimeError), atomic(database):
            database.connection.execute("INSERT INTO assignments VALUES ('interrupted')")
            raise RuntimeError
        assert list(raw.execute('SELECT * FROM assignments')) == [('newer',)]
