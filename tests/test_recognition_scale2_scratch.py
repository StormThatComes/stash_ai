from pathlib import Path
import pytest
from stash_ai.recognition import scratch_space
from stash_ai.recognition.resource_errors import ResourceLimit
from tests.test_recognition_compute import configured


def test_files_and_databases_share_capacity_and_cleanup(tmp_path):
    config=configured(rebuild_scratch_directory=str(tmp_path),rebuild_scratch_bytes=32*1024**2)
    with scratch_space.scope(config) as budget:
        directory=Path(budget.directory)
        db=scratch_space.connect()
        db.execute('CREATE TABLE values_(value BLOB)')
        db.execute('INSERT INTO values_ VALUES (?)',(b'v'*100000,));db.commit()
        assert budget.used>0
        with scratch_space.file() as stream:
            stream.write(b'x'*1024**2)
            before=budget.used
            with pytest.raises(ResourceLimit,match='scratch allocations'):
                stream.write(b'x'*(32*1024**2))
            assert budget.used==before and stream.tell()==1024**2
        path=db.owned_path
        db.close()
        assert not Path(path).exists()
        assert budget.used==0
    assert not directory.exists()


def test_capacity_failure_leaves_transaction_rollback_available(tmp_path):
    config=configured(rebuild_scratch_directory=str(tmp_path),rebuild_scratch_bytes=16*1024**2)
    with scratch_space.scope(config):
        db=scratch_space.connect()
        try:
            db.execute('CREATE TABLE values_(value BLOB)');db.commit()
            db.execute('INSERT INTO values_ VALUES (?)',(b'first',))
            with pytest.raises(ResourceLimit):db.execute('INSERT INTO values_ VALUES (?)',(b'x'*(16*1024**2),))
            db.rollback()
            assert db.execute('SELECT COUNT(*) FROM values_').fetchone()[0]==0
        finally:db.close()


def test_private_database_closes_retained_cursors_before_unlink(tmp_path):
    config=configured(rebuild_scratch_directory=str(tmp_path),rebuild_scratch_bytes=32*1024**2)
    with scratch_space.scope(config):
        db=scratch_space.connect();path=Path(db.owned_path)
        db.execute('CREATE TABLE values_(value INTEGER)')
        db.executemany('INSERT INTO values_ VALUES (?)',[(1,),(2,)])
        cursor=db.execute('SELECT * FROM values_');assert cursor.fetchone()==(1,)
        db.close()
        assert not path.exists()


def test_vector_growth_checks_free_space_per_block_not_per_witness(tmp_path,monkeypatch):
    from types import SimpleNamespace
    calls=[]
    monkeypatch.setattr(scratch_space.shutil,'disk_usage',lambda path:calls.append(path) or SimpleNamespace(free=10**9))
    config=configured(rebuild_scratch_directory=str(tmp_path),rebuild_scratch_bytes=32*1024**2)
    with scratch_space.scope(config) as budget:
        for _ in range(32):scratch_space.grow(Path(budget.directory)/'vectors',16384)
        assert len(calls)<=2
        assert budget.used>=32*16384


def test_retained_matrix_directory_releases_lease_with_last_owner(tmp_path):
    import gc
    config=configured(rebuild_scratch_directory=str(tmp_path),rebuild_scratch_bytes=32*1024**2)
    with scratch_space.scope(config) as budget:
        directory=scratch_space.directory(prefix='witnesses-')
        path=Path(directory.name)
        scratch_space.grow(path/'vectors',16384)
        assert budget.used>0
        del directory;gc.collect()
        assert budget.used==0 and not path.exists()
