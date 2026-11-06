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
