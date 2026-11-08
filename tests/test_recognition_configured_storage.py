import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

from stash_ai.recognition.config import load
from stash_ai.recognition.work_locations import invocation, validate_outputs, work_root


def test_native_work_paths_and_legacy_default(tmp_path):
    path = tmp_path / 'recognition.yml'
    path.write_text('version: 2\nstorage: {database: state/faces.sqlite}\n')
    config = load(path)
    root = tmp_path / 'state' / 'recognition_work'
    assert work_root(config) == root
    previous = os.environ.get('SQLITE_TMPDIR')
    with invocation(config):
        work = Path(tempfile.gettempdir())
        assert work.is_relative_to(root)
        assert {os.environ[name] for name in ('TMP', 'TEMP', 'TMPDIR', 'SQLITE_TMPDIR')} == {str(work)}
        with tempfile.NamedTemporaryFile() as file:
            assert Path(file.name).is_relative_to(work)
        with sqlite3.connect(work/'sort.sqlite') as db:
            assert db.execute('PRAGMA temp_store_directory').fetchone() == (str(work),)
            db.execute('PRAGMA temp_store=FILE')
            db.execute('PRAGMA cache_size=4')
            db.execute('CREATE TABLE data(value BLOB)')
            db.executemany('INSERT INTO data VALUES (?)', ((os.urandom(1024),) for _ in range(2048)))
            assert sum(1 for _ in db.execute('SELECT value FROM data ORDER BY value')) == 2048
        db.close()
        validate_outputs(config, (root/'result.json',))
        with pytest.raises(ValueError, match='outside configured'):
            validate_outputs(config, (tmp_path/'other.json',))
    assert not work.exists()
    assert os.environ.get('SQLITE_TMPDIR') == previous


def test_trace_resolves_relative_file_removal_against_directory_fd(tmp_path):
    from tests.recognition_write_trace import linux
    trace = tmp_path/'native.log'
    trace.write_text(f'100   unlinkat(12<{tmp_path}>, "observations.sqlite", 0) = 0\n'
                     f'101  renameat(8<{tmp_path}>, "a", 9<{tmp_path}/other>, "b") = 0\n')
    assert list(linux(trace)) == [tmp_path/'observations.sqlite', tmp_path/'a', tmp_path/'other'/'b']


def test_direct_solver_uses_configured_work_root(tmp_path):
    from contextlib import closing
    from benchmarks.benchmark_merge_solver import fixture
    from stash_ai.recognition.fragmentation_report import _open_readonly
    from stash_ai.recognition.merge_solver import solve
    config, goals = fixture(tmp_path/'data','face')
    captured = []
    def observe(name, phase, item):
        if name == 'baseline' and phase == 'start':
            captured.append(Path(tempfile.gettempdir()))
    with closing(_open_readonly(config.database)) as source:
        solve(config,goals,source,maximum=1,progress=observe)
    assert len(captured)==1 and captured[0].parent == work_root(config)
    assert not captured[0].exists()
