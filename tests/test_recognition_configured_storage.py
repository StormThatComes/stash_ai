import errno
import os
import shutil
import sqlite3
import stat
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from stash_ai.recognition.config import load
from stash_ai.recognition.disk_budget import Budget, _volume_space
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


@pytest.mark.parametrize('target_is_removed', [False, True])
def test_disk_check_survives_trial_cleanup_after_scan(tmp_path, monkeypatch, target_is_removed):
    database = tmp_path / 'faces.db'
    database.write_bytes(b'db')
    scratch = tmp_path / 'work'
    trial = scratch / 'trial-32'
    trial.mkdir(parents=True)
    spool = trial / 'replay.sqlite'
    spool.write_bytes(b'trial')
    budget = Budget(SimpleNamespace(database=database, data={'execution': {'disk_total_bytes': 1024}}))
    budget.roots.add(scratch)

    disk_usage = shutil.disk_usage

    def cleanup_then_measure(path):
        if trial.exists():
            shutil.rmtree(trial)
        return disk_usage(path)

    device, _ = _volume_space(database)
    monkeypatch.setattr(shutil, 'disk_usage', cleanup_then_measure)

    target = spool if target_is_removed else database.parent
    result = budget.check(growth=3, path=target)[device]
    assert result['used'] == 7
    assert result['requested'] == 10
    assert Path(result['path']).exists()
    assert result['free'] > 0
    assert budget.check()[device]['used'] == 2


def test_disappeared_volume_cannot_borrow_space_from_parent(tmp_path, monkeypatch):
    missing = tmp_path / 'unmounted' / 'trial'
    other_device = 'unmounted-volume' if os.name == 'nt' else tmp_path.stat().st_dev + 1
    monkeypatch.setattr(shutil, 'disk_usage', lambda path: pytest.fail(f'Probed a different volume at {path}'))

    with pytest.raises(OSError) as error:
        _volume_space(missing, other_device)
    assert error.value.errno == errno.ENODEV


def partial_windows_metadata(monkeypatch, anonymous, *, mount=None, lstat_only=False):
    from stash_ai.recognition import disk_budget

    real_stat = Path.stat
    def partial_stat(path, *, follow_symlinks=True):
        info = real_stat(path, follow_symlinks=follow_symlinks)
        if (lstat_only and follow_symlinks) or (path not in anonymous and path != mount):
            return info
        values = {name: getattr(info, name) for name in dir(info) if name.startswith('st_')}
        if path in anonymous:
            values.update(st_dev=0, st_ino=0)
        if path == mount and not follow_symlinks:
            values['st_file_attributes'] = stat.FILE_ATTRIBUTE_REPARSE_POINT
        return SimpleNamespace(**values)

    if mount is not None:
        monkeypatch.setattr(disk_budget, '_windows_volume',
                            lambda path: 'nested-volume' if path.is_relative_to(mount) else 'host-volume')
    elif os.name != 'nt':
        monkeypatch.setattr(disk_budget, '_windows_volume', lambda path: 'test-volume')
    monkeypatch.setattr(Path, 'stat', partial_stat)


@pytest.mark.parametrize('partial_directory', [False, True])
def test_parallel_scratch_files_with_zero_device_and_inode(tmp_path, monkeypatch, partial_directory):
    from stash_ai.recognition.resource_errors import ResourceLimit

    database = tmp_path / 'faces.db'
    database.write_bytes(b'ab')
    work = tmp_path / 'recognition-scratch'
    work.mkdir()
    first, second = work / 'first', work / 'second'
    first.write_bytes(b'123')
    second.write_bytes(b'4567')
    config = SimpleNamespace(database=database, data={'execution': {'disk_total_bytes': 12}})
    budget = Budget(config)
    budget.roots.add(work)
    partial_windows_metadata(monkeypatch, {first, second, work} if partial_directory else {first, second},
                             lstat_only=True)
    rows = budget.check(growth=3)
    assert len(rows) == 1
    row, = rows.values()
    assert (row['used'], row['requested']) == (9, 12)
    with pytest.raises(ResourceLimit, match='per-volume'):
        budget.check(growth=4)


def test_partial_metadata_reservation_is_charged_to_its_mounted_volume(tmp_path, monkeypatch):
    from stash_ai.recognition.disk_budget import reserve_file, scope

    work = tmp_path / 'work'
    mount = work / 'mount'
    mount.mkdir(parents=True)
    spool = mount / 'trial.sqlite'
    spool.write_bytes(b'123')
    monkeypatch.setenv('RECOGNITION_WORK_ROOT', str(work))
    partial_windows_metadata(monkeypatch, {spool}, mount=mount)

    config = SimpleNamespace(database=tmp_path / 'faces.db', data={'execution': {'disk_total_bytes': 1024**2}})
    observer = Budget(config)
    with scope(config), reserve_file(spool, 1024) as grant:
        assert grant == 1024
        rows = observer.check(growth=5, path=spool)
        assert rows['host-volume']['reserved'] == 0
        nested = rows['nested-volume']
        assert (nested['used'], nested['reserved'], nested['requested']) == (3, 1021, 1029)
    assert observer.check()['nested-volume']['reserved'] == 0


@pytest.mark.skipif(os.name != 'nt', reason='Windows mount paths')
@pytest.mark.parametrize('mount', ['\\\\Server\\Share\\', '\\\\?\\UNC\\SERVER\\SHARE\\'])
def test_windows_share_mounts_have_the_same_budget_key(monkeypatch, tmp_path, mount):
    from stash_ai.recognition import windows_volume

    monkeypatch.setattr(windows_volume, '_getvolumepathname', lambda path: mount)
    assert windows_volume.volume_key(tmp_path) == '\\\\server\\share\\'
