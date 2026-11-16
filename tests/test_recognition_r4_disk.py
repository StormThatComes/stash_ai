from pathlib import Path
from types import SimpleNamespace
from stash_ai.recognition.disk_budget import scope,reserve_file


def test_file_reservations_share_one_volume_and_release_after_failure(tmp_path):
    path=tmp_path/'db'
    config=SimpleNamespace(database=path,data={'execution':{'disk_total_bytes':1000,'disk_free_bytes':0}})
    path.write_bytes(b'x'*100)
    a,b=(Path(str(path)+suffix) for suffix in ('.a','.b'))
    a.touch();b.touch()
    with scope(config):
        with reserve_file(a,700) as first:
            assert first==700
            a.write_bytes(b'x'*300)
            with reserve_file(b,500) as second:assert second==200
        with reserve_file(b,900) as freed:assert freed==600


def test_solver_index_releases_unused_capacity_while_trials_are_active(tmp_path):
    from benchmarks.benchmark_merge_solver import fixture
    from contextlib import closing
    from stash_ai.recognition.fragmentation_report import _open_readonly
    from stash_ai.recognition.solver_images import Images
    from stash_ai.recognition.solver_slot_index import coordinator
    config, _ = fixture(tmp_path/'fixture', 'face')
    directory = tmp_path/'index'
    directory.mkdir()
    images = Images(directory, config.database, 96*1024**3, True)
    with closing(_open_readonly(config.database)) as db, scope(config) as budget:
        scans = [dict(id=r[0]) for r in db.execute('SELECT id FROM scans')]
        with coordinator(images, db, config, scans) as (index, _):
            assert len(index) > 0
            assert not budget.reservations


def test_parallel_file_admission_rechecks_capacity_after_another_trial_writes(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from stash_ai.recognition.scratch_ledger import reserve

    monkeypatch.setenv('RECOGNITION_WORK_ROOT', str(tmp_path))
    mib = 1024**2
    config = SimpleNamespace(database=tmp_path/'faces', data={'execution': {'disk_total_bytes': 8*mib}})
    first, second = tmp_path/'first', tmp_path/'second'
    first.touch()
    second.touch()
    scanned, resume = Event(), Event()

    def slow_trial():
        with scope(config) as budget:
            check = budget.check
            def slow_check(*args, **kwargs):
                volumes = check(*args, **kwargs)
                if not scanned.is_set():
                    scanned.set()
                    assert resume.wait(10)
                return volumes
            budget.check = slow_check
            with reserve_file(first, 8*mib) as granted:
                return granted

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(slow_trial)
        try:
            assert scanned.wait(10)
            # A different trial can reserve scratch and finish growing a file
            # while the first is still measuring the volume.
            reserve('other', mib, 8*mib)
            with scope(config), reserve_file(second, 5*mib) as granted:
                assert granted == 5*mib
                second.write_bytes(b'x' * granted)
            reserve('other', 0, 8*mib)
        finally:
            resume.set()
        assert 2*mib < future.result() <= 3*mib
