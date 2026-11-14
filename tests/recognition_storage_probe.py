"""Offline native-write tracing fixture. Run in a subprocess under strace/Procmon."""
from contextlib import closing, redirect_stdout
import io
import os
from pathlib import Path
import sqlite3
import sys


def probe(root):
    import pytest
    from benchmarks.benchmark_merge_solver import fixture
    from stash_ai.recognition.cli import main
    from stash_ai.recognition.work_locations import invocation
    from tests.test_recognition_run import setup
    from tests.test_recognition_solver_yaml_diff import arguments

    root.mkdir(exist_ok=True,parents=True)
    (root/'probe.pid').write_text(str(os.getpid()))
    with pytest.MonkeyPatch.context() as patch:
        cfg, _, services, _ = setup(root, patch)
        argv = ['run','--face-config',str(cfg.source_path),'--scene-id','1','--progress-style','off']
        assert main(argv, services=services) == 0
        assert main(argv, services=services) == 0
    cfg, goals = fixture(root/'solver','disjoint')
    # Materialize a real, file-backed TEMP database before the longer solver runs.
    # An ORDER BY may complete entirely in SQLite's sorter buffers.
    with invocation(cfg), closing(sqlite3.connect(':memory:')) as db:
        db.execute('PRAGMA temp_store=FILE')
        db.execute('PRAGMA temp.cache_size=4')
        db.execute('CREATE TEMP TABLE spill(value BLOB)')
        db.executemany('INSERT INTO temp.spill VALUES (zeroblob(4096))', (() for _ in range(2048)))
        db.commit()
        assert db.execute('PRAGMA temp.page_count').fetchone()[0] > 4
        assert db.execute('SELECT count(*) FROM temp.spill').fetchone()[0] == 2048
    checkpoint = root/'checkpoint.json'
    argv = arguments(cfg,goals,checkpoint,maximum=6)
    argv[argv.index('--yaml-output')] = '--checkpoint'
    argv += ['--solver-cpus','4','--solver-trials','4','--solver-memory-mb','65536']
    # Admission in this probe is synthetic, including on two-core CI runners.
    with pytest.MonkeyPatch.context() as patch, redirect_stdout(io.StringIO()):
        patch.setattr(os, 'process_cpu_count', lambda:4, raising=False)
        assert main(argv) == 0
        assert main(argv+['--resume']) == 0


if __name__ == '__main__':
    probe(Path(sys.argv[1]).resolve())
