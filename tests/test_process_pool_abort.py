"""Failure cleanup kills read-only workers and preserves the original exception."""
import pytest
from pathlib import Path
import signal
import subprocess
import sys
import time

from stash_ai.runtime.process_pool import read_process_pool


@pytest.mark.parametrize('failure', [RuntimeError, KeyboardInterrupt, GeneratorExit])
def test_failure_kills_pool_without_waiting_or_masking(failure):
    calls = []
    class Pool:
        def kill_workers(self):
            calls.append('kill')
            raise OSError('secondary failure')
        def shutdown(self, **options):
            pytest.fail('unbounded shutdown')
    with pytest.raises(failure, match='original'):
        with read_process_pool(Pool):
            raise failure('original')
    assert calls == ['kill']


def test_success_waits_for_normal_pool_shutdown():
    calls = []
    class Pool:
        def shutdown(self, **options):
            calls.append(options)
    with read_process_pool(Pool):
        pass
    assert calls == [dict(wait=True)]


@pytest.mark.skipif(sys.version_info < (3, 14), reason='Public process-pool termination requires supported Python 3.14')
def test_sigint_exits_real_spawn_pool_with_no_orphan_worker(tmp_path):
    pid_path = tmp_path/'worker.pid'
    process = subprocess.Popen([sys.executable, '-m', 'tests.helpers.blocked_pool', str(pid_path)],
        cwd=Path(__file__).resolve().parents[1], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        limit = time.monotonic() + 10
        while not pid_path.exists() and time.monotonic() < limit and process.poll() is None:
            time.sleep(.02)
        assert pid_path.exists()
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 0, stdout + stderr
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
