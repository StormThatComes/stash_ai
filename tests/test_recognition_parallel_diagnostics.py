"""CPU completion is observable without advancing semantic scene progress."""
from threading import Event, Timer, get_ident

from stash_ai.recognition import compute, parallel_scores
from stash_ai.runtime import console
from stash_ai.runtime.console_layout import status_lines
from tests.test_recognition_compute import configured
from tests.test_recognition_score_blocks import groups


def test_wait_heartbeat_is_coordinator_owned_and_does_not_advance_scenes(monkeypatch):
    monkeypatch.setattr(compute, 'available_cpus', lambda: 24)
    owner, updates, release = get_ident(), [], Event()
    def update(report, *, completed=0, **values):
        updates.append((get_ident(), completed, values))
    monkeypatch.setattr(console, 'status', update)
    timer = Timer(.65, release.set)
    timer.start()
    try:
        with compute.compute_scope(configured(rebuild_workers=2)) as pool:
            def work(value):
                assert release.wait(5)
                return value
            assert list(pool.ordered(work, [compute.Work(0, 'fixed', 1, 4096)]))[0].value == 1
            summary = pool.summary()
            assert summary['completed_blocks'] == summary['retired_blocks'] == 1
            assert summary['strict_scenes_applied'] == 0
        assert all(thread == owner and completed == 0 for thread, completed, _ in updates)
        assert any(v.get('cpu_wait') == 'ordered CPU result' for _, _, v in updates)
        assert updates[-1][2]['cpu_wait'] is None
    finally:
        release.set()
        timer.join()


def test_early_prefix_reports_unused_scores_and_fallbacks(monkeypatch):
    monkeypatch.setattr(compute, 'available_cpus', lambda: 24)
    references, query = groups(128)
    rows = [dict(id=str(i), dimensions=128, embedding=query.tobytes()) for i in range(12)]
    with compute.compute_scope(configured(rebuild_workers=4, rebuild_queue_bytes=2097152)) as pool:
        stream = parallel_scores.scored_rows(rows, references, .65)
        assert next(stream)[1] is not None
        stream.close()
        record = pool.summary()['records']['score_rows']
        assert record == dict(computed=12, consumed=1, discarded=11)
        assert pool.summary()['strict_scenes_applied'] == 0
        assert pool.preparation['packing_seconds'] >= 0
        list(parallel_scores.scored_rows(rows, references[:1], .65))
        assert pool.fallbacks['small_catalogue'] == 1


def test_cpu_fields_wrap_without_hiding_stage_or_fabricating_eta():
    fields = dict(stage='ordered replay', cpu_workers='4/16', cpu_wait='ordered CPU result',
                  cpu_blocks='20 computed/12 retired', cpu_queue='65536/1048576 B')
    lines = status_lines(dict(phase='Recognition', elapsed=10, last_progress_age=None), fields, 48)
    assert all(len(line) <= 48 for line in lines)
    assert lines[1].startswith('Stage:')
    assert any(line.startswith('CPU:') for line in lines)
    assert not any('eta=' in line for line in lines)
