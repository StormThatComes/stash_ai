"""Replay pages and concurrent workers must spend the budgets they advertise."""
from contextlib import closing
from dataclasses import replace

import pytest

from stash_ai.recognition.config import plain
from stash_ai.recognition.memory_size import Reservation
from stash_ai.recognition.ram_database import connect
from stash_ai.recognition.resource_errors import ResourceLimit
from stash_ai.recognition.run_cache import scope as cache_scope
from tests.test_recognition_clustering import setup

MIB = 1024**2
GIB = 1024**3


def settings(tmp_path, budget):
    config = setup(tmp_path)
    data = plain(config.data)
    data['execution']['rebuild_memory_bytes'] = budget
    return replace(config, data=data)


def test_work_can_borrow_unused_replay_capacity_and_failed_reservation_is_atomic(tmp_path):
    # Scaled version of 20 GiB rejecting 6 GiB of working reservations at /4.
    config = settings(tmp_path, 20*MIB)
    with cache_scope(config) as cache, closing(connect(config, ':memory:', label='sqlite-test')) as db:
        db.execute('CREATE TABLE work(value BLOB)')
        db.execute('INSERT INTO work VALUES (zeroblob(?))', (2*MIB,))
        db.commit()
        pages = cache.reserved_size
        bank = Reservation(20*MIB, label='face-bank')
        bank.set(6*MIB)
        assert cache.reserved_size == pages + 6*MIB
        before = dict(cache.reservations)
        with pytest.raises(ResourceLimit) as failure:
            bank.set(19*MIB)
        assert cache.reservations == before
        assert failure.value.document() == dict(kind='configured_resource_limit',
            budget='execution.rebuild_memory_bytes', limit=20*MIB, requested=pages+19*MIB)
        assert 'owner=face-bank' in str(failure.value)
        assert 'sqlite-test=' in str(failure.value)
        bank.set(0)
    assert cache.reserved_size == 0 and not cache.databases


@pytest.mark.parametrize('api', ['connection', 'cursor', 'script'])
def test_live_sqlite_pages_and_python_reservations_share_a_hard_page_capacity(tmp_path, api):
    config = settings(tmp_path, 4*MIB)
    with cache_scope(config) as cache, closing(connect(config, ':memory:', label='sqlite-test')) as db:
        db.execute('CREATE TABLE work(value BLOB)')
        db.commit()
        reservation = Reservation(4*MIB, label='queue')
        reservation.set(3*MIB)
        with pytest.raises(ResourceLimit, match='owner=sqlite-test'):
            if api == 'script':
                db.executescript('INSERT INTO work VALUES (zeroblob(2097152));')
            else:
                writer = db if api == 'connection' else db.cursor()
                writer.execute('INSERT INTO work VALUES (zeroblob(?))', (2*MIB,))
        assert not db.execute('SELECT * FROM work').fetchall()
        assert cache.reserved_size <= cache.working
        reservation.set(0)
        # The same database can now use more than the former half-budget cap.
        db.execute('INSERT INTO work VALUES (zeroblob(?))', (3*MIB,))
        db.commit()
        assert cache.reserved_size > 3*MIB


def test_two_ram_databases_borrow_released_space_without_duplicate_budgets(tmp_path):
    config = settings(tmp_path, 4*MIB)
    with cache_scope(config) as cache:
        with closing(connect(config, ':memory:', label='sqlite-outer')) as outer:
            outer.execute('CREATE TABLE work(value BLOB)')
            outer.execute('INSERT INTO work VALUES (zeroblob(?))', (2*MIB,))
            outer.commit()
            with closing(connect(config, ':memory:', label='sqlite-inner')) as inner:
                inner.execute('CREATE TABLE work(value BLOB)')
                with pytest.raises(ResourceLimit):
                    inner.execute('INSERT INTO work VALUES (zeroblob(?))', (2*MIB,))
                inner.execute('INSERT INTO work VALUES (zeroblob(?))', (MIB,))
                inner.commit()
                assert cache.reserved_size <= 4*MIB
            assert all(key[0] != 'sqlite-inner' for key in cache.reservations)
            outer.execute('INSERT INTO work VALUES (zeroblob(?))', (MIB,))
            outer.commit()
        assert cache.reserved_size == 0


def test_sqlite_growth_evicts_optional_cache_and_rollback_releases_pages(tmp_path):
    config = settings(tmp_path, 4*MIB)
    with cache_scope(config) as cache, closing(connect(config, ':memory:', label='sqlite-test')) as db:
        db.execute('CREATE TABLE work(value BLOB)')
        db.commit()
        original = cache.reserved_size
        cache.put(('metadata', 'test'), 'cached', 3*MIB)
        db.execute('INSERT INTO work VALUES (zeroblob(?))', (2*MIB,))
        assert cache.size + cache.reserved_size <= cache.working
        db.rollback()
        assert cache.reserved_size == original
        with pytest.raises(RuntimeError, match='discard transaction'):
            with db:
                db.execute('INSERT INTO work VALUES (zeroblob(?))', (2*MIB,))
                raise RuntimeError('discard transaction')
        assert cache.reserved_size == original


def test_solver_divides_large_yaml_limit_and_enforces_effective_worker_budget(tmp_path, monkeypatch):
    from stash_ai.recognition import solver_execution
    config = settings(tmp_path, 30*GIB)
    monkeypatch.setattr(solver_execution.os, 'process_cpu_count', lambda: 16, raising=False)
    monkeypatch.setattr(solver_execution, 'available_memory', lambda: 80*GIB)
    runtime = solver_execution.plan(config, dict(cpus=16, trials=4))
    assert runtime.trials == 4 and runtime.trial_bytes * 4 <= 64*GIB
    assert 0 < runtime.rebuild_memory_bytes < 30*GIB
    assert 'rebuild_memory_gib=' in solver_execution.plan_text(runtime, maximum=5)
    assert runtime.signature()['rebuild_memory_bytes'] == runtime.rebuild_memory_bytes
    with solver_execution.scope(runtime), cache_scope(config) as cache:
        assert cache.working == runtime.rebuild_memory_bytes
        with pytest.raises(ResourceLimit) as failure:
            cache.reserve(('test', 1), cache.working + 1)
        assert failure.value.limit == runtime.rebuild_memory_bytes
    assert config.data['execution']['rebuild_memory_bytes'] == 30*GIB
    with cache_scope(config) as cache:
        assert cache.working == 30*GIB


def test_effective_solver_cap_also_limits_the_baseline_and_ram_pages(tmp_path, monkeypatch):
    from stash_ai.recognition import solver_execution
    from stash_ai.recognition.offline_evaluation import _ram
    from stash_ai.recognition.replay_memory import memory_replay
    config = settings(tmp_path, 30*GIB)
    monkeypatch.setattr(solver_execution.os, 'process_cpu_count', lambda: 4, raising=False)
    overhead_plan = solver_execution.plan(config, dict(cpus=4, trials=4, memory_bytes=200*GIB))
    overhead = overhead_plan.trial_bytes - overhead_plan.rebuild_memory_bytes
    runtime = solver_execution.plan(config, dict(cpus=4, trials=4, memory_bytes=4*(overhead+4*MIB)))
    baseline = replace(runtime, trials=1)
    with solver_execution.scope(baseline), cache_scope(config) as cache, closing(_ram(config)) as outer:
        with memory_replay(config) as (inner, _):
            assert inner.budget is cache and outer.budget is cache
            inner.execute('CREATE TABLE test_limit(value BLOB)')
            with pytest.raises(ResourceLimit):
                inner.execute('INSERT INTO test_limit VALUES (zeroblob(?))', (4*MIB,))
            assert cache.working == 4*MIB
