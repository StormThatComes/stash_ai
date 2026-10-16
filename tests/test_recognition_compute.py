"""Owned CPU work remains bounded and cannot reorder identity decisions."""
from dataclasses import replace
from threading import Event, get_ident
from types import SimpleNamespace

import pytest

from stash_ai.recognition import compute
from stash_ai.recognition.config import defaults, validate
from stash_ai.recognition.clustering import matching_policy
from stash_ai.runtime.cancellation import Cancelled


def configured(**values):
    data = defaults()
    data['execution'].update(values)
    return SimpleNamespace(data=data)


@pytest.fixture
def cpus(monkeypatch):
    monkeypatch.setattr(compute, 'available_cpus', lambda: 24)


def test_budget_and_execution_options_do_not_change_semantic_policy(cpus):
    serial = configured()
    parallel = configured(rebuild_workers=24, rebuild_native_threads=2)
    validate(parallel.data)
    assert matching_policy(serial) == matching_policy(parallel)
    assert compute.Compute(parallel).workers == 12
    assert compute.Compute(configured(rebuild_workers=0)).workers == 16
    assert compute.Compute(configured(rebuild_workers=20)).workers == 20
    assert compute.Compute(configured(rebuild_memory_bytes=1048576)).limit == 65536


@pytest.mark.parametrize('field,value', [('rebuild_workers', True), ('rebuild_workers', -1),
    ('rebuild_workers', 257), ('rebuild_native_threads', 0), ('rebuild_queue_bytes', 65535)])
def test_invalid_execution_budgets_name_the_field(field, value):
    with pytest.raises(ValueError, match=field):
        validate(configured(**{field: value}).data)


def test_jobs_finish_out_of_order_but_are_consumed_in_order(cpus):
    second = Event()
    caller = get_ident()
    def work(index):
        assert get_ident() != caller
        assert compute.current() is None
        if index == 0:
            assert second.wait(5)
        else:
            second.set()
        return index
    with compute.compute_scope(configured(rebuild_workers=2, rebuild_queue_bytes=65536)) as owner:
        results = list(owner.ordered(work, (compute.Work(i, 'v1', i, 32768) for i in range(8))))
        assert [r.value for r in results] == list(range(8))
        assert owner.peak_bytes <= 65536
        assert owner.submitted == owner.applied == 8
        executor = owner.executor
        list(owner.ordered(work, [compute.Work(0, 'v2', 1, 1024)]))
        assert owner.executor is executor
    assert owner.closed and not owner.pending and owner.native is None


def test_idle_and_serial_scopes_never_create_executor():
    with compute.compute_scope(configured()) as owner:
        assert list(owner.ordered(lambda x: x+1, [compute.Work(0, 1, 3, 1024)]))[0].value == 4
        assert owner.executor is None
    assert compute.current() is None


def test_failure_is_raised_at_its_ordered_position_and_restores_native_limits(cpus):
    from threadpoolctl import threadpool_info
    import numpy as np
    np.dot(np.ones(2), np.ones(2))
    original = threadpool_info()
    def work(index):
        if index == 1:
            raise ValueError('second observation')
        return index
    with pytest.raises(ValueError, match='second observation'):
        with compute.compute_scope(configured(rebuild_workers=2)) as owner:
            stream = owner.ordered(work, (compute.Work(i, 1, i, 1024) for i in range(4)))
            assert next(stream).value == 0
            next(stream)
    assert threadpool_info() == original
    assert owner.native is None and not owner.pending


def test_abandoned_stream_cannot_schedule_more_work(cpus):
    with compute.compute_scope(configured(rebuild_workers=2)) as owner:
        stream = owner.ordered(lambda x: x, (compute.Work(i, 1, i, 1024) for i in range(8)))
        next(stream)
        stream.close()
        with pytest.raises(Cancelled):
            list(owner.ordered(lambda x: x, [compute.Work(0, 1, 0, 1024)]))


def test_reject_stale_generation_and_invalid_reservation(cpus, monkeypatch):
    with compute.compute_scope(configured(rebuild_workers=2)) as owner:
        original = owner._invoke
        monkeypatch.setattr(owner, '_invoke', lambda f, w: replace(original(f, w), generation='stale'))
        with pytest.raises(RuntimeError, match='Stale'):
            list(owner.ordered(lambda x: x, [compute.Work(0, 'current', 0, 1024)]))
    with compute.compute_scope(configured()) as owner:
        with pytest.raises(ValueError, match='size/order'):
            list(owner.ordered(lambda x: x, [compute.Work(0, 1, 0, owner.limit+1)]))


def test_only_coordinator_schedules(cpus):
    from concurrent.futures import ThreadPoolExecutor
    with compute.compute_scope(configured(rebuild_workers=2)) as owner:
        with ThreadPoolExecutor(max_workers=1) as other:
            result = other.submit(lambda: list(owner.ordered(lambda x: x, [])))
            with pytest.raises(RuntimeError, match='coordinator'):
                result.result()


def test_live_native_work_keeps_owners_quarantined_until_it_exits(cpus, monkeypatch):
    from stash_ai.runtime import shutdown
    started, finish = Event(), Event()
    monkeypatch.setattr(shutdown, 'TIMEOUT', .01)
    owner = compute.Compute(configured(rebuild_workers=2))
    def work(value):
        started.set()
        assert finish.wait(5)
        return value
    def interrupted():
        assert started.wait(5)
        raise KeyboardInterrupt('stop during native work')
    monkeypatch.setattr(owner, '_take', interrupted)
    try:
        with pytest.raises(KeyboardInterrupt, match='stop during native work') as error:
            with compute.compute_scope(configured()) as unused:
                try:
                    list(owner.ordered(work, [compute.Work(0, 1, 0, 1024)]))
                finally:
                    from stash_ai.runtime.cleanup import cleanup
                    cleanup([owner.close])
        assert shutdown.busy()
        assert owner.native is not None and owner.pending
        assert any('ShutdownIncomplete' in note for note in error.value.__notes__)
        with pytest.raises(shutdown.ShutdownIncomplete):
            shutdown.require_idle()
    finally:
        finish.set()
        owner.executor.shutdown(wait=True, cancel_futures=True)
        # The owned shutdown helper can finish a moment after the executor's
        # workers. Wait for that actual owner, not a timing-dependent idle poll.
        shutdown._calls[id(owner)][3].join(5)
        shutdown.require_idle()
    assert owner.native is None and not owner.pending
