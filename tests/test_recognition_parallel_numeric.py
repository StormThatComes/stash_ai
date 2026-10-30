from types import SimpleNamespace
from threading import get_ident
import numpy as np

from stash_ai.recognition.compute import compute_scope, Work
from stash_ai.recognition.parallel_numeric import map_bounded
from stash_ai.recognition.merge_solver_measurements import reference_stats, measured_pairs
from stash_ai.recognition.run_cache import scope as cache_scope
from tests.test_recognition_compute import configured


def test_batches_retire_before_nested_consumer_and_obey_budget(monkeypatch):
    monkeypatch.setattr('stash_ai.recognition.compute.available_cpus', lambda: 16)
    owner = get_ident()
    with compute_scope(configured(rebuild_workers=4, rebuild_queue_bytes=1048576)) as pool:
        seen = []
        for item, (value, thread) in map_bounded(lambda x: (x*x, get_ident()), range(33),
                cost=lambda _: 90000, kind='test'):
            assert not pool.pending and thread != owner
            assert list(pool.ordered(lambda x: x, [Work(0, 'nested', value, 4096)]))[0].value == value
            seen.append((item, value))
        assert seen == [(x, x*x) for x in range(33)]
        assert pool.peak_bytes <= pool.limit


def test_complete_measurements_match_serial_including_weakest_ties(monkeypatch):
    monkeypatch.setattr('stash_ai.recognition.compute.available_cpus', lambda: 16)
    rng = np.random.default_rng(182)
    def bank(n):
        return SimpleNamespace(samples=tuple(np.frombuffer(row.tobytes(), dtype='<f4')
            for row in rng.normal(size=(n, 64)).astype('<f4')))
    groups = [bank(n) for n in (31, 23, 17, 9)]
    pairs = [(a, b) for a in groups for b in groups]
    expected = [reference_stats(a.samples, b.samples, max_bytes=32768) for a, b in pairs]
    config = configured(rebuild_workers=4)
    with cache_scope(config), compute_scope(config) as pool:
        actual = [v for _, _, v in measured_pairs(pairs, max_bytes=32768)]
        assert actual == expected
        assert [v for _, _, v in measured_pairs(pairs, max_bytes=32768)] == actual
        assert pool.records['face_measurements']['computed'] == len(pairs)


def test_trial_cancellation_reaches_inner_workers(monkeypatch):
    from threading import Event, Timer
    from stash_ai.runtime.cancellation import Cancellation, Cancelled, use, current
    monkeypatch.setattr('stash_ai.recognition.compute.available_cpus', lambda: 16)
    outer = Cancellation()
    started = Event()
    def work(_):
        started.set()
        while not outer.event.wait(.01):
            current().check()
        current().check()
    timer = Timer(.05, outer.cancel)
    timer.start()
    try:
        with use(outer), compute_scope(configured(rebuild_workers=2)) as pool:
            import pytest
            with pytest.raises(Cancelled):
                list(pool.ordered(work, [Work(0, 'cancel', None, 4096)]))
        assert started.is_set()
    finally:
        outer.cancel()
        timer.join()
