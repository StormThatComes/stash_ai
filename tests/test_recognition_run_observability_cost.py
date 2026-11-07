from collections import OrderedDict
from types import SimpleNamespace
from threading import Event

import pytest

from stash_ai.recognition import gpu_memory, live_resources, prefetch
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.rejection_summary import summary
from tests.test_recognition_gpu_memory import monitor
from tests.test_recognition_offline_evaluation import fixture


def test_rejections_reuse_counts_and_invalidate_transactionally(tmp_path):
    config, _ = fixture(tmp_path)
    with FaceDatabase(config.database) as database:
        db = database.connection
        scan = db.execute('SELECT id FROM scans LIMIT 1').fetchone()[0]
        baseline = summary(db, scan, config)
        queries = []
        db.set_trace_callback(queries.append)
        assert summary(db, scan, config) == baseline
        assert not any('GROUP BY rejected' in query for query in queries)
        db.execute("UPDATE observations SET rejected='blur' WHERE scan=?", (scan,))
        assert summary(db, scan, config)['face']['blur'] > 0 and db.in_transaction
        db.rollback()
        assert summary(db, scan, config) == baseline


def test_volume_with_smallest_enforceable_margin_wins(tmp_path):
    unit = 2**30
    cfg = SimpleNamespace(database=tmp_path/'faces.db', data={'execution': {'disk_total_bytes': 10*unit}})
    budget = SimpleNamespace(checked=10, headroom=unit, volumes={
        1: dict(used=unit,reserved=0,free=4*unit),
        2: dict(used=8*unit,reserved=unit,free=20*unit)})
    view = live_resources.ConsoleResources(cfg,budget=budget)
    assert 'volume 2: cap margin 1.00 GiB · sample 15s ago' in view.disk(25)
    budget.volumes[1]['free'] = unit
    assert 'volume 1: free reserve margin 0.00 GiB' in view.disk(36)


def test_slow_device_probe_does_not_block_or_repeat_per_batch(monkeypatch):
    gpu, saved, _ = monitor(monkeypatch)
    started, release = Event(), Event()
    calls = []
    def query(_):
        calls.append(1)
        started.set()
        assert release.wait(3)
        return 2**30
    monkeypatch.setattr(gpu_memory, 'process_memory', query)
    monkeypatch.setattr(gpu_memory, 'torch_memory', lambda: None)
    gpu.executable = 'nvidia-smi'
    try:
        for _ in range(10):
            gpu.record('faces','before_release')
            gpu.record('faces','after_release')
        assert started.wait(1) and len(calls) == 1 and len(saved) == 2
    finally:
        release.set()
        if gpu.process_thread:
            gpu.process_thread.join(3)
    gpu.record('final','after_release')
    assert gpu.summary()['sampled_peak_bytes']['process_bytes'] == 2**30


def test_prefetch_counts_only_waiting_for_an_unfinished_future(monkeypatch):
    instance = object.__new__(prefetch.ScenePrefetch)
    instance.enabled, instance.report = True, None
    instance.taken = instance.ready = instance.blocked_seconds = 0
    instance.queued = OrderedDict()
    now = iter((10., 12.5))
    monkeypatch.setattr(prefetch, 'monotonic', lambda:next(now))
    for scene, done in [('1',True),('2',False)]:
        future = SimpleNamespace(done=lambda:done, result=lambda:'source')
        instance.queued[scene] = future, None
        assert instance.take(scene) == 'source'
    assert (instance.taken,instance.ready,instance.blocked_seconds) == (2,1,2.5)


def test_recovered_oom_reports_attempt_and_effective_batches_without_cuda(monkeypatch):
    import sys
    from stash_ai.recognition.adaptive_gpu import Batches, observe_batches
    class OOM(RuntimeError):
        pass
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(
        OutOfMemoryError=OOM, is_initialized=lambda:False)))
    gpu, _, _ = monitor(monkeypatch, enabled=False)
    with observe_batches(gpu.batch_change):
        batch = Batches(8,'pose')
    def infer(values):
        if len(values) > 2:
            raise OOM()
        return values
    assert batch.run(list(range(6)), infer, 'cuda') == list(range(6))
    assert gpu.batch_sizes['pose'] == dict(retries=2, maximum=2, previous=3, reduced_to=2)
    assert 'pose 3→2 · effective 2 · 2 retries' in gpu.batch_line()


def test_rejections_follow_body_replacement_and_person_evidence_changes(tmp_path):
    from dataclasses import replace
    from stash_ai.recognition.config import plain
    from stash_ai.recognition.body_observations import contract as body_contract
    from stash_ai.recognition.person_signal import contract
    config, _ = fixture(tmp_path)
    data = plain(config.data)
    data['sapiensid']['enabled'] = data['sapiensid']['collection']['enabled'] = True
    config = replace(config, data=data)
    with FaceDatabase(config.database) as database:
        db = database.connection
        a,b = [r[0] for r in db.execute('SELECT id FROM scans ORDER BY scene')]
        oid = db.execute('SELECT id FROM observations WHERE scan=? LIMIT 1',(a,)).fetchone()[0]
        db.execute("INSERT INTO person_evidence VALUES (?,?, 'unavailable','{}',NULL,NULL)",(oid,contract(config)))
        def body(scan):
            db.execute("INSERT OR REPLACE INTO body_observations VALUES ('p',?,0,0,?,NULL,'{}','{}','blur',NULL,NULL)",
                       (scan,body_contract(config)))
        body(a)
        first = summary(db,a,config)
        assert first['body'] == {'blur':1} and first['person_evidence_unavailable'] == {'unavailable':1}
        body(b)
        db.execute("UPDATE person_evidence SET metadata=? WHERE observation=?",
                   ('{"identity_rejection":"angle"}',oid))
        changed = summary(db,a,config)
        assert changed['body'] == {} and changed['person_evidence_unavailable'] == {'angle':1}
        assert summary(db,b,config)['body'] == {'blur':1}
