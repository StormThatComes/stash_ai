"""Throttled telemetry and bounded worker chunks preserve observed outcomes."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from threading import Event, Thread
from types import SimpleNamespace
import pickle

import pytest

from stash_ai.runtime import profile as profiling
from stash_ai.runtime.profile import ProfileMetrics
from stash_ai.runtime.worker_result import WorkerResult, WorkerChunk, pack_chunk, validated_chunks
from stash_ai.history import lifecycle
from stash_ai.history.lifecycle import RunHistory
from stash_ai.commands.arguments import parse_args
from stash_ai.inference.buffer_usage import BatchQueue, BufferUsage, PreparedBatchUsage, queued_batches
from stash_ai.inference import batches, preparation
from stash_ai.runs.inference_progress import buffer_lines
from stash_ai.models import WD, SOURCES
from stash_ai.storage.store import StateStore
from stash_ai.publication.parallel import iter_parallel_evidence
from stash_ai.reports.model.workers import build_report_parallel
from tests.helpers.runtime import runtime as app
from tests.helpers.evidence import empty_policies
from tests.test_merge_parallel import _outputs

pytestmark = pytest.mark.usefixtures('reset_temporal')


def metric_rows():
    metrics = ProfileMetrics()
    metrics.record('stage', 'compute', .001, .0005, units=1, unit='scenes')
    metrics.record('stage', 'compute', .002, .001, True, unit='scenes')
    return metrics.export()


def test_metric_samples_allocate_once_per_key_under_concurrent_record_and_ingest(monkeypatch):
    original, allocations = profiling.empty_sample, []
    def allocate():
        allocations.append(1)
        return original()
    monkeypatch.setattr(profiling, 'empty_sample', allocate)
    rows = metric_rows()
    allocations.clear()
    metrics = ProfileMetrics()
    def update(_):
        metrics.record('stage', 'compute', .001, .0005, units=1, unit='scenes')
        metrics.ingest(rows)
    with ThreadPoolExecutor(4) as executor:
        list(executor.map(update, range(100)))
    assert len(allocations) == 2
    by_outcome = {row['outcome']: row for row in metrics.export()}
    assert by_outcome['ok']['calls'] == by_outcome['ok']['units'] == 200
    assert by_outcome['error']['calls'] == by_outcome['error']['errors'] == 100
    assert by_outcome['error']['units'] == 0
    assert sum(by_outcome['ok']['histogram']) == 200
    assert profiling.describe(by_outcome['ok'])['p95_seconds'] == .001


def test_history_throttles_before_copying_and_finalizes_interrupted_counts(tmp_path, monkeypatch):
    args = parse_args(app, ['--merge-only', '--state-db', str(tmp_path/'state.db')])
    ticks = [100.]
    monkeypatch.setattr(lifecycle.time, 'perf_counter', lambda: ticks[0])
    history = RunHistory(app, args).__enter__()
    copies, writes, finished = [], [], []
    original = lifecycle.Counter
    def copy(*a, **kw):
        copies.append(1)
        return original(*a, **kw)
    monkeypatch.setattr(lifecycle, 'Counter', copy)
    monkeypatch.setattr(history.store, 'progress', lambda *a, **kw: writes.append(kw['counts']))
    monkeypatch.setattr(history.store, 'finish', lambda *a, **kw: finished.append((a, kw)))
    watched = Counter(processed=1)
    args.report.count_sources.append(watched)
    for _ in range(100):
        history.progress()
    assert not copies and not writes
    watched['processed'] = 32
    history.progress()
    assert len(copies) == 1 and writes[-1]['processed'] == 32
    history.progress(force=True)
    assert len(copies) == 2
    thread = Thread(target=lambda: history.progress(force=True))
    thread.start(); thread.join()
    assert len(copies) == 2
    ticks[0] += 30
    watched['processed'] = 33
    history.progress()
    assert len(copies) == 3 and writes[-1]['processed'] == 33
    watched['processed'] = 34
    history.__exit__(KeyboardInterrupt, KeyboardInterrupt(), None)
    assert len(copies) == 4 and finished[0][1]['counts']['processed'] == 34
    assert finished[0][0][2] == 'interrupted'


def test_queue_ownership_transfer_is_atomic_and_format_shows_all_counts():
    queue = BatchQueue(2)
    queue.producing(True)
    assert queued_batches(queue) == BufferUsage(0, 1, 4)
    queue.put(([1], ['a']))
    assert queued_batches(queue) == BufferUsage(1, 0, 4)
    queue.producing(True)
    assert queued_batches(queue).occupied == 2
    queue.get_nowait()
    assert queued_batches(queue) == BufferUsage(0, 2, 4)
    queue.producing(False)
    queue.consuming(False)
    snapshot = dict(current=dict(input=BufferUsage(1, 2, 4), result=queued_batches(queue)),
                    next=None, cache=(0, 0), checkpoint=(0, 32))
    assert 'ready=1 in-flight=2 occupied=3/4' in buffer_lines(snapshot)[0]
    assert 'ready=0 in-flight=0 occupied=0/4' in buffer_lines(snapshot)[0]


def test_preparation_stays_in_flight_while_consumer_waits(monkeypatch):
    entered, release, finished = Event(), Event(), Event()
    class Image:
        pass
    def frames(*a):
        yield 'a', Image()
    def prepare(image):
        entered.set()
        assert release.wait(5)
        return 1
    monkeypatch.setattr(preparation.sprites, 'iter_sprite_frames', frames)
    tagger = SimpleNamespace(batch_size=1, prepare=prepare)
    usage = PreparedBatchUsage(1)
    stream = preparation.prepared_batches(None, tagger, None, {}, [], prefetch=1, buffer_usage=usage)
    submitted = Event()
    original_submit = usage.submitted
    def track(future):
        original_submit(future)
        submitted.set()
    usage.submitted = track
    results = []
    def consume():
        with closing(stream):
            results.extend(stream)
        finished.set()
    thread = Thread(target=consume)
    thread.start()
    try:
        assert entered.wait(5) and submitted.wait(5)
        assert usage.snapshot() == BufferUsage(0, 1, 2)
    finally:
        release.set()
        thread.join(5)
    assert finished.is_set() and results == [([1], ['a'])]
    assert usage.snapshot() == BufferUsage(0, 0, 2)


@pytest.mark.parametrize('failure', [False, True])
def test_model_in_flight_clears_after_success_or_error(monkeypatch, failure):
    entered, release = Event(), Event()
    def inputs(*a, **kw):
        yield [1], ['a']
    def predict(images):
        import numpy as np
        entered.set()
        assert release.wait(5)
        if failure:
            raise ValueError('model failed')
        return np.array([[.5]], dtype=np.float32)
    monkeypatch.setattr(batches, 'prepared_batches', inputs)
    tagger = SimpleNamespace(settings=SimpleNamespace(source=WD), labels=['tag'], predict_batch=predict)
    with closing(batches.BatchPredictions(None, tagger, None, {}, [], prefetch=1, result_prefetch=1)) as stream:
        try:
            assert entered.wait(5)
            assert stream.buffer_usage()['result'] == BufferUsage(0, 1, 3)
        finally:
            release.set()
        if failure:
            with pytest.raises(ValueError, match='model failed'):
                next(stream)
        else:
            assert next(stream)[1] == ['a']
    assert stream.buffer_usage()['result'] == BufferUsage(0, 0, 3)


def test_chunk_metrics_preserve_full_histograms_and_scene_errors():
    rows = metric_rows()
    results = [WorkerResult(str(i), {'value': i} if i != 2 else None,
                            'bad scene' if i == 2 else None, rows) for i in range(8)]
    chunk = pack_chunk(results)
    assert pickle.loads(pickle.dumps(chunk)) == chunk
    assert chunk.results[2].error == 'bad scene' and chunk.results[2].payload is None
    assert all(result.metrics is None for result in chunk.results)
    expected = ProfileMetrics()
    for result in results:
        expected.ingest(result.metrics)
    assert chunk.metrics == expected.export()
    assert list(validated_chunks(tuple(str(i) for i in range(8)), [chunk], 8)) == [chunk]


@pytest.mark.parametrize('kind', ['unknown', 'duplicate', 'missing', 'reordered', 'extra_chunk', 'missing_chunk', 'wrong_type'])
def test_chunk_validation_rejects_incomplete_or_inconsistent_scene_sets(kind):
    rows = [WorkerResult('1'), WorkerResult('2')]
    if kind == 'unknown': rows[1] = WorkerResult('3')
    if kind == 'duplicate': rows[1] = rows[0]
    if kind == 'missing': rows.pop()
    if kind == 'reordered': rows.reverse()
    chunks = [WorkerChunk(tuple(rows))]
    if kind == 'extra_chunk': chunks.append(chunks[0])
    if kind == 'missing_chunk': chunks.clear()
    if kind == 'wrong_type': chunks = [rows]
    with pytest.raises(RuntimeError, match='scene'):
        list(validated_chunks(('1', '2'), chunks, 2))


def test_multiple_worker_chunks_match_serial_counters_and_merge_evidence(tmp_path):
    from stash_ai.reports.model.report import build_report
    from stash_ai.evidence.pipeline import evidence
    path, scope, policies = tmp_path/'state.db', 'scope', empty_policies()
    scenes = [str(i) for i in range(16)]
    with StateStore(path, scope) as store:
        for scene in scenes:
            store.put(scene, 'wd', _outputs('wd', 'tag-'+scene))
    before = path.read_bytes()
    with StateStore(path, scope, True) as store:
        serial_report = build_report(app, store, scenes, policies, policies)
        serial_evidence = {}
        for scene in scenes:
            diagnostics = Counter()
            hits, spans = evidence(app, store, scene, policies, policies, diagnostics)
            serial_evidence[scene] = hits, spans, diagnostics
    serial_profile = ProfileMetrics()
    observed_serial = build_report_parallel(app, path, scope, scenes, policies, policies,
        workers=1, cache_bytes=0, memory_budget_bytes=128*2**20, profile=serial_profile)
    assert observed_serial == serial_report
    profile = ProfileMetrics()
    parallel_report = build_report_parallel(app, path, scope, scenes, policies, policies,
        workers=2, cache_bytes=0, memory_budget_bytes=128*2**20, profile=profile)
    assert parallel_report == serial_report
    compute = [row for row in profile.export() if row['name'] == 'model_report_worker_compute']
    assert len(compute) == 2 and all(row['calls'] == row['units'] == 16 for row in compute)
    logical = lambda rows: [(row['kind'], row['name'], row['outcome'], row['calls'], row['units'], row['errors']) for row in rows]
    assert logical(profile.export()) == logical(serial_profile.export())
    results = list(iter_parallel_evidence(app, path, scope, scenes, policies, policies,
        workers=2, cache_bytes=0, memory_budget_bytes=128*2**20, profile=True))
    assert [result.scene_id for result in results] == scenes
    assert sum(result.metrics is not None for result in results) == 8
    aggregate = ProfileMetrics()
    for result in results:
        assert result.error is None
        assert (result.payload.scene_hits, result.payload.spans, result.payload.diagnostics) == serial_evidence[result.scene_id]
        if result.metrics is not None:
            aggregate.ingest(result.metrics)
    assert all(row['calls'] == row['units'] == 16 for row in aggregate.export()
               if row['name'] == 'merge_worker_compute')
    assert path.read_bytes() == before
